"""Run the evaluation set against the live agent.

    python -m eval.run_eval --no-judge --workers 2   # crucial cases, fits the free tier
    python -m eval.run_eval --all        # all 18 cases, with LLM judge (needs more quota)
    python -m eval.run_eval --only refusal,mixed-laptop-return   # ids or categories
    python -m eval.run_eval --workers 2   # fewer parallel cases if rate-limited
    python -m eval.run_eval --resume      # re-run only the cases the last run couldn't score

By default only the cases marked "crucial" in questions.json run: 6 cases, one
per behaviour the agent must get right (grounded answer, order lookup, unknown
order, order + policy, two kinds of refusal). Without the judge that is about
12-14 Gemini requests, inside the AI Studio free tier's 20 requests per model
per day. The summary prints how many requests the run actually used.

Each case is scored two ways:
  * Deterministic checks (eval/checks.py): refusal correctness, expected tool
    calls, expected citation, citations point to retrieved passages, key facts.
  * LLM-as-judge: compares the answer to a reference answer and to the
    evidence the tools returned, and rates correctness and groundedness.
A case passes only if every applicable check and the judge pass.
Results are written to eval/results/<timestamp>.json for comparing runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "questions.json").read_text())

# Pin "today" before the agent reads it, so date-based answers are reproducible.
os.environ.setdefault("NOVAMART_TODAY", CASES["today"])

from novamart import config, quota  # noqa: E402
from novamart.agent import SupportAgent, Turn  # noqa: E402
from eval.checks import run_checks  # noqa: E402

JUDGE_PROMPT = """\
You are grading an internal retail support assistant.

Question: {question}

Reference answer (written by a human, ground truth): {reference}

Evidence the assistant retrieved with its tools (JSON):
{evidence}

Assistant's answer:
{answer}

Grade two things:
- correct: does the answer agree with the reference answer on the facts that
  matter (decision, numbers, conditions)? Extra correct detail is fine. If the
  reference says the assistant should decline, it is correct only if it declined.
- grounded: is every factual claim in the answer supported by the evidence
  above (or is it a refusal that makes no factual claims)? Claims from outside
  knowledge are NOT grounded, even if true.
Give a one-sentence reason."""


class Verdict(BaseModel):
    correct: bool
    grounded: bool
    reason: str


def judge(client, case: dict, turn: Turn) -> Verdict:
    from google.genai import types

    evidence = [{"tool": c.name, "args": c.args, "result": c.result} for c in turn.tool_calls]
    prompt = JUDGE_PROMPT.format(
        question=case["question"],
        reference=case["reference"],
        evidence=json.dumps(evidence, indent=1)[:12000],
        answer=turn.answer,
    )
    response = quota.call(
        config.JUDGE_MODEL,
        lambda: client.models.generate_content(
            model=config.JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=Verdict
            ),
        ),
    )
    # If the judge's JSON can't be parsed, count it as a failure rather than crash.
    return response.parsed or Verdict(correct=False, grounded=False, reason="unparseable judge output")


def mark(value: bool | None) -> str:
    return {True: "ok", False: "FAIL", None: "-"}[value]


# Set once the daily free-tier quota is gone: remaining cases are skipped,
# since no request can succeed until the quota resets.
QUOTA_EXHAUSTED = threading.Event()
QUOTA_MESSAGE: list[str] = []  # the first QuotaExhaustedError message, for the summary

# Progress output, so a long run never looks stuck: which case each worker
# thread is on, and a heartbeat while no case has finished.
CURRENT = threading.local()
HEARTBEAT_SECONDS = 30


def announce_wait(reason: str, seconds: float) -> None:
    case_id = getattr(CURRENT, "case_id", "?")
    print(f"  ... {case_id}: waiting {seconds:.0f}s ({reason})", flush=True)


# Model overload across cases. quota.call() already retries a 503 for about 2
# minutes; if a case still fails on overload, pause before starting the next
# case so the spike can pass. If the next case is overloaded too, stop sending
# requests (they would only use up quota) and leave the rest for --resume.
OVERLOAD_PAUSE_SECONDS = 180
OVERLOADED_CASES_BEFORE_STOP = 2  # consecutive overloaded cases
OVERLOAD_LOCK = threading.Lock()
OVERLOAD = {"paused_until": 0.0, "streak": 0}
OVERLOAD_STOP = threading.Event()
OVERLOAD_SKIP_MESSAGE = "model still overloaded; stopped to save quota (re-run later with --resume)"

# Statuses of cases that got no result; --resume runs these again.
UNSCORED = ("ERROR", "QUOTA", "OVERLOADED", "SKIPPED")


def note_overload(exc: Exception) -> None:
    with OVERLOAD_LOCK:
        OVERLOAD["streak"] += 1
        if OVERLOAD["streak"] >= OVERLOADED_CASES_BEFORE_STOP:
            if not OVERLOAD_STOP.is_set():
                OVERLOAD_STOP.set()
                print(f"\n!!! MODEL OVERLOADED: {exc}\n!!! {OVERLOAD['streak']} cases in a row failed on "
                      "overload; skipping the remaining cases. Re-run later with --resume.", flush=True)
        else:
            OVERLOAD["paused_until"] = time.monotonic() + OVERLOAD_PAUSE_SECONDS
            print(f"\n!!! MODEL OVERLOADED: {exc}\n!!! Pausing {OVERLOAD_PAUSE_SECONDS}s before "
                  "the next case.", flush=True)


def note_success() -> None:
    with OVERLOAD_LOCK:
        OVERLOAD["streak"] = 0


def timing(log: dict | None) -> dict:
    """Where a case's time went, from the API calls and retries quota.call recorded.

    api_seconds includes the SDK's own retry waits (they happen inside a call);
    rate_limit_wait_seconds and overload_wait_seconds are time spent sleeping
    between calls on per-minute 429s and on 503 overloads.
    """
    log = log or {"calls": [], "retries": []}
    return {
        "api_calls": len(log["calls"]),
        "api_seconds": round(sum(c["seconds"] for c in log["calls"]), 1),
        "rate_limit_wait_seconds": round(
            sum(r["wait_seconds"] for r in log["retries"] if r["by"] == "rate_limit"), 1
        ),
        "overload_wait_seconds": round(
            sum(r["wait_seconds"] for r in log["retries"] if r["by"] == "overload"), 1
        ),
        "retries": log["retries"],
        "calls": log["calls"],
    }


def unscored(case: dict, status: str, message: str, seconds: float = 0.0, log: dict | None = None) -> dict:
    """Result for a case that could not be scored (API error or skipped)."""
    t = timing(log)
    # The case may have made chat/judge requests before it failed; count them.
    requests = sum(1 for c in t["calls"] if c["model"] != config.EMBED_MODEL)
    return {
        "id": case["id"], "category": case["category"], "passed": False,
        "status": status, "error": message, "seconds": round(seconds, 1),
        "checks": {}, "answer": "", "cited": [], "tool_calls": [],
        "requests": requests, "judge": None, "timing": t,
    }


def run_case(agent: SupportAgent, case: dict, use_judge: bool) -> dict:
    """Answer one case in a fresh conversation, then score it.

    API errors are recorded as an ERROR result rather than raised, so one
    failure does not lose the results of the other cases.
    """
    if QUOTA_EXHAUSTED.is_set():
        return unscored(case, "SKIPPED", "daily Gemini quota exhausted earlier in this run")
    if OVERLOAD_STOP.is_set():
        return unscored(case, "SKIPPED", OVERLOAD_SKIP_MESSAGE)
    CURRENT.case_id = case["id"]  # lets progress messages name the case
    pause = OVERLOAD["paused_until"] - time.monotonic()
    if pause > 0:
        print(f"  ... {case['id']}: model overloaded, pausing {pause:.0f}s before starting", flush=True)
        time.sleep(pause)
        if OVERLOAD_STOP.is_set():  # another worker gave up while we waited
            return unscored(case, "SKIPPED", OVERLOAD_SKIP_MESSAGE)
    print(f"  ... started {case['id']}", flush=True)
    started = time.perf_counter()
    quota.start_tracking()  # records this case's API calls and retries (this thread only)
    try:
        turn = agent.ask(case["question"])
        checks = run_checks(case, turn)
        verdict = judge(agent.client, case, turn) if use_judge else None
    except quota.QuotaExhaustedError as exc:
        if not QUOTA_EXHAUSTED.is_set():
            QUOTA_EXHAUSTED.set()
            QUOTA_MESSAGE.append(str(exc))
            print(f"\n!!! QUOTA EXHAUSTED: {exc}\n!!! Skipping the remaining cases.", flush=True)
        return unscored(case, "QUOTA", str(exc), time.perf_counter() - started, quota.stop_tracking())
    except quota.ModelOverloadedError as exc:
        note_overload(exc)
        return unscored(case, "OVERLOADED", str(exc), time.perf_counter() - started, quota.stop_tracking())
    except Exception as exc:  # noqa: BLE001 - report any other API failure per case
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        return unscored(case, "ERROR", message, time.perf_counter() - started, quota.stop_tracking())
    log = quota.stop_tracking()
    note_success()
    if verdict:
        checks["judge_correct"] = verdict.correct
        checks["judge_grounded"] = verdict.grounded
    passed = all(v is not False for v in checks.values())
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "seconds": round(time.perf_counter() - started, 1),
        "checks": checks,
        "answer": turn.answer,
        "cited": sorted(turn.cited_sources),
        "tool_calls": [c.__dict__ for c in turn.tool_calls],
        "requests": turn.model_calls + (1 if use_judge else 0),
        "judge": verdict.model_dump() if verdict else None,
        "timing": timing(log),
    }


def timing_lines(t: dict) -> list[str]:
    calls = ", ".join(f"{c['model']} {c['seconds']}s" for c in t["calls"])
    lines = [f"  time in API: {t['api_seconds']}s over {t['api_calls']} calls ({calls})"]
    if t["retries"]:
        waits = "; ".join(f"{r['by']}: {r['reason']} (waited {r['wait_seconds']}s)" for r in t["retries"])
        lines.append(f"  retries: {len(t['retries'])} - {waits}")
    else:
        lines.append("  retries: none")
    return lines


def report(r: dict, done: int, total: int) -> str:
    header = f"\n[{done}/{total}] {r['status']} {r['id']} ({r['category']}, {r['seconds']}s)"
    if r["status"] == "SKIPPED":
        return f"{header}\n  {r['error']}"
    if r["status"] in ("ERROR", "QUOTA", "OVERLOADED"):
        return "\n".join([header, f"  {r['error']}", *timing_lines(r["timing"])])
    lines = [
        header,
        "  " + "  ".join(f"{k}={mark(v)}" for k, v in r["checks"].items()),
        f"  tools: {[c['name'] for c in r['tool_calls']]}  cited: {r['cited']}  "
        f"gemini requests: {r['requests']}",
        *timing_lines(r["timing"]),
    ]
    if r["judge"]:
        lines.append(f"  judge: {r['judge']['reason']}")
    if not r["passed"]:
        lines.append(f"  answer: {r['answer'][:500]}")
    return "\n".join(lines)


def run_settings(args: argparse.Namespace, selection: str) -> dict:
    """Every setting that changes the agent's answers or speed, so the results
    file says exactly what was run (and --resume can refuse to mix configurations)."""
    return {
        "model": config.CHAT_MODEL,
        "judge_model": None if args.no_judge else config.JUDGE_MODEL,
        "thinking_budget": config.THINKING_BUDGET,  # None = model default
        "embed_model": config.EMBED_MODEL, "embed_dim": config.EMBED_DIM,
        "top_k": config.TOP_K, "min_score": config.MIN_SCORE,
        "today": CASES["today"], "selection": selection, "workers": args.workers,
    }


def latest_results_file() -> Path:
    files = sorted((HERE / "results").glob("*.json"))
    if not files:
        raise SystemExit("--resume: there is no results file in eval/results/ to resume from.")
    return files[-1]


def resume_plan(previous: dict, settings: dict) -> tuple[list[dict], list[dict]]:
    """For --resume: (cases to run again, results to keep) from a previous run.

    Only cases that got no result (UNSCORED) run again; PASS and FAIL are real
    results and are kept. Refuses if the settings differ, since the merged file
    would then mix two configurations.
    """
    old = previous.get("settings")
    if old is None:
        raise SystemExit("--resume: that results file was written before settings were recorded, "
                         "so it can't be resumed safely. Run the eval without --resume.")
    differing = [k for k in sorted(set(old) | set(settings))
                 if k not in ("workers", "selection") and old.get(k) != settings.get(k)]
    if differing:
        details = ", ".join(f"{k}: was {old.get(k)!r}, now {settings.get(k)!r}" for k in differing)
        raise SystemExit(f"--resume: settings differ from the previous run ({details}). "
                         "Use the same .env and flags (e.g. --no-judge), or run without --resume.")
    keep = [r for r in previous["results"] if r["status"] not in UNSCORED]
    redo = {r["id"] for r in previous["results"] if r["status"] in UNSCORED}
    return [c for c in CASES["cases"] if c["id"] in redo], keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge (about 1/3 fewer calls)")
    parser.add_argument("--all", action="store_true",
                        help="run all cases, not just the crucial ones (needs more than the free daily quota)")
    parser.add_argument("--only", help="comma-separated case ids or categories, e.g. refusal,tool-in-transit")
    parser.add_argument("--workers", type=int, default=4,
                        help="cases run in parallel (default 4; use 1-2 if you hit free-tier rate limits)")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="RESULTS_FILE",
                        help="re-run only the cases that got no result (ERROR, QUOTA, OVERLOADED, SKIPPED) "
                             "in a previous results file (default: the newest) and write a merged file")
    args = parser.parse_args()
    if args.resume and (args.only or args.all):
        parser.error("--resume re-runs the cases of a previous run; it can't be combined with --only or --all")

    kept: list[dict] = []
    resumed_from = None
    if args.resume:
        path = latest_results_file() if args.resume == "latest" else Path(args.resume)
        previous = json.loads(path.read_text())
        selection = previous.get("selection", "crucial")
        settings = run_settings(args, selection)
        cases, kept = resume_plan(previous, settings)
        resumed_from = path.name
        print(f"Resuming {path.name}: keeping {len(kept)} result(s), re-running {len(cases)} case(s).")
        if not cases:
            print("Nothing to resume: every case in that file already has a result.")
            return
    else:
        wanted = set(args.only.split(",")) if args.only else None
        if wanted:
            cases = [c for c in CASES["cases"] if c["id"] in wanted or c["category"] in wanted]
        elif args.all:
            cases = CASES["cases"]
        else:
            cases = [c for c in CASES["cases"] if c.get("crucial")]
        selection = "selected" if wanted else ("all" if args.all else "crucial")
        settings = run_settings(args, selection)

    # One client and one policy index, shared by all cases (both are read-only
    # here). Each case gets its own SupportAgent, i.e. its own conversation.
    print("Loading the policy index (embeds the documents only if they changed)...", flush=True)
    base = SupportAgent.create()
    quota.track_sdk_retries()
    quota.on_wait = announce_wait
    print(f"Running {len(cases)} {selection} cases with {args.workers} workers, model={config.CHAT_MODEL}, "
          f"judge={'off' if args.no_judge else config.JUDGE_MODEL}, "
          f"thinking_budget={config.THINKING_BUDGET if config.THINKING_BUDGET is not None else 'model default'}")
    run_started_at = dt.datetime.now()
    started = time.perf_counter()

    new_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_case, SupportAgent(base.client, base.toolbox, base.model), case, not args.no_judge)
            for case in cases
        ]
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
            if not finished:
                print(f"  ... still working: {len(new_results)}/{len(cases)} cases done, "
                      f"{time.perf_counter() - started:.0f}s elapsed", flush=True)
            for future in finished:
                new_results.append(future.result())
                print(report(new_results[-1], len(new_results), len(cases)), flush=True)

    order = {c["id"]: i for i, c in enumerate(CASES["cases"])}
    results = sorted(kept + new_results, key=lambda r: order[r["id"]])  # stable order in the file
    total_seconds = time.perf_counter() - started
    print(f"\nTotal time: {total_seconds:.0f}s")

    # Summary per category, then overall (including results kept by --resume).
    print("\n=== Summary ===" + (f" (includes {len(kept)} result(s) kept from {resumed_from})" if kept else ""))
    for category in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == category]
        print(f"{category:10s} {sum(r['passed'] for r in rows)}/{len(rows)}")
    total = sum(r["passed"] for r in results)
    print(f"{'overall':10s} {total}/{len(results)}")
    requests_used = sum(r["requests"] for r in new_results)
    print(f"\nGemini requests used by this run: {requests_used} "
          f"(free tier: about 20 per model per day; embeddings are cached)")
    retries = [x for r in new_results for x in r["timing"]["retries"]]
    api_calls = sum(r["timing"]["api_calls"] for r in new_results)
    by = {kind: sum(x["by"] == kind for x in retries) for kind in ("sdk", "rate_limit", "overload")}
    print(f"API calls incl. query embeddings: {api_calls}; retries: {len(retries)} "
          f"(sdk {by['sdk']}, rate limit {by['rate_limit']}, overload {by['overload']}), "
          f"{sum(x['wait_seconds'] for x in retries):.0f}s spent waiting")
    unscored_count = sum(r["status"] in UNSCORED for r in results)
    if unscored_count:
        print(f"Not scored: {unscored_count} case(s) hit API errors or were skipped; "
              "`python -m eval.run_eval --resume` (same flags) re-runs only those.")
    if QUOTA_EXHAUSTED.is_set():
        print(f"\n!!! QUOTA EXHAUSTED: {QUOTA_MESSAGE[0]}")
    if OVERLOAD_STOP.is_set():
        print("\n!!! MODEL OVERLOADED: the run was stopped early. Try again later with --resume.")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out_file.write_text(
        json.dumps(
            {"model": config.CHAT_MODEL, "judge_model": config.JUDGE_MODEL,
             "min_score": config.MIN_SCORE, "today": CASES["today"], "selection": selection,
             "settings": settings, "started_at": run_started_at.isoformat(timespec="seconds"),
             "resumed_from": resumed_from,
             "total_seconds": round(total_seconds, 1),
             "requests": sum(r["requests"] for r in results),
             "results": results},
            indent=2,
        )
    )
    print(f"\nDetailed results: {out_file.relative_to(HERE.parent)}")
    if resumed_from:
        print(f"(merged with {resumed_from}; that file is unchanged and can be deleted)")


if __name__ == "__main__":
    main()

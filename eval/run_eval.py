"""Run the evaluation set against the live agent.

    python -m eval.run_eval --no-judge --workers 2   # crucial cases, fits the free tier
    python -m eval.run_eval --all        # all 18 cases, with LLM judge (needs more quota)
    python -m eval.run_eval --only refusal,mixed-laptop-return   # ids or categories
    python -m eval.run_eval --workers 2   # fewer parallel cases if rate-limited

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


def timing(log: dict | None) -> dict:
    """Where a case's time went, from the API calls and retries quota.call recorded.

    api_seconds includes the SDK's own retry waits (they happen inside a call);
    rate_limit_wait_seconds is time spent sleeping between calls on per-minute 429s.
    """
    log = log or {"calls": [], "retries": []}
    return {
        "api_calls": len(log["calls"]),
        "api_seconds": round(sum(c["seconds"] for c in log["calls"]), 1),
        "rate_limit_wait_seconds": round(
            sum(r["wait_seconds"] for r in log["retries"] if r["by"] == "rate_limit"), 1
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
    CURRENT.case_id = case["id"]  # lets progress messages name the case
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
    except Exception as exc:  # noqa: BLE001 - report any other API failure per case
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        return unscored(case, "ERROR", message, time.perf_counter() - started, quota.stop_tracking())
    log = quota.stop_tracking()
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
    if r["status"] in ("ERROR", "QUOTA"):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge (about 1/3 fewer calls)")
    parser.add_argument("--all", action="store_true",
                        help="run all cases, not just the crucial ones (needs more than the free daily quota)")
    parser.add_argument("--only", help="comma-separated case ids or categories, e.g. refusal,tool-in-transit")
    parser.add_argument("--workers", type=int, default=4,
                        help="cases run in parallel (default 4; use 1-2 if you hit free-tier rate limits)")
    args = parser.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    if wanted:
        cases = [c for c in CASES["cases"] if c["id"] in wanted or c["category"] in wanted]
    elif args.all:
        cases = CASES["cases"]
    else:
        cases = [c for c in CASES["cases"] if c.get("crucial")]

    # One client and one policy index, shared by all cases (both are read-only
    # here). Each case gets its own SupportAgent, i.e. its own conversation.
    print("Loading the policy index (embeds the documents only if they changed)...", flush=True)
    base = SupportAgent.create()
    quota.track_sdk_retries()
    quota.on_wait = announce_wait
    selection = "selected" if wanted else ("all" if args.all else "crucial")
    # Every setting that changes the agent's answers or speed, so the results
    # file says exactly what was run.
    settings = {
        "model": config.CHAT_MODEL,
        "judge_model": None if args.no_judge else config.JUDGE_MODEL,
        "thinking_budget": config.THINKING_BUDGET,  # None = model default
        "embed_model": config.EMBED_MODEL, "embed_dim": config.EMBED_DIM,
        "top_k": config.TOP_K, "min_score": config.MIN_SCORE,
        "today": CASES["today"], "selection": selection, "workers": args.workers,
    }
    print(f"Running {len(cases)} {selection} cases with {args.workers} workers, model={config.CHAT_MODEL}, "
          f"judge={'off' if args.no_judge else config.JUDGE_MODEL}, "
          f"thinking_budget={config.THINKING_BUDGET if config.THINKING_BUDGET is not None else 'model default'}")
    run_started_at = dt.datetime.now()
    started = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_case, SupportAgent(base.client, base.toolbox, base.model), case, not args.no_judge)
            for case in cases
        ]
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
            if not finished:
                print(f"  ... still working: {len(results)}/{len(cases)} cases done, "
                      f"{time.perf_counter() - started:.0f}s elapsed", flush=True)
            for future in finished:
                results.append(future.result())
                print(report(results[-1], len(results), len(cases)), flush=True)

    order = {c["id"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order[r["id"]])  # stable order in the results file
    total_seconds = time.perf_counter() - started
    print(f"\nTotal time: {total_seconds:.0f}s")

    # Summary per category, then overall.
    print("\n=== Summary ===")
    for category in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == category]
        print(f"{category:10s} {sum(r['passed'] for r in rows)}/{len(rows)}")
    total = sum(r["passed"] for r in results)
    print(f"{'overall':10s} {total}/{len(results)}")
    requests_used = sum(r["requests"] for r in results)
    print(f"\nGemini requests used: {requests_used} "
          f"(free tier: about 20 per model per day; embeddings are cached)")
    retries = [x for r in results for x in r["timing"]["retries"]]
    api_calls = sum(r["timing"]["api_calls"] for r in results)
    print(f"API calls incl. query embeddings: {api_calls}; retries: {len(retries)} "
          f"(sdk {sum(x['by'] == 'sdk' for x in retries)}, rate limit {sum(x['by'] == 'rate_limit' for x in retries)}), "
          f"{sum(x['wait_seconds'] for x in retries):.0f}s spent waiting")
    errors = [r for r in results if r["status"] in ("ERROR", "QUOTA", "SKIPPED")]
    if errors:
        print(f"Not scored: {len(errors)} case(s) hit API errors or were skipped.")
    if QUOTA_EXHAUSTED.is_set():
        print(f"\n!!! QUOTA EXHAUSTED: {QUOTA_MESSAGE[0]}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out_file.write_text(
        json.dumps(
            {"model": config.CHAT_MODEL, "judge_model": config.JUDGE_MODEL,
             "min_score": config.MIN_SCORE, "today": CASES["today"], "selection": selection,
             "settings": settings, "started_at": run_started_at.isoformat(timespec="seconds"),
             "total_seconds": round(total_seconds, 1), "requests": requests_used,
             "results": results},
            indent=2,
        )
    )
    print(f"\nDetailed results: {out_file.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()

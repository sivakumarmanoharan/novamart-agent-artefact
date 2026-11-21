# NovaMart Internal Support Agent (prototype)

A small agent that answers NovaMart store employees' questions about policies,
returns and orders. It is grounded in NovaMart's policy documents, cites the
section it used, looks up order status through a mock order API, and declines
when the documents don't cover the question.

Built with the **Google Gen AI SDK** (`google-genai`) and **Gemini** via a
Google AI Studio API key, with no agent framework. The agent core is about 670
lines of commented Python (about 920 with the CLI and web UI). You can use it
from the command line or a small browser UI.

**Status.** 22 offline tests pass. All 6 crucial eval cases pass every
deterministic check and both LLM-judge checks against the live API (see
[Results](#results)). One known quality
issue, which neither the checks nor the judge catch: the order-plus-policy
answer reaches the right conclusion but opens with the wrong one and corrects
itself. The 12 non-crucial cases have not been run (see [Limits](#limits)).

```mermaid
flowchart LR
    U["Store employee<br/>novamart/cli.py or novamart/web.py"] -->|question| G["Gemini<br/>system prompt + 2 tools<br/>novamart/agent.py"]
    G -->|function call| TB["Toolbox<br/>novamart/tools.py"]
    TB -->|search_policies| IDX["PolicyIndex<br/>cosine top-k, drop below MIN_SCORE<br/>novamart/retrieval.py"]
    TB -->|get_order_status| ORD[("data/orders.json<br/>mock order API")]
    IDX -->|"passages, or NO_RELEVANT_POLICY"| TB
    ORD -->|"order record, or not found"| TB
    TB -->|function response| G
    DOCS["data/policies/*.md<br/>one chunk per section"] -.->|"Gemini embeddings<br/>cached in .cache/"| IDX
    G -->|final answer| OUT["Answer with source citations,<br/>or fixed refusal + contact Operations"]
    OUT --> U
```

Diagram sources: `docs/architecture.mmd`, `docs/eval.mmd`.

## Setup

Requires Python 3.10+ and a free Gemini API key from https://aistudio.google.com/apikey.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                 # Windows: Copy-Item .env.example .env
# then put your key in .env (it is git-ignored)

python -m pytest -q                 # 22 offline tests, no key needed
python -m novamart.cli --trace      # chat; --trace shows tool calls
python -m novamart.cli -q "Can a customer return an opened laptop?"
python -m novamart.web              # browser UI at http://127.0.0.1:8000 (--port to change)
python -m eval.run_eval --no-judge --workers 2   # 6 crucial cases, fits the free tier
python -m eval.run_eval --all       # all 18 cases + LLM judge (needs more than the free quota)
python -m eval.run_eval --resume    # re-run only the cases the last run couldn't score (same flags)
```

The key is read from `.env` (via `python-dotenv`) or from a `GEMINI_API_KEY`
environment variable, which takes precedence. The first run embeds the policy
documents in one batched call and caches the vectors in `.cache/`; they are
re-embedded automatically when a document or the embedding settings change.

Things to try: `Where is order NM-10002?`, `The customer on NM-10001 wants to
return their opened laptop today. Eligible?`, `What's our policy on dogs in the
store?`. Mock orders are `NM-10001` to `NM-10005` in `data/orders.json`.

## Configuration

Every setting is in `novamart/config.py` and can be overridden with an
environment variable (in the shell or in `.env`):

| Variable | Default | What it controls |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.5-flash` | Chat model the agent uses ([why](#key-decisions)) |
| `GEMINI_JUDGE_MODEL` | `gemini-3.5-flash-lite` | Model used as LLM-as-judge in the eval |
| `GEMINI_THINKING_BUDGET` | unset (model default) | Cap on the model's thinking tokens. The eval runs used `0` ([why](#key-decisions)) |
| `GEMINI_EMBED_MODEL` / `GEMINI_EMBED_DIM` | `gemini-embedding-001` / `768` | Embedding model and size |
| `NOVAMART_TOP_K` | `4` | Passages returned per search |
| `NOVAMART_MIN_SCORE` | `0.60` | Minimum cosine similarity for a passage to count as relevant |
| `NOVAMART_TODAY` | today's date | The date the agent assumes (the eval pins it) |

The agent loop is also capped at 5 model/tool round trips per question
(`MAX_TOOL_STEPS`).

## Web UI

`python -m novamart.web` serves a single-page chat on `127.0.0.1:8000` using
only Python's standard library, so it is reachable from this machine only. Each
page load is its own conversation. Answers show citations as tags, with any
citation the search never returned flagged as unsupported; refusals are marked
"Declined — not in documentation"; and a collapsible trace under each answer
shows every tool call and its result. The sidebar has example questions, the
mock orders, and the knowledge base: every policy section with its citation
id, which opens to show the section's full text exactly as the agent searches
it. Clicking a citation tag in an answer opens that section, so an answer can
be checked against its source in one click.

| Endpoint | What it does |
|---|---|
| `GET /api/info` | Model, today's date, threshold, examples, mock orders, policy sections with their full text |
| `POST /api/ask` | `{"session", "question"}` → answer, refused flag, citations, unsupported citations, tool calls |
| `POST /api/reset` | `{"session"}` → clears that conversation |

Request bodies are limited to 20 KB, questions to 2,000 characters, and the
server keeps at most 200 conversations in memory.

## Errors and the free-tier quota

Every Gemini call goes through `novamart/quota.py`:

- **Model overloaded** (HTTP 503, "the model is experiencing high demand"):
  this is load on Google's side for that model, not a limit on the key or
  project, so a different key doesn't help. The agent waits 5, 15, 45 and 60 s
  between attempts, about 2 minutes in total, then reports
  `!!! MODEL OVERLOADED`. The SDK's own retries are only a few seconds apart,
  too short for a demand spike.
- **Other server errors and hung requests** (408, 500, 502, 504, or no
  response within 120 s): the SDK retries up to 3 attempts with backoff.
  Without the 120 s limit, a request the API never answers would block a run
  indefinitely.
- **Per-minute rate limits** (HTTP 429): wait the delay the API suggests (at
  most 60 s) and retry, up to 5 times.
- **Daily quota used up** (HTTP 429, per-day quota): retrying can't help, so
  the app says so at once. The web UI shows a "Gemini quota exhausted" banner
  with the time until reset; the CLI and eval print `!!! QUOTA EXHAUSTED: ...`.

AI Studio's free tier allows about 20 requests per model per day. A question
usually takes 2 chat-model requests (the model asks for its tools, then
answers), up to 5 if it searches again, plus one embedding request per search
on the embedding model's separate quota. That is roughly 10 questions a day.
When the quota is used up, wait for the reset, switch `GEMINI_MODEL` (quotas
are per model), or enable billing.

## Project layout

| Path | What it is |
|---|---|
| `data/policies/*.md` | 7 invented NovaMart policy documents, 37 sections in total |
| `data/orders.json` | Mock order database behind the order-status tool (5 orders) |
| `novamart/config.py` | Every tunable setting |
| `novamart/retrieval.py` | Chunking, Gemini embeddings, cosine search, embedding cache |
| `novamart/tools.py` | The two tools and their declarations as shown to Gemini |
| `novamart/agent.py` | System prompt, citation and refusal parsing, the tool-calling loop |
| `novamart/quota.py` | Retry and quota handling for every Gemini call, plus per-call timing for the eval |
| `novamart/cli.py`, `novamart/web.py`, `novamart/static/index.html` | Command-line chat and browser UI |
| `eval/questions.json` | 18 test cases with reference answers and expectations |
| `eval/checks.py`, `eval/run_eval.py` | Deterministic checks, LLM-as-judge, report |
| `eval/results/*.json` | One file per eval run |
| `tests/test_offline.py`, `tests/test_quota.py`, `tests/test_eval_resilience.py` | Unit tests: chunking, tools, citations, eval checks, the agent loop against a scripted fake Gemini, quota and overload handling, the eval's pause/stop policy and `--resume` |
| `PRODUCTION_NOTE.md` | How I would take this to production on Google Cloud |

## Sample answers
<img width="1656" height="855" alt="image" src="https://github.com/user-attachments/assets/a71cd40b-5c6f-436d-9828-c5f856c9fdea" />
Correct answers
<img width="1657" height="351" alt="image" src="https://github.com/user-attachments/assets/999c8540-d727-4f78-9ece-76a56e38024e" />
Not in policy documents
<img width="1650" height="371" alt="image" src="https://github.com/user-attachments/assets/5ea84fcc-322a-48bb-954e-7108ddeea78e" />
Scam reports


## Key decisions

1. **Google Gen AI SDK with a hand-written tool loop, not ADK.** The loop in
   `agent.py` is about 40 lines, every step is visible, and it records a trace
   of tool calls that the eval checks. ADK is the right choice for production
   (sessions, deployment to Vertex AI Agent Engine, registration in Gemini
   Enterprise); the tools are plain functions, so porting is straightforward.
2. **Retrieval is a tool the model calls** rather than a step that always
   runs. The model can write a focused query, search twice, skip search for a
   pure order lookup, and combine an order record with policy. The risk is that
   it answers without searching, so the prompt requires a search for policy
   questions and the eval checks that it happened.
3. **One chunk per Markdown section, in-memory NumPy search.** 37 short
   sections don't justify a vector database. Sections are natural citation
   units, and each is embedded with its document title and heading so
   "Eligibility" keeps its context. Embeddings use the `RETRIEVAL_DOCUMENT` /
   `RETRIEVAL_QUERY` task types.
4. **Two layers of refusal.** (a) The search tool drops passages below a
   cosine-similarity threshold and returns an explicit `NO_RELEVANT_POLICY`
   signal when nothing is left. (b) The prompt tells the model to answer only
   from returned passages and, when they don't answer the question, to open
   with a fixed sentence: *"I can't find this in NovaMart's documentation."*
   Layer (b) catches near misses that pass the threshold (e.g. "sick days for
   part-time employees" retrieves the employee-discount doc). The fixed phrase
   makes refusals measurable and points staff to the Operations team.
5. **Citations are section ids and are validated.** Answers cite
   `[source: returns-and-refunds#electronics]` (several ids can share one tag).
   The eval and the web UI check that every cited id was actually returned by
   the search tool, which catches invented citations. The prompt also says
   never to cite a tool and to refer to order details by order id; that rule
   was added after the model cited `[source: get_order_status]` in an eval run.
6. **Tool output is treated as data.** The prompt says so explicitly, the order
   tool validates the id format, and unknown tools return an error instead of
   raising. The loop is capped at 5 round trips and falls back to the refusal.
7. **`gemini-3.5-flash` as the chat model, pinned.** `gemini-2.5-flash` is no
   longer available through the API. A Flash model fits the job: answers are
   short and grounded in retrieved passages, so a Pro model would add latency,
   cost and quota use without a clear benefit. A concrete version rather than
   the `gemini-flash-latest` alias keeps eval runs comparable, and every
   results file records the model. The judge uses `gemini-3.5-flash-lite`:
   grading against a reference answer is easier than answering, and it has its
   own quota.
8. **Thinking turned off for testing (`GEMINI_THINKING_BUDGET=0`).** Private
   reasoning before answering is the main driver of request time. The answers
   need little reasoning (a date comparison, which fee applies), and on a
   free tier of about 20 requests a day, faster requests make it practical to
   change something and re-run. Not yet measured, but a larger budget (e.g.
   1024) would be expected to help the multi-step cases most (the
   self-correcting answer under [Results](#results)), change little for
   lookups and refusals, make every request slower, and bill thinking tokens as
   output on a paid key. To try it for one terminal:
   `$env:GEMINI_THINKING_BUDGET = "1024"; python -m eval.run_eval --only mixed --no-judge`.
   Some models reject `0` (e.g. `gemini-3.5-flash-lite` returns
   `400 INVALID_ARGUMENT`); use a small positive budget there.

## Evaluation

`eval/questions.json` has 18 cases in four categories. By default
`python -m eval.run_eval` runs the 6 marked `"crucial": true`, which fits the
free tier; `--all` runs all 18 and `--only` takes ids or categories.

| Category | # | What it tests |
|---|---|---|
| grounded | 9 | Policy answers, including traps (tablet vs laptop restocking fee, employee discount on electronics, holiday extension) |
| tool | 2 | Order lookup, and a non-existent order that must not be invented |
| mixed | 3 | Order lookup **and** policy reasoning, including a date calculation |
| refusal | 4 | Not in the docs, a competitor's policy (outside knowledge), a near-miss topic, off-topic |

```mermaid
flowchart LR
    Q["eval/questions.json<br/>18 cases"] --> RUN["Agent answers each case<br/>fresh conversation"]
    RUN --> C["Deterministic checks<br/>eval/checks.py<br/>refusal, tools, citations, key facts"]
    RUN --> J["LLM-as-judge<br/>correct + grounded"]
    C --> RES["PASS / FAIL per case<br/>eval/results/*.json"]
    J --> RES
```

Each case passes only if everything applicable passes:

- **Deterministic checks** (`eval/checks.py`): declined exactly when expected;
  called the expected tools; cited an expected section; every citation points to
  a retrieved passage; key facts present (e.g. "15%", "January 31"). Cheap,
  repeatable, and they explain *why* a case failed.
- **LLM-as-judge**: a second model compares the answer with a human-written
  reference answer and with the evidence the tools returned, using structured
  output, and rates `correct` and `grounded` separately. This catches what
  string checks can't, such as a wrong decision that still mentions "15%" or a
  true claim taken from outside knowledge. `--no-judge` skips it.

"Today" is pinned to 2026-11-20 so date-based answers are reproducible. Each
run writes `eval/results/<timestamp>.json` with the run's settings (models,
thinking budget, retrieval settings, workers, total time and requests) and, for
each case, the checks, answer, citations, tool calls, judge reasons, and a
timing block listing every API call and every retry. The file is written even
when a run is cut short:

- **Daily quota used up:** the remaining cases are skipped.
- **Model overloaded:** if a case still fails after the 503 retries, the run
  pauses 3 minutes before the next case. If that case is overloaded too, the
  run stops and skips the rest, because further requests would only use up
  quota.
- **`--resume`** re-runs only the cases from the newest results file that got
  no result (`ERROR`, `QUOTA`, `OVERLOADED`, `SKIPPED`), keeps every `PASS` and
  `FAIL`, and writes a merged file. It refuses to run if the settings differ
  from the original run (model, judge, thinking budget, retrieval settings),
  so a merged file never mixes two configurations. Use the same `.env` and
  flags as the original run.

The 6 crucial cases cover one behaviour each:

| Case | What it proves |
|---|---|
| `grounded-electronics-window` | Grounded policy answer with a citation. The tablet-vs-laptop fee trap is graded only by the judge: the key-fact check just looks for "15". |
| `tool-in-transit` | Calls the order-status tool and reports the record |
| `tool-order-not-found` | Says an unknown order doesn't exist instead of inventing one |
| `mixed-laptop-return` | Combines the order tool with policy search and a date calculation |
| `refuse-pets` | Declines a topic that isn't in the documents |
| `refuse-near-miss-sick-days` | Declines when search returns a similar but non-answering passage |

### Results

2026-11-21, `gemini-3.5-flash`, judge `gemini-3.5-flash-lite`, thinking
budget `0`, 1 worker: **6/6 passed** every deterministic check and both judge
checks (`correct` and `grounded`): grounded 1/1, tool 2/2, mixed 1/1,
refusal 2/2. The run (03:38) scored 5 cases before the daily quota ran out;
`--resume` (03:57) ran the sixth and merged it in. 22 minutes in total, 21 chat
and judge requests. Full output: `eval/results/20261121-040427.json`.
Abridged terminal output:

```text
[1/6] PASS grounded-electronics-window (grounded, 213.4s)
  refusal=ok  tools=ok  cites_source=ok  citations_valid=ok  key_facts=ok  judge_correct=ok  judge_grounded=ok
  judge: The assistant's answer correctly matches the reference answer on all facts and is fully supported by the retrieved policy evidence.
[2/6] PASS tool-in-transit (tool, 232.2s)
  retries: 2 - overload: 503 model overloaded (gemini-3.5-flash) (waited 5.0s); sdk: ServerError: 500 INTERNAL ... (waited 2.5s)
  judge: The assistant's answer accurately reflects all details from the evidence and aligns with the reference answer.
[3/6] PASS tool-order-not-found (tool, 54.7s)
  judge: The assistant correctly communicated that the order was not found based on the tool result and did not invent any false status.
[4/6] PASS mixed-laptop-return (mixed, 229.4s)
  cited: ['returns-and-refunds#electronics', 'returns-and-refunds#holiday-return-extension', 'returns-and-refunds#standard-return-window']
  judge: The assistant correctly concluded that the customer is eligible for a return under the holiday extension policy and subject to a 15% restocking fee, and every claim is fully supported by the retrieved evidence.
[5/6] PASS refuse-pets (refusal, 158.2s)
  judge: The assistant correctly declined to answer due to a lack of information in the documents and appropriately referred the user to the Operations team, perfectly matching the reference answer.

!!! QUOTA EXHAUSTED: Daily Gemini quota exhausted for model 'gemini-3.5-flash' (AI Studio free tier). It resets in about 22h 06m. ...
[6/6] QUOTA refuse-near-miss-sick-days (refusal, 0.1s)
API calls incl. query embeddings: 24; retries: 6 (sdk 3, rate limit 0, overload 3), 23s spent waiting

$ python -m eval.run_eval --workers 1 --resume
Resuming 20261121-035302.json: keeping 5 result(s), re-running 1 case(s).
[1/1] PASS refuse-near-miss-sick-days (refusal, 420.8s)
  refusal=ok  tools=-  cites_source=-  citations_valid=ok  key_facts=-  judge_correct=ok  judge_grounded=ok
  retries: 2 - sdk: ServerError: 504 DEADLINE_EXCEEDED ... (waited 2.4s); sdk: ServerError: 504 DEADLINE_EXCEEDED ... (waited 2.7s)
  judge: The assistant correctly declined to answer because the documents do not cover sick days, and appropriately suggested contacting Operations.

=== Summary === 
overall    6/6
```

What this run shows:

- **The judge agrees with the deterministic checks** on all 6 cases, and its
  reasons name the specific facts it checked, such as no restocking fee on a
  tablet, or that the order wasn't found and no status was invented.
- **The judge missed a self-contradicting answer.** The `mixed-laptop-return`
  answer opens with "**No, the customer is not eligible** for a return",
  stops to say "*Wait, let me double check the math*", and ends with "Yes, they
  are eligible … under the Holiday Return Extension", with the 15% restocking
  fee. Its final conclusion and every fact are right: this time it applied the
  holiday extension (returnable until January 31, 2027) and cited it. The judge
  grades whether the facts that matter are right and grounded, not whether the
  answer is consistent, so it passed the answer, as the deterministic checks
  did. An employee who read only the first line would get the wrong answer.
  This is the most important gap the eval found (see [Limits](#limits)).
- **The citation rule held.** No answer cited a tool.
- **Overload handling worked.** Across both runs there were 3 overload (503)
  waits and 5 SDK retries on 500/504 errors, and every one recovered. Google was
  under heavy load: single requests took up to 220 s (2–6 s on a quiet run),
  which is why the runs took 22 minutes. Judge requests took 1–6 s each.
- **`--resume` worked as designed.** After the quota ran out, it re-ran only
  the missing case (4 requests) instead of all 6, and kept the 5 earlier
  results.

### Limits

- **Not yet run:** the 12 non-crucial cases. `--all` needs about 60
  requests, more than the free tier allows in a day.
- **Nothing checks that an answer is consistent with itself.** Neither the
  deterministic checks nor the judge do, so an answer that opens with the wrong
  conclusion can pass (see Results).
- 18 cases is a smoke test, not a benchmark, and one run doesn't show variance.
- Key-fact checks are substring matches, so some traps are only caught by the
  judge.
- The judge (Flash-Lite) is from the same model family as the agent; set
  `GEMINI_JUDGE_MODEL` to use a different one.

## Assumptions

- Users are **store employees**, not customers, so the prototype has no
  customer authentication, and the order tool returns no personal data.
- NovaMart operates in **Canada** (prices in CAD, a Quebec rule in the gift
  card policy). All documents and orders are invented.
- The policy documents are the single source of truth. If they don't answer a
  question, the right behaviour is to say so and route to the Operations team.
- The order API is **read-only**. Actions such as cancelling or refunding are out
  of scope and would need human confirmation in production.
- English only; one store-wide policy set (no regional variants beyond the text).

## What I would do with more time

- **Fix the self-correcting answers**: test a thinking budget on the mixed
  cases over several runs, and catch the problem in the eval: give the judge a
  third rating, `consistent` (the answer leads with its conclusion and doesn't
  contradict itself). Confirm the citation fix over a few more runs.
- **Calibrate the refusal threshold** on a labelled set of answerable and
  unanswerable questions instead of a reasonable first guess (0.60); the helper
  `python -m novamart.retrieval "<question>"` prints raw scores for this.
- **Grow the eval set** from real historical tickets with the operations team,
  run each case several times to measure variance, add multi-turn and
  prompt-injection cases, tighten key-fact checks so traps are caught without
  the judge, and run it in CI on every change.
- **Hybrid retrieval** (keyword + embeddings) and a reranker, which helps with
  exact terms like SKUs and plan names.
- **Port to ADK**, deploy on Vertex AI Agent Engine, and register the agent in
  Gemini Enterprise (see `PRODUCTION_NOTE.md`).
- **Observability**: OpenTelemetry traces per turn, plus thumbs up/down feedback
  in the web UI that flows back into the eval set.
- **Web UI for real use**: stream answers, and add sign-in (e.g. Google
  Identity / IAP) before serving it beyond `localhost`.
- Handle document versions and conflicts (e.g. which policy date applies to a
  purchase), and regional policy variants via metadata filters.

"""Tell a real (daily) quota exhaustion apart from a short rate limit.

The AI Studio free tier has two kinds of limit, and both come back as HTTP 429:
  * per-minute limits: wait the few seconds the API asks for, then retry;
  * per-day limits (e.g. 20 requests per model per day): retrying cannot help
    until the quota resets, often many hours later.

Every Gemini call goes through `call()`. A per-day 429 raises
QuotaExhaustedError straight away with a message that says which model ran
out and when it resets, so the CLI, web UI and eval can tell the user plainly
instead of hanging on retries or showing a raw API error.

A 503 ("the model is experiencing high demand") is also handled here rather
than by the SDK: `call()` waits longer between attempts (up to about 2 minutes
in total) and then raises ModelOverloadedError.

`call()` also records how long each API call took and every retry (its own
per-minute and overload waits and, with track_sdk_retries(), the SDK's other
5xx retries), so the eval can show where a slow case spent its time.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, TypeVar

from google.genai import errors

T = TypeVar("T")

# Status codes the SDK retries by itself, quickly (a few seconds apart). Two are
# deliberately left out so they reach `call()` below:
#   429: call() knows the difference between per-minute and per-day limits;
#   503 ("model is experiencing high demand"): a demand spike lasts longer than
#        the SDK's short backoff, so call() waits longer between attempts.
SDK_RETRY_STATUS_CODES = [408, 500, 502, 504]

RATE_LIMIT_RETRIES = 5   # per-minute 429s retried at most this many times
MAX_WAIT_SECONDS = 60.0  # never sleep longer than this for a per-minute limit

# Waits before each retry of a 503, so one call rides out about 2 minutes of
# overload (5 attempts) before giving up with ModelOverloadedError.
OVERLOAD_WAITS = [5.0, 15.0, 45.0, 60.0]


class ModelOverloadedError(RuntimeError):
    """The model kept answering 503 UNAVAILABLE (high demand) after every retry."""

    def __init__(self, model: str, attempts: int, waited_seconds: float):
        self.model = model
        super().__init__(
            f"Gemini model '{model}' is overloaded (503 UNAVAILABLE, high demand): "
            f"{attempts} attempts failed over {waited_seconds:.0f}s of waiting. Spikes are "
            "usually temporary: try again later (in the eval, --resume re-runs only the "
            "cases that didn't finish)."
        )


class QuotaExhaustedError(RuntimeError):
    """The daily request quota for a model is used up."""

    def __init__(self, model: str, retry_after_seconds: float | None = None):
        self.model = model
        self.retry_after_seconds = retry_after_seconds
        reset = f" It resets in about {format_duration(retry_after_seconds)}." if retry_after_seconds else ""
        super().__init__(
            f"Daily Gemini quota exhausted for model '{model}' (AI Studio free tier).{reset} "
            "Wait for the reset, set GEMINI_MODEL in .env to another model "
            "(quotas are counted per model), or enable billing on the API key."
        )


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{max(minutes, 1)} min"


def retry_after_seconds(error_text: str) -> float | None:
    """The wait the API suggests, from RetryInfo ('retryDelay': '48766s')."""
    match = re.search(r"retryDelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s", error_text)
    return float(match.group(1)) if match else None


def is_daily_quota(error_text: str) -> bool:
    """True when the 429 is a per-day limit (quotaId like '...PerDay...')."""
    return "PerDay" in error_text or "per day" in error_text.lower()


# --- Where the time went: per-thread log of API calls and retries -----------
# Eval cases run in a thread pool, and each case runs entirely in one thread
# (the SDK client is synchronous, and so are its retries), so a thread-local
# log attributes every call and retry to the case that made it.

_local = threading.local()


def start_tracking() -> None:
    """Start recording the API calls and retries this thread makes."""
    _local.log = {"calls": [], "retries": []}


def stop_tracking() -> dict | None:
    """Stop recording; return {"calls": [...], "retries": [...]} or None if not tracking."""
    log = getattr(_local, "log", None)
    _local.log = None
    return log


# Optional progress hook, called as on_wait(reason, seconds) just before any
# retry wait, so a caller (the eval) can say what it is waiting for instead of
# going silent for up to a minute.
on_wait: Callable[[str, float], None] | None = None


def _record(kind: str, entry: dict) -> None:
    log = getattr(_local, "log", None)
    if log is not None:
        log[kind].append(entry)
    if kind == "retries" and on_wait is not None:
        on_wait(entry["reason"], entry["wait_seconds"])


# The SDK retries 5xx errors and timeouts itself (tenacity) and logs each retry
# at INFO as "Retrying <fn> in 4.123 seconds as it raised ServerError: 503 ...".
_SDK_RETRY_RE = re.compile(r"^Retrying \S+ in ([\d.]+) seconds as it (?:raised|returned) (.*)", re.DOTALL)


class _SdkRetryRecorder(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        match = _SDK_RETRY_RE.match(record.getMessage())
        if match:
            _record("retries", {
                "by": "sdk",
                "wait_seconds": round(float(match.group(1)), 1),
                "reason": match.group(2).strip().rstrip(".")[:200],
            })


def track_sdk_retries() -> None:
    """Also record the SDK's own retries. Safe to call more than once.

    Lowering the SDK logger to INFO prints nothing by itself: with no logging
    configured, Python only prints WARNING and above.
    """
    logger = logging.getLogger("google_genai._api_client")
    if not any(isinstance(h, _SdkRetryRecorder) for h in logger.handlers):
        logger.addHandler(_SdkRetryRecorder())
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)


def call(model: str, fn: Callable[[], T], sleep: Callable[[float], None] = time.sleep) -> T:
    """Run one Gemini API call, handling HTTP 429 (quota) and 503 (overload)."""
    rate_limited = overloaded = 0
    overload_waited = 0.0
    while True:
        started = time.perf_counter()
        try:
            return fn()
        except errors.APIError as exc:
            error = exc
        finally:
            # Includes any SDK retries and their waits inside fn().
            _record("calls", {"model": model, "seconds": round(time.perf_counter() - started, 1)})
        code = getattr(error, "code", None)

        if code == 503:  # model overloaded: wait longer each time, then give up
            if overloaded == len(OVERLOAD_WAITS):
                raise ModelOverloadedError(model, overloaded + 1, overload_waited) from error
            wait = OVERLOAD_WAITS[overloaded]
            overloaded += 1
            overload_waited += wait
            _record("retries", {"by": "overload", "wait_seconds": wait,
                                "reason": f"503 model overloaded ({model})"})
            sleep(wait)
            continue

        if code != 429:
            raise error
        text = str(error)
        delay = retry_after_seconds(text)
        if is_daily_quota(text):
            raise QuotaExhaustedError(model, delay) from error
        if rate_limited == RATE_LIMIT_RETRIES:
            raise error
        wait = min(delay or 2.0 * 2**rate_limited, MAX_WAIT_SECONDS)
        rate_limited += 1
        _record("retries", {"by": "rate_limit", "wait_seconds": round(wait, 1),
                            "reason": f"429 per-minute rate limit on {model}"})
        sleep(wait)

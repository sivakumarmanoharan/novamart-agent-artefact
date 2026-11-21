"""All tunable settings in one place, overridable with environment variables.

Keeping these in one file makes it easy to explain (and change) every knob
during a review: which models we call, how retrieval is cut off, and which
"today" the agent assumes.
"""

import datetime as dt
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Read settings (e.g. GEMINI_API_KEY) from a .env file in the project root, if
# present. Variables already set in the shell take precedence over the file.
load_dotenv(ROOT / ".env")
POLICY_DIR = ROOT / "data" / "policies"
ORDERS_FILE = ROOT / "data" / "orders.json"
CACHE_DIR = ROOT / ".cache"

# Chat model, pinned to a concrete version so eval runs stay comparable (every
# results file records it). The "gemini-flash-latest" alias would survive model
# retirements but could change answers between runs without notice.
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Model used as LLM-as-judge in the eval. It compares an answer with a
# human-written reference answer, which is an easier task than answering, so a
# fast Flash-Lite model is enough and keeps eval runs quick. Using a different
# model from the agent also avoids a model grading its own output.
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")

# Cap on the model's internal "thinking" tokens, the main driver of latency.
# Default 0 (thinking off): the configuration the eval was run with, and much
# faster for short, retrieval-grounded answers. Set a budget (e.g. 1024) to
# turn thinking on, or set it empty (GEMINI_THINKING_BUDGET=) for the model's
# own default. Some models reject 0 (gemini-3.5-flash-lite returns
# 400 INVALID_ARGUMENT); use a small positive budget if you switch to one.
THINKING_BUDGET = os.getenv("GEMINI_THINKING_BUDGET", "0") or None

# Embedding model for retrieval.
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

# Retrieval: how many chunks to return and the minimum cosine similarity a
# chunk needs to count as "relevant". If nothing clears the bar the search
# tool returns NO_RELEVANT_POLICY. 0.60 is an uncalibrated first guess and, in
# the eval, never filters anything out: off-topic queries score 0.60-0.62, the
# same as weak matches for answerable ones (README, Key decisions 4).
# Inspect raw scores with:  python -m novamart.retrieval "some question"
TOP_K = int(os.getenv("NOVAMART_TOP_K", "4"))
MIN_SCORE = float(os.getenv("NOVAMART_MIN_SCORE", "0.60"))

# Max model<->tool round trips per user turn (guards against loops).
MAX_TOOL_STEPS = 5


def today() -> dt.date:
    """The date the agent believes it is.

    Return-window questions depend on the date, so the eval pins it with
    NOVAMART_TODAY to keep results reproducible.
    """
    override = os.getenv("NOVAMART_TODAY")
    return dt.date.fromisoformat(override) if override else dt.date.today()

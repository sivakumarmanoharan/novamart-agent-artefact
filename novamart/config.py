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

# Chat model. "gemini-flash-latest" is Google's alias for the current Flash
# model, so the prototype keeps working as models are retired. For anything
# reproducible (e.g. comparing eval runs) pin a concrete version instead,
# for example GEMINI_MODEL=gemini-2.5-flash.
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Model used as LLM-as-judge in the eval. It compares an answer with a
# human-written reference answer, which is an easier task than answering, so a
# fast Flash-Lite model is enough and keeps eval runs quick. Using a different
# model from the agent also avoids a model grading its own output.
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")

# Optional cap on the model's internal "thinking" tokens, the main driver of
# latency on recent Gemini models. Unset = model default. For example
# GEMINI_THINKING_BUDGET=0 turns thinking off on gemini-2.5-flash, which is much
# faster and is plenty for short, retrieval-grounded answers.
THINKING_BUDGET = os.getenv("GEMINI_THINKING_BUDGET")

# Embedding model for retrieval.
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

# Retrieval: how many chunks to return and the minimum cosine similarity a
# chunk needs to count as "relevant". If nothing clears the bar the search
# tool says so explicitly, which is the first layer of our refusal behaviour.
# Calibrate with:  python -m novamart.retrieval "some question"
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

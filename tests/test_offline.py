"""Offline tests: no API key or network needed.

They cover the deterministic parts (chunking, tools, refusal threshold,
citation parsing, eval checks) and the agent's tool loop with a scripted
fake Gemini client.  Run:  python -m pytest -q
"""

import re
from types import SimpleNamespace

import numpy as np
from google.genai import types

from eval.checks import run_checks
from novamart import config
from novamart.agent import REFUSAL_PHRASE, SupportAgent, ToolCall, Turn
from novamart.retrieval import PolicyIndex, load_chunks
from novamart.tools import Toolbox


def bag_of_words_embedder(texts, task_type):
    """Stand-in for Gemini embeddings: hashed word counts."""
    vecs = np.zeros((len(texts), 512), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            vecs[i, hash(word) % 512] += 1
    return vecs + 1e-6  # avoid zero vectors


def make_toolbox():
    return Toolbox(PolicyIndex(load_chunks(), bag_of_words_embedder))


def test_chunks_have_citable_ids():
    chunks = load_chunks()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert "returns-and-refunds#electronics" in ids
    assert all(c.text for c in chunks)


def test_search_finds_relevant_section(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.0)  # fake embedder has a different score scale
    result = make_toolbox().search_policies("restocking fee opened laptops electronics")
    assert result["results"][0]["source"] == "returns-and-refunds#electronics"


def test_search_below_threshold_reports_no_policy(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.99)
    result = make_toolbox().search_policies("dogs in the store")
    assert result["results"] == [] and "NO_RELEVANT_POLICY" in result["note"]


def test_order_tool():
    tb = make_toolbox()
    assert tb.get_order_status("nm-10002")["status"] == "IN_TRANSIT"
    assert "not found" in tb.get_order_status("NM-99999")["error"]
    assert "not a valid" in tb.get_order_status("12345")["error"]
    assert "Unknown tool" in tb.call("delete_order", {})["error"]


def test_citation_parsing_and_refusal():
    turn = Turn("q", answer="15 days [source: a#b] and [source: c#d, e#f].")
    assert turn.cited_sources == {"a#b", "c#d", "e#f"}
    assert not turn.refused
    assert Turn("q", answer=REFUSAL_PHRASE.replace("'", "’") + ".").refused


def test_eval_checks():
    case = {
        "expect_tools": ["search_policies"],
        "expect_sources": ["x#y"],
        "must_contain": [["15%", "15 percent"]],
        "expect_refusal": False,
    }
    good = Turn("q", "Fee is 15% [source: x#y]",
                [ToolCall("search_policies", {}, {"results": [{"source": "x#y"}]})])
    assert all(v is not False for v in run_checks(case, good).values())
    invented = Turn("q", "Fee is 15% [source: x#y]", [])  # cites something never retrieved
    assert run_checks(case, invented)["citations_valid"] is False


def _model_reply(*parts):
    content = types.Content(role="model", parts=list(parts))
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class ScriptedClient:
    """Returns pre-scripted Gemini responses and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.requests.append(list(contents))
        return self.replies.pop(0)


def test_agent_runs_tool_loop_and_returns_answer():
    client = ScriptedClient([
        _model_reply(types.Part(function_call=types.FunctionCall(
            name="get_order_status", args={"order_id": "NM-10002"}))),
        _model_reply(types.Part(text="It is in transit with Purolator.")),
    ])
    agent = SupportAgent(client, make_toolbox(), model="fake")
    turn = agent.ask("Where is NM-10002?")
    assert turn.answer == "It is in transit with Purolator."
    assert turn.tool_calls[0].result["carrier"] == "Purolator"
    # The second request must include the function response we sent back.
    last_sent = client.requests[1][-1]
    assert last_sent.parts[0].function_response.name == "get_order_status"


def test_agent_falls_back_when_model_loops(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_STEPS", 2)
    loop = _model_reply(types.Part(function_call=types.FunctionCall(
        name="search_policies", args={"query": "x"})))
    agent = SupportAgent(ScriptedClient([loop, loop]), make_toolbox(), model="fake")
    turn = agent.ask("?")
    assert turn.refused
    assert agent.history[-1].role == "model"  # history stays valid for the next turn

"""Deterministic checks for one eval case. Pure functions, no API calls,
so they are unit-tested in tests/test_offline.py."""

from __future__ import annotations

from novamart.agent import Turn


def run_checks(case: dict, turn: Turn) -> dict[str, bool | None]:
    """Return {check_name: True/False, or None when the check doesn't apply}."""
    answer = turn.answer.lower()
    called = {c.name for c in turn.tool_calls}

    checks: dict[str, bool | None] = {}

    # Did the agent decline exactly when it should have?
    expected = case.get("expect_refusal")
    checks["refusal"] = None if expected is None else (turn.refused == expected)

    # Did it call the tools this question needs (e.g. the order lookup)?
    expect_tools = set(case.get("expect_tools", []))
    checks["tools"] = expect_tools <= called if expect_tools else None

    # Did it cite at least one of the sections that actually hold the answer?
    expect_sources = set(case.get("expect_sources", []))
    checks["cites_source"] = bool(expect_sources & turn.cited_sources) if expect_sources else None

    # Every citation must point to a passage the search tool really returned
    # (catches invented or mangled citations).
    checks["citations_valid"] = turn.cited_sources <= turn.retrieved_sources

    # Key facts present? Each group passes if any of its variants appears.
    groups = case.get("must_contain", [])
    checks["key_facts"] = (
        all(any(v.lower() in answer for v in group) for group in groups) if groups else None
    )
    return checks

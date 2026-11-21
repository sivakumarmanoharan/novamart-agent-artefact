"""Offline tests for the eval's overload pause/stop policy and --resume rules."""

import argparse

import pytest

from eval import run_eval as R
from novamart import quota


@pytest.fixture(autouse=True)
def fresh_run_state(monkeypatch):
    monkeypatch.setattr(R, "OVERLOAD_PAUSE_SECONDS", 0)  # don't really pause in tests
    R.OVERLOAD.update(paused_until=0.0, streak=0)
    R.OVERLOAD_STOP.clear()
    R.QUOTA_EXHAUSTED.clear()
    yield
    R.OVERLOAD.update(paused_until=0.0, streak=0)
    R.OVERLOAD_STOP.clear()


class OverloadedAgent:
    client = None

    def ask(self, question):
        raise quota.ModelOverloadedError("gemini-test", 5, 125)


CASE = {"id": "case-a", "category": "tool", "question": "Where is NM-10002?"}


def test_one_overloaded_case_pauses_two_in_a_row_stop_the_run():
    first = R.run_case(OverloadedAgent(), CASE, use_judge=False)
    assert first["status"] == "OVERLOADED" and not R.OVERLOAD_STOP.is_set()

    second = R.run_case(OverloadedAgent(), CASE, use_judge=False)
    assert second["status"] == "OVERLOADED" and R.OVERLOAD_STOP.is_set()

    third = R.run_case(OverloadedAgent(), CASE, use_judge=False)
    assert third["status"] == "SKIPPED" and "--resume" in third["error"]


def test_a_successful_case_resets_the_overload_streak():
    R.note_overload(quota.ModelOverloadedError("gemini-test", 5, 125))
    R.note_success()
    R.note_overload(quota.ModelOverloadedError("gemini-test", 5, 125))
    assert not R.OVERLOAD_STOP.is_set()


def settings(**overrides):
    args = argparse.Namespace(no_judge=False, workers=1)
    return {**R.run_settings(args, "crucial"), **overrides}


def previous_run(statuses: dict, **setting_overrides):
    return {"selection": "crucial", "settings": settings(**setting_overrides),
            "results": [{"id": cid, "status": status} for cid, status in statuses.items()]}


def test_resume_reruns_only_cases_without_a_result():
    prev = previous_run({"grounded-electronics-window": "PASS", "tool-in-transit": "FAIL",
                         "tool-order-not-found": "ERROR", "mixed-laptop-return": "OVERLOADED",
                         "refuse-pets": "QUOTA", "refuse-near-miss-sick-days": "SKIPPED"})
    cases, kept = R.resume_plan(prev, settings(workers=4))  # workers may differ
    assert {c["id"] for c in cases} == {"tool-order-not-found", "mixed-laptop-return",
                                        "refuse-pets", "refuse-near-miss-sick-days"}
    assert {r["id"] for r in kept} == {"grounded-electronics-window", "tool-in-transit"}


def test_resume_refuses_to_mix_settings():
    prev = previous_run({"tool-in-transit": "ERROR"}, judge_model=None)  # previous run had --no-judge
    with pytest.raises(SystemExit) as info:
        R.resume_plan(prev, settings())
    assert "judge_model" in str(info.value)


def test_resume_refuses_files_without_settings():
    with pytest.raises(SystemExit):
        R.resume_plan({"results": []}, settings())

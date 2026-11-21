"""Offline tests for novamart.quota: daily quota vs per-minute rate limit."""

import pytest
from google.genai import errors

from novamart import quota

# Shapes taken from real AI Studio 429 responses.
DAILY = {"error": {
    "code": 429, "status": "RESOURCE_EXHAUSTED",
    "message": "You exceeded your current quota ... Generate content requests per day limit exceeded on Free Tier.",
    "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "48766s"},
    ],
}}
PER_MINUTE = {"error": {
    "code": 429, "status": "RESOURCE_EXHAUSTED",
    "message": "You exceeded your current quota.",
    "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "5"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"},
    ],
}}


def failing_then(results):
    """A fake API call that raises / returns the given items in order."""
    calls = []

    def fn():
        item = results[len(calls)]
        calls.append(item)
        if isinstance(item, Exception):
            raise item
        return item

    return fn, calls


def test_daily_quota_raises_immediately_with_reset_time():
    fn, calls = failing_then([errors.ClientError(429, DAILY)])
    with pytest.raises(quota.QuotaExhaustedError) as info:
        quota.call("gemini-test", fn, sleep=lambda s: pytest.fail("must not wait on a daily quota"))
    assert len(calls) == 1
    assert info.value.model == "gemini-test"
    assert info.value.retry_after_seconds == 48766
    assert "13h 32m" in str(info.value) and "gemini-test" in str(info.value)


def test_per_minute_limit_waits_and_retries():
    waits = []
    fn, calls = failing_then([errors.ClientError(429, PER_MINUTE), "ok"])
    assert quota.call("gemini-test", fn, sleep=waits.append) == "ok"
    assert waits == [7.0] and len(calls) == 2


def test_other_client_errors_are_not_treated_as_quota():
    fn, _ = failing_then([errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})])
    with pytest.raises(errors.ClientError):
        quota.call("gemini-test", fn, sleep=lambda s: None)


def test_tracking_records_calls_and_rate_limit_waits():
    fn, _ = failing_then([errors.ClientError(429, PER_MINUTE), "ok"])
    quota.start_tracking()
    quota.call("gemini-test", fn, sleep=lambda s: None)
    log = quota.stop_tracking()
    assert [c["model"] for c in log["calls"]] == ["gemini-test", "gemini-test"]
    assert log["retries"] == [{"by": "rate_limit", "wait_seconds": 7.0,
                               "reason": "429 per-minute rate limit on gemini-test"}]
    assert quota.stop_tracking() is None  # tracking is off again


def test_sdk_retries_are_recorded_from_its_log():
    import logging

    quota.track_sdk_retries()
    quota.track_sdk_retries()  # idempotent: no duplicate handler
    quota.start_tracking()
    # The message format tenacity uses for the SDK's retries.
    logging.getLogger("google_genai._api_client").info(
        "Retrying google.genai._api_client.BaseApiClient._request_once in 4.137 seconds "
        "as it raised ServerError: 503 UNAVAILABLE. The model is overloaded.."
    )
    logging.getLogger("google_genai._api_client").info("Some other SDK message")
    log = quota.stop_tracking()
    assert len(log["retries"]) == 1
    retry = log["retries"][0]
    assert retry["by"] == "sdk" and retry["wait_seconds"] == 4.1
    assert retry["reason"].startswith("ServerError: 503 UNAVAILABLE")


def test_on_wait_hook_announces_retry_waits():
    announced = []
    quota.on_wait = lambda reason, seconds: announced.append((reason, seconds))
    try:
        fn, _ = failing_then([errors.ClientError(429, PER_MINUTE), "ok"])
        quota.call("gemini-test", fn, sleep=lambda s: None)
    finally:
        quota.on_wait = None
    assert announced == [("429 per-minute rate limit on gemini-test", 7.0)]


OVERLOADED_503 = {"error": {"code": 503, "status": "UNAVAILABLE",
                            "message": "This model is currently experiencing high demand."}}


def test_overload_waits_longer_each_time_then_succeeds():
    waits = []
    fn, calls = failing_then([errors.ServerError(503, OVERLOADED_503),
                              errors.ServerError(503, OVERLOADED_503), "ok"])
    assert quota.call("gemini-test", fn, sleep=waits.append) == "ok"
    assert waits == [5.0, 15.0] and len(calls) == 3


def test_overload_that_outlasts_every_retry_raises_model_overloaded():
    waits = []
    fn, calls = failing_then([errors.ServerError(503, OVERLOADED_503)] * 5)
    with pytest.raises(quota.ModelOverloadedError) as info:
        quota.call("gemini-test", fn, sleep=waits.append)
    assert waits == quota.OVERLOAD_WAITS and len(calls) == 5
    assert "gemini-test" in str(info.value) and "--resume" in str(info.value)


def test_503_is_not_left_to_the_sdks_short_retries():
    assert 503 not in quota.SDK_RETRY_STATUS_CODES

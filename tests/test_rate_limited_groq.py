import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq

from src.providers.rate_limited_groq import (
    GroqRateLimitError,
    RateLimitedChatGroq,
    _parse_retry_after_seconds,
)

# Real message from a live 429 hit during eval testing (tokens-per-day limit).
_REAL_TPD_MESSAGE = (
    "Error code: 429 - {'error': {'message': \"Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_x` service tier "
    "`on_demand` on tokens per day (TPD): Limit 100000, Used 99772, "
    "Requested 459. Please try again in 3m19.584s. Need more tokens? "
    "Upgrade to Dev Tier today at https://console.groq.com/settings/billing\", "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


def _fake_result(text="ok"):
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class TestRpmThrottle(unittest.TestCase):
    def test_waits_when_budget_exhausted(self):
        llm = RateLimitedChatGroq(model="llama-3.3-70b-versatile", api_key="fake", rpm=2)
        clock = {"t": 0.0}
        sleeps = []

        with patch("time.monotonic", side_effect=lambda: clock["t"]), \
             patch("time.sleep", side_effect=lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s))):
            llm._wait_for_rpm_budget()
            llm._wait_for_rpm_budget()
            # third call within the same 60s window should wait for room
            llm._wait_for_rpm_budget()

        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0)

    def test_old_calls_age_out_of_window(self):
        llm = RateLimitedChatGroq(model="llama-3.3-70b-versatile", api_key="fake", rpm=1)
        clock = {"t": 0.0}

        with patch("time.monotonic", side_effect=lambda: clock["t"]), \
             patch("time.sleep", side_effect=lambda s: clock.__setitem__("t", clock["t"] + s)):
            llm._wait_for_rpm_budget()
            clock["t"] += 61  # past the rolling window
            llm._wait_for_rpm_budget()

        self.assertEqual(len(llm._call_times), 1)


class TestBackoff(unittest.TestCase):
    def test_retries_on_429_then_succeeds(self):
        llm = RateLimitedChatGroq(
            model="llama-3.3-70b-versatile", api_key="fake", rpm=1000, base_backoff_seconds=0.01
        )
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Error code: 429 - rate limit exceeded")
            return _fake_result("recovered")

        with patch.object(ChatGroq, "_generate", side_effect=flaky), \
             patch("time.sleep", return_value=None):
            result = llm._generate([])

        self.assertEqual(calls["n"], 3)
        self.assertEqual(result.generations[0].message.content, "recovered")

    def test_gives_up_after_max_retries(self):
        llm = RateLimitedChatGroq(
            model="llama-3.3-70b-versatile", api_key="fake", rpm=1000,
            max_retries_on_429=2, base_backoff_seconds=0.01,
        )

        def always_429(*args, **kwargs):
            raise RuntimeError("429 too many requests")

        with patch.object(ChatGroq, "_generate", side_effect=always_429), \
             patch("time.sleep", return_value=None):
            with self.assertRaises(GroqRateLimitError):
                llm._generate([])

    def test_non_rate_limit_errors_are_not_retried(self):
        llm = RateLimitedChatGroq(model="llama-3.3-70b-versatile", api_key="fake", rpm=1000)
        calls = {"n": 0}

        def broken(*args, **kwargs):
            calls["n"] += 1
            raise ValueError("not a rate limit problem")

        with patch.object(ChatGroq, "_generate", side_effect=broken):
            with self.assertRaises(ValueError):
                llm._generate([])
        self.assertEqual(calls["n"], 1)


class TestRetryAfterParsing(unittest.TestCase):
    def test_parses_minutes_and_seconds(self):
        self.assertAlmostEqual(_parse_retry_after_seconds(_REAL_TPD_MESSAGE), 3 * 60 + 19.584)

    def test_parses_seconds_only(self):
        self.assertAlmostEqual(_parse_retry_after_seconds("try again in 33.5s"), 33.5)

    def test_no_hint_returns_none(self):
        self.assertIsNone(_parse_retry_after_seconds("Error code: 429 - rate limit exceeded"))


class TestRetryAfterBackoff(unittest.TestCase):
    def test_honors_the_suggested_wait_then_succeeds(self):
        llm = RateLimitedChatGroq(model="llama-3.3-70b-versatile", api_key="fake", rpm=1000)
        calls = {"n": 0}

        def tpd_then_ok(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(_REAL_TPD_MESSAGE)
            return _fake_result("recovered")

        sleeps = []
        with patch.object(ChatGroq, "_generate", side_effect=tpd_then_ok), \
             patch("time.sleep", side_effect=sleeps.append):
            result = llm._generate([])

        self.assertEqual(result.generations[0].message.content, "recovered")
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 3 * 60 + 19.584 + 2.0)

    def test_wait_is_capped_at_max_retry_after_wait_seconds(self):
        llm = RateLimitedChatGroq(
            model="llama-3.3-70b-versatile", api_key="fake", rpm=1000,
            max_retry_after_wait_seconds=60.0,
        )
        message = "Error code: 429 - rate limit. Please try again in 45m0s."
        calls = {"n": 0}

        def tpd_then_ok(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(message)
            return _fake_result("recovered")

        sleeps = []
        with patch.object(ChatGroq, "_generate", side_effect=tpd_then_ok), \
             patch("time.sleep", side_effect=sleeps.append):
            llm._generate([])

        self.assertEqual(sleeps, [60.0])

    def test_gives_up_after_max_retry_after_attempts(self):
        llm = RateLimitedChatGroq(
            model="llama-3.3-70b-versatile", api_key="fake", rpm=1000,
            max_retry_after_attempts=2,
        )

        def always_tpd(*args, **kwargs):
            raise RuntimeError(_REAL_TPD_MESSAGE)

        with patch.object(ChatGroq, "_generate", side_effect=always_tpd), \
             patch("time.sleep", return_value=None):
            with self.assertRaises(GroqRateLimitError):
                llm._generate([])


if __name__ == "__main__":
    unittest.main()

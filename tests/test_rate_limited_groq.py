import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq

from src.providers.rate_limited_groq import GroqRateLimitError, RateLimitedChatGroq


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


if __name__ == "__main__":
    unittest.main()

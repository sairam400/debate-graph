"""Wraps ChatGroq with a requests-per-minute limiter and exponential backoff,
so the eval harness can run unattended against Groq's free tier without
tripping (or getting stuck retrying past) its rate limit.

Two layers, deliberately separate:
  1. RPM throttle: before every call, wait until there's room in a rolling
     60s window. This is what should fire in the common case -- pacing, not
     recovering from an error.
  2. Backoff: if Groq still returns a 429 (clock skew, a burst from another
     process, whatever), retry with exponential delay + jitter, capped at
     max_retries. If that's exhausted, the exception propagates -- the
     caller (the eval driver) is expected to have a checkpoint to resume
     from rather than this wrapper silently giving up forever.
"""
import random
import re
import time
from collections import deque
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_groq import ChatGroq
from pydantic import PrivateAttr


class GroqRateLimitError(Exception):
    pass


def _is_rate_limit_error(exc: Exception) -> bool:
    # Prefer the SDK's own type/status over string matching -- matching on
    # "rate limit" in the message text false-positives on messages that
    # mention rate limiting while explicitly saying it isn't the problem.
    try:
        import groq
        if isinstance(exc, groq.RateLimitError):
            return True
    except ImportError:
        pass
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429:
        return True
    return bool(re.search(r"\b429\b", str(exc)))


class RateLimitedChatGroq(ChatGroq):
    rpm: int = 28
    max_retries_on_429: int = 5
    base_backoff_seconds: float = 2.0

    _call_times: deque = PrivateAttr(default_factory=deque)

    def _wait_for_rpm_budget(self) -> None:
        now = time.monotonic()
        while self._call_times and now - self._call_times[0] > 60:
            self._call_times.popleft()
        if len(self._call_times) >= self.rpm:
            sleep_for = 60 - (now - self._call_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._call_times.append(time.monotonic())

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempt = 0
        while True:
            self._wait_for_rpm_budget()
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                if attempt >= self.max_retries_on_429:
                    raise GroqRateLimitError(
                        f"gave up after {attempt + 1} attempts, still rate-limited: {exc}"
                    ) from exc
                delay = self.base_backoff_seconds * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                attempt += 1

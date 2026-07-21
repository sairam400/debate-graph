"""Wraps ChatGroq with a requests-per-minute limiter and exponential backoff,
so the eval harness can run unattended against Groq's free tier without
tripping (or getting stuck retrying past) its rate limit.

Three layers:
  1. RPM throttle: before every call, wait until there's room in a rolling
     60s window. This is what should fire in the common case -- pacing, not
     recovering from an error.
  2. Retry-after-aware backoff: Groq's tokens-per-day 429 responses name an
     exact wait ("Please try again in 3m19.584s"), discovered live when the
     real eval run hit this limit mid-experiment -- a *different* limit than
     RPM, with no fixed reset time, so blind exponential backoff either gives
     up long before the window clears or wastes time waiting less than
     necessary. Parsed and honored here instead of guessed at.
  3. Blind exponential backoff as the fallback when no hint is present
     (ordinary burst 429s, clock skew, etc.), capped at max_retries.
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


_RETRY_AFTER = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def _parse_retry_after_seconds(message: str):
    match = _RETRY_AFTER.search(message)
    if not match:
        return None
    minutes, seconds = match.groups()
    return (int(minutes) * 60 if minutes else 0) + float(seconds)


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
    # Separate, smaller budget for retry-after-driven waits: these are
    # already told exactly how long to wait, so each attempt is far more
    # likely to succeed than blind backoff -- fewer attempts needed, but
    # each one can be a multi-minute sleep, so this also bounds worst case.
    max_retry_after_attempts: int = 3
    max_retry_after_wait_seconds: float = 700.0

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
        retry_after_attempt = 0
        while True:
            self._wait_for_rpm_budget()
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise

                retry_after = _parse_retry_after_seconds(str(exc))
                if retry_after is not None:
                    if retry_after_attempt >= self.max_retry_after_attempts:
                        raise GroqRateLimitError(
                            f"gave up after {retry_after_attempt + 1} retry-after waits, "
                            f"still rate-limited: {exc}"
                        ) from exc
                    time.sleep(min(retry_after + 2.0, self.max_retry_after_wait_seconds))
                    retry_after_attempt += 1
                    continue

                if attempt >= self.max_retries_on_429:
                    raise GroqRateLimitError(
                        f"gave up after {attempt + 1} attempts, still rate-limited: {exc}"
                    ) from exc
                delay = self.base_backoff_seconds * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                attempt += 1

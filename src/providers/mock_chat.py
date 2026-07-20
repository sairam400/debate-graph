"""A LangChain-compatible chat model that replays a scripted plan instead of
calling a real API -- same philosophy as the MockProvider in
Data-Analyst-Agent: the *decision* of what to say/call is scripted and
deterministic, but every tool call a node makes with the response still
executes for real against the real database. That is what keeps mock runs
honest enough to catch prompt-wiring and parsing bugs, not just a rigged
demo.

plan: list[dict], one entry per call to .invoke(). Each entry is either
  {"tool_calls": [{"name": str, "args": dict}]}
or
  {"content": str}
Raises IndexError with a clear message if the graph asks for more turns than
the plan provides -- a loud failure, not a silent default answer.
"""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class MockChatModel(BaseChatModel):
    plan: List[dict]

    _call_count: int = PrivateAttr(default=0)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._call_count >= len(self.plan):
            raise IndexError(
                f"MockChatModel plan exhausted after {self._call_count} calls "
                f"-- the graph asked for another turn the plan doesn't script."
            )
        step = self.plan[self._call_count]
        self._call_count += 1

        if "tool_calls" in step:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": tc["name"], "args": tc["args"], "id": f"call_{self._call_count}_{i}"}
                    for i, tc in enumerate(step["tool_calls"])
                ],
            )
        else:
            message = AIMessage(content=step["content"])

        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def bind_tools(self, tools, **kwargs):
        # Decisions come from the scripted plan, not real tool-schema
        # reasoning, so there's nothing to bind -- return self as-is.
        return self

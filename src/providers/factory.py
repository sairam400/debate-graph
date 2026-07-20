"""One switch, three providers: ollama (local, free), groq (hosted, free
tier), mock (no network, no model). Nothing here ever touches an Anthropic
or OpenAI key -- that's a hard constraint of this project, not an oversight.
"""
from ..config import SETTINGS


def get_chat_model(provider: str, model: str | None = None, plan: list | None = None):
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or SETTINGS.ollama_model,
            base_url=SETTINGS.ollama_base_url,
            temperature=0,
        )

    if provider == "groq":
        from .rate_limited_groq import RateLimitedChatGroq

        if not SETTINGS.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set -- required for provider=groq")
        return RateLimitedChatGroq(
            model=model or SETTINGS.groq_model,
            api_key=SETTINGS.groq_api_key,
            temperature=0,
            rpm=SETTINGS.groq_rpm,
            max_retries_on_429=SETTINGS.groq_max_retries,
        )

    if provider == "mock":
        from .mock_chat import MockChatModel

        if plan is None:
            raise ValueError("provider=mock requires a plan")
        return MockChatModel(plan=plan)

    raise ValueError(f"unknown provider: {provider!r} (expected ollama, groq, or mock)")

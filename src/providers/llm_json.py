"""Structured JSON out of a plain chat completion, with one retry that feeds
the parse error back to the model. Used for assign_positions and judge,
where the output needs a fixed shape but the target models (a 7b local model
included) aren't reliably wired for provider-native structured output modes
-- asking for JSON in plain text and parsing leniently is the more portable
choice across ollama/groq/mock."""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class JSONCompletionError(Exception):
    pass


def extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("no JSON object found", text, 0)


def complete_json(llm, system: str, user: str, retries: int = 1):
    prompt = user
    last_error = None
    for _ in range(retries + 1):
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        try:
            return extract_json(response.content)
        except json.JSONDecodeError as exc:
            last_error = exc
            prompt = (
                f"{user}\n\nYour previous response was not valid JSON ({exc}). "
                "Respond with a single valid JSON object only, no prose, no markdown fences."
            )
    raise JSONCompletionError(f"model did not return valid JSON after {retries + 1} attempts: {last_error}")

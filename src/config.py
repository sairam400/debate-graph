"""Central settings, read from the environment with a gitignored .env file at
the repo root as fallback. Zero API spend is a hard constraint: the only
providers are ollama (local), groq (free tier), and mock (no network at
all). There is no anthropic/openai path anywhere in this project."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def _dotenv_values():
    if not ENV_PATH.exists():
        return {}
    values = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_DOTENV = _dotenv_values()


def get_env(key, default=None):
    import os
    return os.environ.get(key) or _DOTENV.get(key) or default


class Settings:
    def __init__(self):
        self.llm_provider = get_env("LLM_PROVIDER", "mock")

        self.ollama_model = get_env("OLLAMA_MODEL", "qwen2.5:7b")
        self.ollama_base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_timeout_seconds = int(get_env("OLLAMA_TIMEOUT_SECONDS", "180"))

        self.groq_api_key = get_env("GROQ_API_KEY")
        self.groq_model = get_env("GROQ_MODEL", "llama-3.3-70b-versatile")
        # Free-tier RPM for llama-3.3-70b-versatile is 30 as of this writing;
        # default sits a bit under that so the limiter -- not a 429 -- is
        # what paces requests in the common case.
        self.groq_rpm = int(get_env("GROQ_RPM", "28"))
        self.groq_max_retries = int(get_env("GROQ_MAX_RETRIES", "5"))

        self.max_rounds = int(get_env("MAX_ROUNDS", "3"))
        self.max_tool_calls_per_turn = int(get_env("MAX_TOOL_CALLS_PER_TURN", "3"))

        self.checkpoint_db_path = get_env("CHECKPOINT_DB_PATH", str(REPO_ROOT / "data" / "checkpoints.sqlite"))


SETTINGS = Settings()

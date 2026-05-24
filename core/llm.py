"""LLM client abstraction.

Returns an OpenAI-compatible async client configured for the active provider.
Switch providers by setting LLM_PROVIDER in the environment:

  LLM_PROVIDER=openai      — uses OPENAI_API_KEY, base_url=api.openai.com
  LLM_PROVIDER=openrouter  — uses OPENROUTER_API_KEY, base_url=openrouter.ai/api/v1

Both providers expose the same OpenAI-compatible chat completions API, so the
rest of the codebase never branches on provider — only this module does.

Required env vars (validated at startup in bot.py):
  openai     → OPENAI_API_KEY
  openrouter → OPENROUTER_API_KEY
"""

from functools import lru_cache

from openai import AsyncOpenAI

from core.settings import get_settings

DEFAULT_MODEL: dict[str, str] = {
    "openai": "gpt-4o",
    "openrouter": "openai/gpt-4o",
}


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    if settings.llm_provider == "openrouter":
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_default_model() -> str:
    return DEFAULT_MODEL[get_settings().llm_provider]
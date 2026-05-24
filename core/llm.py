"""LLM client abstraction.

Wraps OpenAI (GPT-4o) or OpenRouter models behind a single interface.
Switch between providers by setting LLM_PROVIDER in the environment.
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
from functools import lru_cache
from typing import Literal

import pytz
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["openai", "openrouter"] = "openai"
    openai_api_key: str = ""
    openrouter_api_key: str = ""

    # Database
    database_url: str = ""

    # Memory (mem0 cloud)
    mem0_api_key: str = ""

    # Search
    tavily_api_key: str = ""

    # Discord — required
    discord_bot_token: str
    discord_guild_id: int
    discord_user_id: int

    # Digest scheduler
    digest_time: str = "09:00"
    evening_digest_time: str = "20:00"
    digest_timezone: str = "America/Toronto"

    # App config
    environment: Literal["development", "production"] = "development"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    context_window_size: int = 12
    profile_extraction_interval: int = 10
    max_react_iterations: int = 8
    task_token_budget: int = 8000

    @model_validator(mode="after")
    def check_llm_key(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        return self

    @model_validator(mode="after")
    def check_digest_timezone(self) -> "Settings":
        try:
            pytz.timezone(self.digest_timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValueError(
                f"DIGEST_TIMEZONE={self.digest_timezone!r} is not a valid IANA timezone. "
                "Examples: America/Toronto, Europe/London, Asia/Tokyo"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

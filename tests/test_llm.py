"""Tests for core/llm.py — client factory and model resolution."""

import pytest
from unittest.mock import patch, MagicMock
from openai import AsyncOpenAI

from core.llm import DEFAULT_MODEL, get_client, get_default_model


def _make_settings(provider: str, openai_key: str = "sk-openai", openrouter_key: str = "sk-openrouter"):
    s = MagicMock()
    s.llm_provider = provider
    s.openai_api_key = openai_key
    s.openrouter_api_key = openrouter_key
    return s


@pytest.fixture(autouse=True)
def clear_client_cache():
    get_client.cache_clear()
    yield
    get_client.cache_clear()


class TestDefaultModel:
    def test_openai_model(self):
        assert DEFAULT_MODEL["openai"] == "gpt-4o"

    def test_openrouter_model(self):
        assert DEFAULT_MODEL["openrouter"] == "openai/gpt-4o"

    def test_get_default_model_openai(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openai")):
            assert get_default_model() == "gpt-4o"

    def test_get_default_model_openrouter(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openrouter")):
            assert get_default_model() == "openai/gpt-4o"


class TestGetClient:
    def test_openai_returns_async_openai(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openai")):
            client = get_client()
            assert isinstance(client, AsyncOpenAI)

    def test_openai_uses_correct_api_key(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openai", openai_key="sk-test-123")):
            client = get_client()
            assert client.api_key == "sk-test-123"

    def test_openai_uses_default_base_url(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openai")):
            client = get_client()
            assert "openai.com" in str(client.base_url)

    def test_openrouter_returns_async_openai(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openrouter")):
            client = get_client()
            assert isinstance(client, AsyncOpenAI)

    def test_openrouter_uses_correct_api_key(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openrouter", openrouter_key="sk-or-test")):
            client = get_client()
            assert client.api_key == "sk-or-test"

    def test_openrouter_uses_openrouter_base_url(self):
        with patch("core.llm.get_settings", return_value=_make_settings("openrouter")):
            client = get_client()
            assert "openrouter.ai" in str(client.base_url)

    def test_get_client_is_cached(self):
        settings = _make_settings("openai")
        with patch("core.llm.get_settings", return_value=settings):
            client1 = get_client()
            client2 = get_client()
            assert client1 is client2

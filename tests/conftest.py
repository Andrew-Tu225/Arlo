"""Shared pytest fixtures and configuration."""

import pytest


@pytest.fixture(autouse=True)
def _langchain_debug_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangGraph invokes langchain_core globals that expect langchain.debug."""
    try:
        import langchain
        if not hasattr(langchain, "debug"):
            monkeypatch.setattr(langchain, "debug", False, raising=False)
    except ImportError:
        pass

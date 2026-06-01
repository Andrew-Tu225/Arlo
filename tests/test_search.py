"""Tests for core/tools/search.py — Tavily web search wrapper."""

import logging
from unittest.mock import MagicMock, patch

from core.tools.search import web_search


def _make_client(results: list[dict]) -> MagicMock:
    client = MagicMock()
    client.search.return_value = {"results": results}
    return client


class TestWebSearch:
    async def test_returns_expected_shape(self):
        client = _make_client([
            {"url": "https://example.com", "title": "Example", "content": "Some content"},
        ])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("test query")
        assert results == [
            {"url": "https://example.com", "title": "Example", "snippet": "Some content"}
        ]

    async def test_maps_content_to_snippet(self):
        client = _make_client([
            {"url": "https://a.com", "title": "A", "content": "Content A"},
            {"url": "https://b.com", "title": "B", "content": "Content B"},
        ])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("query")
        assert results[0]["snippet"] == "Content A"
        assert results[1]["snippet"] == "Content B"

    async def test_passes_max_results_to_client(self):
        client = _make_client([])
        with patch("core.tools.search._get_client", return_value=client):
            await web_search("test query", max_results=3)
        client.search.assert_called_once_with("test query", max_results=3)

    async def test_default_max_results_is_five(self):
        client = _make_client([])
        with patch("core.tools.search._get_client", return_value=client):
            await web_search("test query")
        client.search.assert_called_once_with("test query", max_results=5)

    async def test_returns_empty_list_on_error(self):
        client = MagicMock()
        client.search.side_effect = RuntimeError("API error")
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("test query")
        assert results == []

    async def test_logs_error_on_failure(self, caplog):
        client = MagicMock()
        client.search.side_effect = RuntimeError("API error")
        with patch("core.tools.search._get_client", return_value=client):
            with caplog.at_level(logging.ERROR, logger="core.tools.search"):
                await web_search("failing query")
        assert "failing query" in caplog.text

    async def test_returns_empty_list_when_no_results(self):
        client = _make_client([])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("obscure query")
        assert results == []

    async def test_handles_missing_fields_gracefully(self):
        client = _make_client([{"url": "https://x.com"}])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("test")
        assert results == [{"url": "https://x.com", "title": "", "snippet": ""}]

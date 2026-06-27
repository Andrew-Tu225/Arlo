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
        """Result dicts contain url, title, and snippet keys."""
        client = _make_client([
            {"url": "https://example.com", "title": "Example", "content": "Some content"},
        ])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("test query")
        assert results == [
            {"url": "https://example.com", "title": "Example", "snippet": "Some content"}
        ]

    async def test_maps_content_to_snippet(self):
        """Tavily's content field is exposed as snippet in the output."""
        client = _make_client([
            {"url": "https://a.com", "title": "A", "content": "Content A"},
            {"url": "https://b.com", "title": "B", "content": "Content B"},
        ])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("query")
        assert results[0]["snippet"] == "Content A"
        assert results[1]["snippet"] == "Content B"

    async def test_passes_max_results_to_client(self):
        """max_results is forwarded to the Tavily client unchanged."""
        client = _make_client([])
        with patch("core.tools.search._get_client", return_value=client):
            await web_search("test query", max_results=3)
        client.search.assert_called_once_with("test query", max_results=3)

    async def test_default_max_results_is_five(self):
        """Default max_results is 5 when the caller omits it."""
        client = _make_client([])
        with patch("core.tools.search._get_client", return_value=client):
            await web_search("test query")
        client.search.assert_called_once_with("test query", max_results=5)

    async def test_returns_empty_list_on_error(self):
        """Any Tavily client exception returns [] without propagating."""
        client = MagicMock()
        client.search.side_effect = RuntimeError("API error")
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("test query")
        assert results == []

    async def test_logs_error_on_failure(self, caplog):
        """Failed search logs at ERROR with the query string for traceability."""
        client = MagicMock()
        client.search.side_effect = RuntimeError("API error")
        with patch("core.tools.search._get_client", return_value=client):
            with caplog.at_level(logging.ERROR, logger="core.tools.search"):
                await web_search("failing query")
        assert "failing query" in caplog.text

    async def test_returns_empty_list_when_no_results(self):
        """Empty Tavily results return [] rather than None."""
        client = _make_client([])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("obscure query")
        assert results == []

    async def test_handles_missing_fields_gracefully(self):
        """Missing url, title, or content fields default to empty string."""
        client = _make_client([{"url": "https://x.com"}])
        with patch("core.tools.search._get_client", return_value=client):
            results = await web_search("test")
        assert results == [{"url": "https://x.com", "title": "", "snippet": ""}]

    async def test_snippet_truncated_to_tavily_snippet_max_chars(self):
        """Snippet is sliced to tavily_snippet_max_chars from settings."""
        long_content = "x" * 500
        client = _make_client([{"url": "https://a.com", "title": "A", "content": long_content}])
        with (
            patch("core.tools.search._get_client", return_value=client),
            patch("core.tools.search.get_settings") as mock_settings,
        ):
            mock_settings.return_value.tavily_snippet_max_chars = 100
            results = await web_search("query")
        assert len(results[0]["snippet"]) == 100

    async def test_snippet_not_truncated_when_within_limit(self):
        """Snippet shorter than the limit is returned unmodified."""
        client = _make_client([{"url": "https://a.com", "title": "A", "content": "short"}])
        with (
            patch("core.tools.search._get_client", return_value=client),
            patch("core.tools.search.get_settings") as mock_settings,
        ):
            mock_settings.return_value.tavily_snippet_max_chars = 280
            results = await web_search("query")
        assert results[0]["snippet"] == "short"

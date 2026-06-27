"""Tests for core/tools/reader.py — basic URL checks and Jina fetch."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from core.tools.reader import read_url


class TestReadUrl:
    async def test_rejects_empty_url(self):
        result = await read_url("")
        assert result.startswith("Error:")

    async def test_rejects_non_http_scheme(self):
        result = await read_url("ftp://example.com/file")
        assert "http(s)" in result

    async def test_returns_page_text_on_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Article body text"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com/a")

        assert result == "Article body text"
        mock_client.get.assert_called_once_with("https://r.jina.ai/https://example.com/a")

    async def test_truncates_long_content(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "x" * 15_000

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com")

        assert len(result) <= 10_100
        assert result.endswith("[truncated]")

    async def test_non_200_returns_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com")

        assert "HTTP 403" in result

    async def test_timeout_returns_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await read_url("https://example.com")

        assert "timed out" in result

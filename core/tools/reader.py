"""Jina Reader URL fetcher.

Fetches page content via https://r.jina.ai/{url}.

SSRF validation (private IPs, encoded hosts, DNS checks) is deferred — see
phase4-plan.md. Only basic http(s) + non-empty checks run today.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_JINA_BASE = "https://r.jina.ai/"
_DEFAULT_TIMEOUT = 30.0
_MAX_CONTENT_CHARS = 10_000


def _normalize_url(url: str) -> str:
    text = url.strip()
    if not text:
        raise ValueError("Error: URL is empty")
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Error: URL must be http(s) with a host")
    return text


async def read_url(url: str) -> str:
    """Fetch page text via Jina Reader."""
    try:
        url = _normalize_url(url)
    except ValueError as exc:
        return str(exc)

    jina_url = f"{_JINA_BASE}{url}"
    try:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, follow_redirects=False) as client:
            response = await client.get(jina_url)
    except httpx.TimeoutException:
        return "Error: request timed out"
    except httpx.HTTPError as exc:
        logger.error("Jina fetch failed for %r: %s", url, exc)
        return f"Error: failed to fetch URL ({exc.__class__.__name__})"

    if response.status_code != 200:
        return f"Error: Jina returned HTTP {response.status_code}"

    text = response.text.strip()
    if len(text) > _MAX_CONTENT_CHARS:
        text = text[:_MAX_CONTENT_CHARS] + "\n\n[truncated]"
    return text or "Error: empty page content"

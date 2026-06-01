"""Tavily web search wrapper.

Exposes web_search(query: str) -> list[dict] for use inside the ReAct loop.
Each result contains {url, snippet, title}.
"""

import asyncio
import logging
from functools import lru_cache

from tavily import TavilyClient

from core.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> TavilyClient:
    return TavilyClient(api_key=get_settings().tavily_api_key)


async def web_search(query: str, *, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily. Returns list of {url, title, snippet}."""
    try:
        response = await asyncio.to_thread(
            _get_client().search, query, max_results=max_results
        )
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
    except Exception as exc:
        logger.error("Tavily search failed for query %r: %s", query, exc)
        return []

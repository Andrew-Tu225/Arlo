"""Tavily web search wrapper.

Exposes web_search(query: str) -> list[dict] for use inside the ReAct loop.
Each result contains {url, snippet, title}.
"""

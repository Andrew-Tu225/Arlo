"""Jina Reader URL fetcher.

Exposes read_url(url: str) -> str for use inside the ReAct loop.
Validates URLs before fetching: rejects non-http(s), private IPs, and malformed addresses.
Never passes LLM-generated URLs directly to the HTTP client without validation.
Fetches via https://r.jina.ai/{url}.
"""

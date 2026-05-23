"""LLM client abstraction.

Returns an OpenAI-compatible async client configured for the active provider.
Switch providers by setting LLM_PROVIDER in the environment:

  LLM_PROVIDER=openai      — uses OPENAI_API_KEY, base_url=api.openai.com
  LLM_PROVIDER=openrouter  — uses OPENROUTER_API_KEY, base_url=openrouter.ai/api/v1

Both providers expose the same OpenAI-compatible chat completions API, so the
rest of the codebase never branches on provider — only this module does.

Required env vars (validated at startup in bot.py):
  openai     → OPENAI_API_KEY
  openrouter → OPENROUTER_API_KEY
"""

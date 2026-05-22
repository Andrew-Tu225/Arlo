"""Tone and intent classifier.

Single LLM call with structured output that returns:
  - tone:   casual | task | venting | excited
  - intent: chat | task | memory_update

On parse failure, defaults to {casual, chat} — never raises.
"""

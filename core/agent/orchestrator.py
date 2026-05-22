"""LangGraph ReAct orchestrator.

Executes tool-use loops up to MAX_REACT_ITERATIONS (default: 8).
Nodes: Reason → tool call (web_search / read_url) → Synthesize.
On iteration cap or token budget exceeded: returns an honest fallback message.
"""

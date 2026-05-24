"""Unified LangGraph ReAct agent — the single routing and execution layer.

All message types (casual chat, task requests, memory updates) pass through
this agent. The model decides what to do by which tools it calls — there is
no separate classifier or planner call.

Tool registry (4 tools):
  web_search(query)   — Tavily API; returns list of {url, title, snippet}
  read_url(url)       — SSRF validation → Jina Reader (r.jina.ai/{url}); returns page text
  search_memory(q)    — mem0 semantic search; retrieves relevant user facts on demand
  remember(fact)      — mem0.add(); stores a fact with contradiction handling

Graph shape:
  Reason node (LLM with tools bound)
    → if tool_calls: ToolNode → back to Reason
    → if no tool_calls: done

Exit conditions (first one wins):
  - LLM returns no tool calls          → send response
  - MAX_REACT_ITERATIONS reached       → honest fallback (no hallucination)
  - TASK_TOKEN_BUDGET tokens consumed  → honest fallback

Env vars read: MAX_REACT_ITERATIONS (default 8), TASK_TOKEN_BUDGET (default 8000).
"""

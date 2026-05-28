"""Unified LangGraph ReAct agent — the single routing and execution layer.

All message types (casual chat, task requests, memory updates) pass through
this agent. The model decides what to do by which tools it calls — there is
no separate classifier or planner call.

Phase 1 graph shape (single chat node, no tools):
  START ──► chat ──► END

Phase 4 graph shape (tools added, no schema migration needed):
  START ──► reason ──┬──► tool_executor ──► reason (loop)
                     └──► END

State schema includes iteration_count and token_usage as placeholders so
Phase 4 can wire the ReAct ceiling and token budget without restructuring state.

Exit conditions (Phase 4, first one wins):
  - LLM returns no tool calls          → send response
  - MAX_REACT_ITERATIONS reached       → honest fallback (no hallucination)
  - TASK_TOKEN_BUDGET tokens consumed  → honest fallback
"""

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai.types.chat import ChatCompletionMessageParam

from core.agent.persona import build_system_prompt
from core.llm import get_client, get_default_model

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = "Something went sideways on my end — can you try again?"


class AgentState(TypedDict):
    messages: list[ChatCompletionMessageParam]  # user/assistant dicts (system prepended inside node)
    response: str  # Final reply text sent to Discord
    iteration_count: int  # Phase 1: always 0; Phase 4: incremented per ReAct loop
    token_usage: int  # Phase 1: always 0; Phase 4: accumulated from response.usage


async def _chat_node(state: AgentState) -> dict[str, str]:
    """Single chat node for Phase 1. Calls the LLM with the full message history.

    Prepends the system prompt, calls chat completions, and returns the
    response text. Returns FALLBACK_RESPONSE on any error — never raises.
    """
    system_message: ChatCompletionMessageParam = {"role": "system", "content": build_system_prompt()}
    messages = [system_message, *state["messages"]]

    try:
        result = await get_client().chat.completions.create(
            model=get_default_model(),
            messages=messages,
        )
        content = result.choices[0].message.content
        return {"response": content if content else FALLBACK_RESPONSE}
    except Exception:
        logger.exception("LLM call failed in _chat_node")
        return {"response": FALLBACK_RESPONSE}


def _build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("chat", _chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()


# Compiled at import time — no lazy init needed and avoids blocking the event loop.
_compiled_graph: CompiledStateGraph = _build_graph()


async def run(messages: list[ChatCompletionMessageParam]) -> str:
    """Run the agent on a list of conversation messages.

    Args:
        messages: OpenAI-format dicts with "role" and "content" keys.
            Do not include a system message — the agent prepends it.

    Returns:
        The agent's reply as a plain string.
    """
    initial_state: AgentState = {
        "messages": messages,
        "response": "",
        "iteration_count": 0,
        "token_usage": 0,
    }
    result = await _compiled_graph.ainvoke(initial_state)
    return result["response"] or FALLBACK_RESPONSE

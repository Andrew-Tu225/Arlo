"""Shared LangGraph ReAct engine for user and proactive agents.

Graph shape:
  START → reason → route
                    ├─ tools → reason (loop)
                    └─ END

Ceiling checks run before each ``reason`` LLM call. One iteration is one
completed ``tools`` pass (tool round-trip), then back to ``reason``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai.types.chat import ChatCompletionMessageParam

from core.agent.tools import RESEARCH_FALLBACK, ToolContext, invoke_tool
from core.llm import get_client, get_default_model

logger = logging.getLogger(__name__)

LLM_ERROR_FALLBACK = "Something went sideways on my end — can you try again?"


class ReActState(TypedDict, total=False):
    messages: list[ChatCompletionMessageParam]
    user_id: str | None
    discord_channel_id: str | None
    response: str           #response message at the end
    iteration_count: int
    token_usage: int


SystemPromptBuilder = Callable[[ReActState], str | Awaitable[str]]
ToolContextFactory = Callable[[ReActState], ToolContext]


@dataclass(frozen=True)
class ReactGraphConfig:
    """Configuration for a compiled ReAct graph."""

    build_system_prompt: SystemPromptBuilder
    tool_schemas: list[dict[str, Any]]
    tool_context_factory: ToolContextFactory
    max_react_iterations: int
    task_token_budget: int
    ceiling_fallback: str = RESEARCH_FALLBACK
    llm_error_fallback: str = LLM_ERROR_FALLBACK


def _last_assistant(
    messages: list[ChatCompletionMessageParam],
) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return dict(message)
    return None


def build_react_graph(config: ReactGraphConfig) -> CompiledStateGraph:
    """Compile a ReAct graph: reason ↔ tools until reply or ceiling."""

    async def reason(state: ReActState) -> dict[str, Any]:
        iterations = state.get("iteration_count", 0)
        tokens = state.get("token_usage", 0)
        if (
            iterations >= config.max_react_iterations
            or tokens >= config.task_token_budget
        ):
            return {"response": config.ceiling_fallback}

        prompt = config.build_system_prompt(state)
        system_content = prompt if isinstance(prompt, str) else await prompt
        llm_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
            *state.get("messages", []),
        ]

        try:
            completion = await get_client().chat.completions.create(
                model=get_default_model(),
                messages=llm_messages,
                tools=config.tool_schemas,
                tool_choice="auto",
            )
        except Exception:
            logger.exception("LLM call failed in reason node")
            return {"response": config.llm_error_fallback}

        choice = completion.choices[0].message
        assistant_message: ChatCompletionMessageParam = choice.model_dump(
            exclude_none=True,
        )
        usage_delta = (
            completion.usage.total_tokens or 0 if completion.usage is not None else 0
        )

        new_messages = [*state.get("messages", []), assistant_message]
        new_token_usage = tokens + usage_delta

        if new_token_usage >= config.task_token_budget:
            return {
                "messages": new_messages,
                "token_usage": new_token_usage,
                "response": config.ceiling_fallback,
            }

        if assistant_message.get("tool_calls"):
            return {
                "messages": new_messages,
                "token_usage": new_token_usage,
            }

        content = assistant_message.get("content")
        text = content if isinstance(content, str) and content.strip() else ""
        return {
            "messages": new_messages,
            "token_usage": new_token_usage,
            "response": text or config.llm_error_fallback,
        }

    async def tools(state: ReActState) -> dict[str, Any]:
        assistant = _last_assistant(state.get("messages", []))
        if assistant is None:
            return {
                "response": config.llm_error_fallback,
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        ctx = config.tool_context_factory(state)
        tool_messages: list[ChatCompletionMessageParam] = []

        for call in assistant.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            observation = await invoke_tool(name, args, ctx)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": observation,
                },
            )

        return {
            "messages": [*state.get("messages", []), *tool_messages],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def route_after_reason(state: ReActState) -> str:
        if state.get("response"):
            return END
        assistant = _last_assistant(state.get("messages", []))
        if assistant and assistant.get("tool_calls"):
            return "tools"
        return END

    graph = StateGraph(ReActState)
    graph.add_node("reason", reason)
    graph.add_node("tools", tools)
    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason",
        route_after_reason,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "reason")
    return graph.compile()


async def run_react_graph(
    graph: CompiledStateGraph,
    *,
    messages: list[ChatCompletionMessageParam],
    user_id: str | None = None,
    discord_channel_id: str | None = None,
    initial_response: str = "",
) -> str:
    """Invoke a compiled ReAct graph and return the final response string."""
    initial_state: ReActState = {
        "messages": messages,
        "user_id": user_id,
        "discord_channel_id": discord_channel_id,
        "response": initial_response,
        "iteration_count": 0,
        "token_usage": 0,
    }
    result = await graph.ainvoke(initial_state)
    return result.get("response") or LLM_ERROR_FALLBACK

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

from core.agent.tools import RESEARCH_FALLBACK, StructuredTool, ToolContext, invoke_tool
from core.llm import get_client, get_default_model
from core.settings import get_settings

logger = logging.getLogger(__name__)

LLM_ERROR_FALLBACK = "Something went sideways on my end — can you try again?"


class ReActState(TypedDict, total=False):
    messages: list[ChatCompletionMessageParam]
    user_id: str | None
    discord_channel_id: str | None
    response: str       # the final response to be returned to the user
    iteration_count: int
    token_usage: int


SystemPromptBuilder = Callable[[ReActState], str | Awaitable[str]]
ToolBuilder = Callable[[ToolContext], list[StructuredTool]]


@dataclass(frozen=True)
class ReactGraphConfig:
    """Configuration for a compiled ReAct graph."""
    build_system_prompt: SystemPromptBuilder
    tool_schemas: list[dict[str, Any]]
    tool_builder: ToolBuilder
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


def build_react_graph(
    graph_config: ReactGraphConfig,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Compile a ReAct graph: reason ↔ tools until reply or ceiling.

    Args:
        graph_config: Prompt builder, tool schemas, and budget limits.
        checkpointer: LangGraph checkpointer for state persistence. Pass a
            ``MemorySaver`` for the orchestrator (enables interrupt/resume);
            pass ``None`` for sub-agent graphs.
    """

    async def reason(state: ReActState) -> dict[str, Any]:
        iterations = state.get("iteration_count", 0)
        tokens = state.get("token_usage", 0)
        if (
            iterations >= graph_config.max_react_iterations
            or tokens >= graph_config.task_token_budget
        ):
            return {"response": graph_config.ceiling_fallback}

        prompt = graph_config.build_system_prompt(state)
        system_content = prompt if isinstance(prompt, str) else await prompt
        llm_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
            *state.get("messages", []),
        ]

        try:
            completion = await get_client().chat.completions.create(
                model=get_default_model(),
                messages=llm_messages,
                tools=graph_config.tool_schemas,
                tool_choice="auto",
            )
        except Exception:
            logger.exception("LLM call failed in reason node")
            return {"response": graph_config.llm_error_fallback}

        choice = completion.choices[0].message
        assistant_message: ChatCompletionMessageParam = choice.model_dump(
            exclude_none=True,
        )
        usage_delta = (
            completion.usage.total_tokens or 0 if completion.usage is not None else 0
        )

        new_messages = [*state.get("messages", []), assistant_message]
        new_token_usage = tokens + usage_delta

        # check whether it goes over token limit
        if new_token_usage >= graph_config.task_token_budget:
            return {
                "messages": new_messages,
                "token_usage": new_token_usage,
                "response": graph_config.ceiling_fallback,
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
            "response": text or graph_config.llm_error_fallback,
        }

    async def tools(state: ReActState, config: dict[str, Any]) -> dict[str, Any]:
        assistant = _last_assistant(state.get("messages", []))
        if assistant is None:
            return {
                "response": graph_config.llm_error_fallback,
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        configurable = config.get("configurable", {})
        pool = configurable.get("pool")
        bot = configurable.get("bot")

        ctx = ToolContext(
            user_id=state.get("user_id") or "",
            pool=pool,
            bot=bot,
            discord_channel_id=state.get("discord_channel_id"),
        )
        tool_map = {t.name: t for t in graph_config.tool_builder(ctx)}
        max_obs_chars = get_settings().tool_observation_max_chars
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

            observation = await invoke_tool(name, args, tool_map, tool_call_id=call.get("id", ""))
            if len(observation) > max_obs_chars:
                logger.debug(
                    "Tool observation truncated: tool=%s original=%d chars",
                    name,
                    len(observation),
                )
                observation = observation[:max_obs_chars] + " … [truncated]"
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
    return graph.compile(checkpointer=checkpointer)


async def run_react_graph(
    graph: CompiledStateGraph,
    *,
    messages: list[ChatCompletionMessageParam] | None = None,
    user_id: str | None = None,
    pool: Any = None,
    bot: Any = None,
    discord_channel_id: str | None = None,
    initial_response: str = "",
    thread_id: str | None = None,
    resume_command: Any = None,
) -> str:
    """Invoke a compiled ReAct graph and return the final response string."""
    config = {
        "configurable": {
            "thread_id": thread_id or user_id or "default",
            "pool": pool,
            "bot": bot,
        }
    }
    
    if resume_command is not None:
        await graph.ainvoke(resume_command, config=config)
    else:
        initial_state: ReActState = {
            "messages": messages or [],
            "user_id": user_id,
            "discord_channel_id": discord_channel_id,
            "response": initial_response,
            "iteration_count": 0,
            "token_usage": 0,
        }
        await graph.ainvoke(initial_state, config=config)

    state = await graph.aget_state(config)
    if state.tasks and state.tasks[0].interrupts:
        interrupt_val = state.tasks[0].interrupts[0].value
        from core.agent import actions
        ctx = ToolContext(
            user_id=user_id or "",
            pool=pool,
            bot=bot,
            discord_channel_id=discord_channel_id,
        )
        return await actions.request_medium_risk_approval(
            tool_name=interrupt_val["tool_name"],
            args=interrupt_val["tool_args"],
            ctx=ctx,
            tool_call_id=interrupt_val["tool_call_id"],
            thread_id=thread_id or "default",
        )

    return state.values.get("response") or LLM_ERROR_FALLBACK



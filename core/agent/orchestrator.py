"""Unified LangGraph ReAct agent — the single routing and execution layer.

All message types (casual chat, task requests, memory updates) pass through
this agent. The model decides what to do by which tools it calls — there is
no separate classifier or planner call.

Graph shape (Phase 4):
  START → reason ──┬──► tools → reason (loop)
                   └──► END

Medium-risk schedule writes (create, edit, delete) require Discord approval
before execution — see core.agent.actions.

Exit conditions (first one wins):
  - LLM returns no tool calls          → send response
  - MAX_REACT_ITERATIONS reached       → honest fallback (no hallucination)
  - TASK_TOKEN_BUDGET tokens consumed  → honest fallback
"""

from __future__ import annotations

import asyncio
from typing import Any
import uuid

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from openai.types.chat import ChatCompletionMessageParam

from langgraph.checkpoint.memory import MemorySaver

from core import db
from core.agent.prompts import build_orchestrator_prompt
from core.agent.react import ReactGraphConfig, build_react_graph, run_react_graph
from core.agent.tools import ToolContext, build_orchestrator_tools, get_orchestrator_schemas
from core.memory import extractor
from core.settings import get_settings

FALLBACK_RESPONSE = "Something went sideways on my end — can you try again?"


def _build_config() -> ReactGraphConfig:
    settings = get_settings()

    def build_system_prompt_for_state(_state: dict[str, Any]) -> str:
        return build_orchestrator_prompt()

    return ReactGraphConfig(
        build_system_prompt=build_system_prompt_for_state,
        tool_schemas=get_orchestrator_schemas(),
        tool_builder=build_orchestrator_tools,
        max_react_iterations=settings.max_react_iterations,
        task_token_budget=settings.task_token_budget,
        llm_error_fallback=FALLBACK_RESPONSE,
    )


_compiled_graph: CompiledStateGraph = build_react_graph(_build_config(), checkpointer=MemorySaver())


async def run(
    messages: list[ChatCompletionMessageParam],
    *,
    user_id: str | None = None,
    pool: Any = None,
    bot: Any = None,
    discord_channel_id: str | None = None,
) -> str:
    """Run the agent on a list of conversation messages.

    Args:
        messages: OpenAI-format dicts with "role" and "content" keys.
            Do not include a system message — the agent prepends it.
        user_id: Discord user ID for memory and schedule tools.
        pool: asyncpg pool for schedule tools and pending_actions.
        bot: discord.py Bot for schedule registration and approval UI.
        discord_channel_id: Channel where the user message arrived (approval UI).

    Returns:
        The agent's reply as a plain string.
    """
    thread_id = str(uuid.uuid4())
    return await run_react_graph(
        _compiled_graph,
        messages=messages,
        user_id=user_id,
        pool=pool,
        bot=bot,
        discord_channel_id=discord_channel_id,
        thread_id=thread_id,
    )


async def resume(
    thread_id: str,
    approved: bool,
    *,
    user_id: str,
    pool: Any = None,
    bot: Any = None,
    discord_channel_id: str | None = None,
) -> str:
    """Resume a paused agent execution thread and process the final response.

    Args:
        thread_id: The unique thread ID used when the graph was paused.
        approved: Whether the human approved (True) or cancelled (False) the action.
        user_id: Discord user ID for logging.
        pool: asyncpg pool for episodic messages and extraction.
        bot: discord.py Bot context.
        discord_channel_id: Channel ID for context.

    Returns:
        The agent's reply as a plain string.
    """
    resume_command = Command(resume=approved)

    response = await run_react_graph(
        _compiled_graph,
        messages=None,
        user_id=user_id,
        pool=pool,
        bot=bot,
        discord_channel_id=discord_channel_id,
        thread_id=thread_id,
        resume_command=resume_command,
    )

    if pool is not None and not response.startswith("Awaiting your confirmation"):
        await db.insert_episodic_message(
            pool,
            user_id=user_id,
            role="assistant",
            content=response,
        )
        asyncio.create_task(extractor.maybe_extract(pool, user_id))

    return response


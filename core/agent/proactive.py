"""Scheduled task executor — reads a task instruction and produces a Discord message.

Handles any schedule type: proactive outreach, reminders, market summaries, topic
discovery, or any task the user defines. The task instruction drives what the agent
does; the system prompt explains how to operate.

Returns a plain string (the Discord message) or "" on agent failure / iteration ceiling.
Read-only: no write tools, no HITL, no memory writes.

Adding a new content source (Twitter, Reddit): implement the tool coroutine, append
to build_proactive_tools(). No other changes required.

Invocation: ainvoke directly — run_react_graph requires a checkpointer (see react.py).
schedule_id is passed via config["configurable"] for the get_recent_sends tool.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from core.agent.prompts import build_proactive_prompt, build_proactive_system_prompt
from core.agent.react import ReactGraphConfig, build_react_graph
from core.agent.tools import PROACTIVE_FALLBACK, build_proactive_tools, get_proactive_schemas
from core.settings import get_settings

logger = logging.getLogger(__name__)

proactive_agent_graph = build_react_graph(
    ReactGraphConfig(
        build_system_prompt=lambda _: build_proactive_system_prompt(),
        tool_schemas=get_proactive_schemas(),
        tool_builder=build_proactive_tools,
        max_react_iterations=get_settings().proactive_max_react_iterations,
        task_token_budget=get_settings().proactive_task_token_budget,
        ceiling_fallback=PROACTIVE_FALLBACK,
        llm_error_fallback=PROACTIVE_FALLBACK,
    ),
    checkpointer=None,
)


async def run_proactive_agent(
    task: str,
    *,
    user_id: str,
    schedule_id: int,
    channel_topic: str | None = None,
    pool: Any = None,
) -> str:
    """Run the proactive agent and return a Discord message string or "" on failure.

    schedule_id is passed via config["configurable"] for the get_recent_sends tool.
    APScheduler never receives an exception from this function.
    """
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": f"proactive-{uuid.uuid4()}",
            "pool": pool,
            "bot": None,
            "schedule_id": schedule_id,
        }
    }
    initial_state: dict[str, Any] = {
        "messages": [{"role": "user", "content": build_proactive_prompt(task, channel_topic)}],
        "user_id": user_id,
        "discord_channel_id": None,
        "response": "",
        "iteration_count": 0,
        "token_usage": 0,
    }

    try:
        final_state: dict[str, Any] = await proactive_agent_graph.ainvoke(
            initial_state, config=config
        )
    except Exception:
        logger.exception("Proactive agent failed for task=%r", task)
        return ""

    return (final_state.get("response") or "").strip()

"""Schedule planner sub-agent — natural language → structured SchedulePlan.

The orchestrator calls run_schedule_planner() as a tool and receives either a
SchedulePlan JSON string or a plain clarifying question. This sub-agent runs
its own ReAct graph (no checkpointer, no interrupts) with list_schedules +
search_memory tools. Tool output never reaches the orchestrator's context window.

Invocation pattern: call graph.ainvoke() directly and read the response from
the returned state dict. Do NOT use run_react_graph() — that function calls
aget_state() which requires a checkpointer (see react.py for details).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from openai.types.chat import ChatCompletionMessageParam

from core.agent.prompts import get_temporal_context
from core.agent.react import LLM_ERROR_FALLBACK, ReactGraphConfig, build_react_graph
from core.agent.tools import PLANNER_FALLBACK, build_planner_tools, get_planner_schemas
from core.settings import get_settings

logger = logging.getLogger(__name__)


class SchedulePlan(BaseModel):
    """Structured schedule plan returned by the planner sub-agent to the orchestrator."""

    action: Literal["create", "edit", "delete"] = Field(
        description="Operation to perform on the schedule."
    )
    name: str = Field(
        description=(
            "Exact schedule name. For edit/delete, must match a name from list_schedules. "
            "For create, derive a short descriptive name from the request."
        )
    )
    task: str = Field(
        description=(
            "Proactive directive the scheduled job will execute — "
            "written as an instruction to Arlo (e.g. 'Send the user a gym reminder')."
        )
    )
    cron_schedule: str = Field(
        description=(
            "Five-field cron expression (e.g. '0 7 * * 1-5') or HH:MM shorthand for daily "
            "(e.g. '07:00')."
        )
    )
    discord_channel_id: str | None = Field(
        default=None,
        description="Discord channel ID to send the message to. Null means DM the user.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the schedule is active immediately after creation.",
    )
    rationale: str = Field(
        description=(
            "One-line explanation for the orchestrator to relay to the user "
            "(e.g. 'Set a weekday morning gym reminder at 7:00 AM')."
        )
    )


PLANNER_SYSTEM_PROMPT = """\
You are a schedule planning engine. The orchestrator gives you a natural-language \
request — you interpret it and output a structured schedule plan. Neutral tone; \
you are not the user-facing voice.

TOOLS
list_schedules — call first for edit/delete to get exact names and detect collisions. \
Also call for create if the request might duplicate an existing schedule.
search_memory  — call if the request references user preferences (timezone, habits, interests).

ACTIONS
create: derive a short descriptive name; infer cron from the request and TEMPORAL CONTEXT.
edit:   match exact name from list_schedules; only include fields that change.
delete: match exact name from list_schedules.

CRON FORMAT
Five-field cron (minute hour day month weekday) or HH:MM shorthand for daily schedules.
"weekdays at 7 AM" → "0 7 * * 1-5" | "Sundays at 9" → "0 9 * * 0" | "daily 8:30" → "08:30"

CLARIFYING QUESTION
If the request is genuinely ambiguous (missing time, unclear action), respond with a \
plain-text question — no JSON. The orchestrator will ask the user.

OUTPUT — when the request is clear, respond only with valid JSON, no markdown fences:
{
  "action": "create" | "edit" | "delete",
  "name": "short descriptive schedule name",
  "task": "Proactive directive the job will run (e.g. 'Send the user a gym reminder')",
  "cron_schedule": "0 7 * * 1-5",
  "discord_channel_id": null,
  "enabled": true,
  "rationale": "One-line summary for the user reply"
}"""


def _build_sub_agent_config() -> ReactGraphConfig:
    settings = get_settings()

    def build_prompt(_: dict[str, Any]) -> str:
        return PLANNER_SYSTEM_PROMPT + "\n" + get_temporal_context()

    return ReactGraphConfig(
        build_system_prompt=build_prompt,
        tool_schemas=get_planner_schemas(),
        tool_builder=build_planner_tools,
        max_react_iterations=settings.planner_max_react_iterations,
        task_token_budget=settings.planner_task_token_budget,
        ceiling_fallback=PLANNER_FALLBACK,
    )


planner_agent_graph = build_react_graph(_build_sub_agent_config(), checkpointer=None)


async def run_schedule_planner(
    request: str,
    *,
    user_id: str,
    pool: Any,
) -> str:
    """Run the schedule planner sub-agent and return a SchedulePlan JSON or clarifying question.

    The sub-agent receives the user's natural-language schedule request,
    optionally calls list_schedules / search_memory, and returns a structured
    plan. Calls graph.ainvoke() directly — no checkpointer, no interrupts.

    Args:
        request: Natural-language description of the schedule change.
        user_id: Used to scope list_schedules and search_memory calls.
        pool: Database connection pool required by list_schedules.

    Returns:
        SchedulePlan serialised as a JSON string when the request is unambiguous.
        A plain-text clarifying question string when more information is needed.
        Falls back to PLANNER_FALLBACK on sub-agent failure or ceiling hit.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": request.strip()},
    ]

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": f"planner-{uuid.uuid4()}",
            "pool": pool,
            "bot": None,
        }
    }
    initial_state: dict[str, Any] = {
        "messages": messages,
        "user_id": user_id,
        "discord_channel_id": None,
        "response": "",
        "iteration_count": 0,
        "token_usage": 0,
    }

    try:
        final_state: dict[str, Any] = await planner_agent_graph.ainvoke(
            initial_state, config=config
        )
    except Exception:
        logger.exception("Schedule planner sub-agent failed for request=%r", request)
        return PLANNER_FALLBACK

    raw = final_state.get("response") or ""

    try:
        plan = SchedulePlan.model_validate_json(raw)
        return plan.model_dump_json()
    except Exception:
        # Non-JSON is a clarifying question unless it's a known error/ceiling string
        if raw and raw not in (LLM_ERROR_FALLBACK, PLANNER_FALLBACK):
            return raw
        return PLANNER_FALLBACK

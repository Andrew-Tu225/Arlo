"""Agent tool definitions: LangChain tools, OpenAI schemas, and invocation.

Schedule context (user_id, pool, bot) is injected via ToolContext when building tools.
Medium-risk approval for schedule writes lands in actions.py (Phase 4.3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import interrupt

from core.memory import store
from core.tools import reader, schedules, search

logger = logging.getLogger(__name__)

RESEARCH_FALLBACK = (
    "I couldn't find a reliable source for that — want to narrow it down?"
)

_READONLY_TOOL_NAMES = frozenset({"web_search", "search_memory"})

MEDIUM_RISK_TOOLS = frozenset({
    "create_schedule",
    "edit_schedule",
    "delete_schedule",
})


def is_medium_risk(tool_name: str) -> bool:
    return tool_name in MEDIUM_RISK_TOOLS


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    pool: Any = None
    bot: Any = None
    discord_channel_id: str | None = None


def build_tools(ctx: ToolContext) -> list[StructuredTool]:
    """Build LangChain tools bound to the current request context."""

    async def web_search(query: str) -> str:
        text = query.strip()
        if not text:
            return "Error: query is empty"
        results = await search.web_search(text)
        return json.dumps(results)

    async def read_url(url: str) -> str:
        text = url.strip()
        if not text:
            return "Error: url is empty"
        return await reader.read_url(text)

    async def search_memory(query: str) -> str:
        text = query.strip()
        if not text:
            return "Error: query is empty"
        if not ctx.user_id:
            return json.dumps([])
        try:
            facts = await store.search(text, ctx.user_id)
        except Exception:
            logger.exception("search_memory failed")
            return "Error: memory search failed"
        return json.dumps(facts)

    async def remember(fact: str, short_term: bool = False) -> str:
        text = fact.strip()
        if not text:
            return "Error: fact is empty"
        if not ctx.user_id:
            return "Error: user_id not available"
        try:
            await store.add(text, ctx.user_id, short_term)
        except Exception:
            logger.exception("remember failed")
            return "Error: could not save memory"
        return "Saved."

    async def list_schedules() -> str:
        if ctx.pool is None:
            return "Error: schedule tools unavailable"
        return await schedules.list_schedules(pool=ctx.pool, user_id=ctx.user_id)

    async def create_schedule(
        name: str,
        task: str,
        cron_schedule: str,
        discord_channel_id: str | None = None,
    ) -> str:
        if ctx.pool is None or ctx.bot is None:
            return "Error: schedule tools unavailable"
        channel = discord_channel_id
        if channel is None and ctx.discord_channel_id is not None:
            channel = ctx.discord_channel_id
        return await schedules.create_schedule(
            pool=ctx.pool,
            bot=ctx.bot,
            user_id=ctx.user_id,
            name=name,
            task=task,
            cron_schedule=cron_schedule,
            discord_channel_id=channel,
        )

    async def delete_schedule(name: str) -> str:
        if ctx.pool is None:
            return "Error: schedule tools unavailable"
        return await schedules.delete_schedule(
            pool=ctx.pool,
            user_id=ctx.user_id,
            name=name,
        )

    async def edit_schedule(
        name: str,
        task: str | None = None,
        cron_schedule: str | None = None,
        discord_channel_id: str | None = None,
        enabled: bool | None = None,
    ) -> str:
        if ctx.pool is None or ctx.bot is None:
            return "Error: schedule tools unavailable"
        return await schedules.edit_schedule(
            pool=ctx.pool,
            bot=ctx.bot,
            user_id=ctx.user_id,
            name=name,
            task=task,
            cron_schedule=cron_schedule,
            discord_channel_id=discord_channel_id,
            enabled=enabled,
        )

    return [
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description="Search the web for current information. Returns JSON list of url, title, snippet.",
        ),
        StructuredTool.from_function(
            coroutine=read_url,
            name="read_url",
            description="Fetch and read plain text from a public HTTP(S) URL via Jina Reader.",
        ),
        StructuredTool.from_function(
            coroutine=search_memory,
            name="search_memory",
            description="Semantic search over stored facts about the user.",
        ),
        StructuredTool.from_function(
            coroutine=remember,
            name="remember",
            description="Save a durable fact about the user to long-term memory.",
        ),
        StructuredTool.from_function(
            coroutine=list_schedules,
            name="list_schedules",
            description=(
                "List the user's proactive schedules (name, task, cron). "
                "Call before delete_schedule to get the exact name."
            ),
        ),
        StructuredTool.from_function(
            coroutine=create_schedule,
            name="create_schedule",
            description=(
                "Create a recurring proactive schedule. "
                "cron_schedule: five-field cron or HH:MM daily time. "
                "discord_channel_id null = DM. Requires user confirmation in Discord."
            ),
        ),
        StructuredTool.from_function(
            coroutine=edit_schedule,
            name="edit_schedule",
            description=(
                "Update an existing schedule by exact name from list_schedules. "
                "Only pass fields to change. Requires user confirmation in Discord."
            ),
        ),
        StructuredTool.from_function(
            coroutine=delete_schedule,
            name="delete_schedule",
            description=(
                "Delete a schedule by exact name from list_schedules. "
                "Requires user confirmation in Discord."
            ),
        ),
    ]


def _schema_ctx() -> ToolContext:
    return ToolContext(user_id="")


def get_openai_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI Chat Completions tool definitions for the reason node."""
    return [convert_to_openai_tool(t) for t in build_tools(_schema_ctx())]


def get_readonly_openai_tool_schemas() -> list[dict[str, Any]]:
    """Schemas for proactive run_schedule_agent (web_search + search_memory only)."""
    return [
        s
        for s in get_openai_tool_schemas()
        if s["function"]["name"] in _READONLY_TOOL_NAMES
    ]


async def invoke_tool(name: str, args: dict[str, Any], ctx: ToolContext, tool_call_id: str = "") -> str:
    """Execute a tool by name. Returns tool observation text."""
    if is_medium_risk(name):
        try:
            approved = interrupt({
                "tool_name": name,
                "tool_args": args,
                "tool_call_id": tool_call_id,
            })
            if not approved:
                return f"Error: User rejected the request to execute '{name}'. Do not perform the action."
        except RuntimeError:
            # Standalone execution/tests outside of a runnable context
            pass


    tools = {tool.name: tool for tool in build_tools(ctx)}
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    if tool.coroutine is None:
        return f"Error: tool '{name}' has no async implementation"
    result = await tool.coroutine(**args)
    return result if isinstance(result, str) else str(result)


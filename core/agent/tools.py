"""Agent tool definitions: LangChain tools, OpenAI schemas, and invocation.

Three independent tool surfaces for the multi-agent design:
  build_orchestrator_tools — memory + schedule writes (no web_search/read_url)
  build_research_tools     — web_search + read_url (research sub-agent only)
  build_planner_tools      — list_schedules + search_memory (schedule planner only)

Each builder owns its tool implementations directly. Shared tools (search_memory,
list_schedules) are extracted as private _make_* factories to avoid duplication.

ToolContext injects request-scoped values (user_id, pool, bot).
Medium-risk schedule writes (create/edit/delete) require Discord HITL — see actions.py.
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

MEDIUM_RISK_TOOLS = frozenset({
    "create_schedule",
    "edit_schedule",
    "delete_schedule",
})


def is_medium_risk(tool_name: str) -> bool:
    """Return True if tool_name requires Discord approval before execution."""
    return tool_name in MEDIUM_RISK_TOOLS


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    pool: Any = None
    bot: Any = None
    discord_channel_id: str | None = None


# ── Shared tool factories (used by multiple builders) ──────────────────────

def _make_search_memory_tool(ctx: ToolContext) -> StructuredTool:
    """Build a search_memory StructuredTool bound to ctx; shared across sub-agent builders."""
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

    return StructuredTool.from_function(
        coroutine=search_memory,
        name="search_memory",
        description="Semantic search over stored facts about the user.",
    )


def _make_list_schedules_tool(ctx: ToolContext) -> StructuredTool:
    """Build a list_schedules StructuredTool bound to ctx; shared across sub-agent builders."""
    async def list_schedules() -> str:
        if ctx.pool is None:
            return "Error: schedule tools unavailable"
        return await schedules.list_schedules(pool=ctx.pool, user_id=ctx.user_id)

    return StructuredTool.from_function(
        coroutine=list_schedules,
        name="list_schedules",
        description=(
            "List the user's proactive schedules (name, task, cron). "
            "Call before delete_schedule to get the exact name."
        ),
    )


# ── Public builders ────────────────────────────────────────────────────────

def build_orchestrator_tools(ctx: ToolContext) -> list[StructuredTool]:
    """Orchestrator tool surface: research + memory + schedule writes.

    web_search and read_url are intentionally absent — the orchestrator never
    calls them directly. All web research goes through research(), which runs
    the research sub-agent in isolation and returns a compact brief.
    """

    async def research(task: str) -> str:
        # Lazy import breaks the import cycle:
        # researcher.py imports tools.py at module level (fine).
        # tools.py importing researcher.py at module level would be circular.
        from core.agent import researcher
        try:
            return await researcher.run_research(task, user_id=ctx.user_id)
        except Exception:
            logger.exception("research tool failed")
            return RESEARCH_FALLBACK

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

    async def delete_schedule(name: str) -> str:
        if ctx.pool is None:
            return "Error: schedule tools unavailable"
        return await schedules.delete_schedule(
            pool=ctx.pool,
            user_id=ctx.user_id,
            name=name,
        )

    return [
        StructuredTool.from_function(
            coroutine=research,
            name="research",
            description=(
                "Run a dedicated web research loop and return a compact brief with cited sources. "
                "Use for any facts, news, or current information you don't already know. "
                "Pass a plain-language description of what you need — the sub-agent decides how to search. "
                "Example: 'Find the current GPT-4o API pricing and note when it was last updated.' "
                "Do not pass a raw search query string."
            ),
        ),
        _make_search_memory_tool(ctx),
        StructuredTool.from_function(
            coroutine=remember,
            name="remember",
            description="Save a durable fact about the user to long-term memory.",
        ),
        _make_list_schedules_tool(ctx),
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


def build_research_tools(ctx: ToolContext) -> list[StructuredTool]:
    """Research sub-agent tools: web_search + read_url."""

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
    ]


def build_planner_tools(ctx: ToolContext) -> list[StructuredTool]:
    """Schedule planner sub-agent tools: list_schedules + search_memory."""
    return [_make_list_schedules_tool(ctx), _make_search_memory_tool(ctx)]


# ── OpenAI schema getters ──────────────────────────────────────────────────

def _schema_ctx() -> ToolContext:
    """Return a minimal ToolContext for building schema-only tool instances."""
    return ToolContext(user_id="")


def get_orchestrator_schemas() -> list[dict[str, Any]]:
    """OpenAI tool schemas for the orchestrator reason node."""
    return [convert_to_openai_tool(t) for t in build_orchestrator_tools(_schema_ctx())]


def get_research_schemas() -> list[dict[str, Any]]:
    """OpenAI tool schemas for the research sub-agent (web_search + read_url)."""
    return [convert_to_openai_tool(t) for t in build_research_tools(_schema_ctx())]


def get_planner_schemas() -> list[dict[str, Any]]:
    """OpenAI tool schemas for the schedule planner sub-agent (list_schedules + search_memory)."""
    return [convert_to_openai_tool(t) for t in build_planner_tools(_schema_ctx())]


# ── Dispatch ───────────────────────────────────────────────────────────────

async def invoke_tool(
    name: str,
    args: dict[str, Any],
    tool_map: dict[str, StructuredTool],
    *,
    tool_call_id: str = "",
) -> str:
    """Execute a tool by name from the caller-provided tool_map.

    tool_map must be pre-built from the graph's own tool builder so that each
    agent only dispatches within its own tool surface.
    """
    logger.info("Tool invocation: %s with args: %s", name, args)
    if is_medium_risk(name):
        try:
            approved = interrupt({
                "tool_name": name,
                "tool_args": args,
                "tool_call_id": tool_call_id,
            })
            if not approved:
                logger.info("Tool execution rejected by user: %s", name)
                return f"Error: User rejected the request to execute '{name}'. Do not perform the action."
        except RuntimeError:
            # Standalone execution / tests outside a runnable LangGraph context
            pass

    tool = tool_map.get(name)
    if tool is None:
        logger.warning("Unknown tool invoked: %s", name)
        return f"Error: unknown tool '{name}'"
    if tool.coroutine is None:
        logger.warning("Tool '%s' has no async implementation", name)
        return f"Error: tool '{name}' has no async implementation"
    result = await tool.coroutine(**args)
    result_str = result if isinstance(result, str) else str(result)
    logger.info("Tool result: %s -> %s", name, result_str[:1000] + ("..." if len(result_str) > 1000 else ""))
    return result_str

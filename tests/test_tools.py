"""Tests for core/agent/tools.py — schemas and invocation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from core.agent.tools import (
    ToolContext,
    build_orchestrator_tools,
    build_research_tools,
    get_orchestrator_schemas,
    get_planner_schemas,
    get_research_schemas,
    invoke_tool,
)


class TestToolMetadata:
    def test_orchestrator_tools(self):
        names = {s["function"]["name"] for s in get_orchestrator_schemas()}
        assert names == {
            "research",
            "search_memory",
            "remember",
            "list_schedules",
            "plan_schedule_change",
            "create_schedule",
            "edit_schedule",
            "delete_schedule",
        }

    def test_research_tools(self):
        names = {s["function"]["name"] for s in get_research_schemas()}
        assert names == {"web_search", "read_url"}

    def test_planner_tools(self):
        names = {s["function"]["name"] for s in get_planner_schemas()}
        assert names == {"list_schedules", "search_memory"}


class TestInvokeTool:
    async def test_web_search_returns_json(self):
        ctx = ToolContext(user_id="u1")
        tool_map = {t.name: t for t in build_research_tools(ctx)}
        with patch(
            "core.agent.tools.search.web_search",
            new=AsyncMock(return_value=[{"url": "https://a.com", "title": "A", "snippet": "s"}]),
        ):
            result = await invoke_tool("web_search", {"query": "ai news"}, tool_map)
        assert json.loads(result)[0]["url"] == "https://a.com"

    async def test_search_memory_injects_user_id(self):
        ctx = ToolContext(user_id="user-99")
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        with patch(
            "core.agent.tools.store.search",
            new=AsyncMock(return_value=["vegetarian"]),
        ) as mock_search:
            result = await invoke_tool("search_memory", {"query": "diet"}, tool_map)
        mock_search.assert_awaited_once_with("diet", "user-99")
        assert json.loads(result) == ["vegetarian"]

    async def test_remember_returns_saved(self):
        ctx = ToolContext(user_id="u1")
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        with patch("core.agent.tools.store.add", new=AsyncMock()):
            result = await invoke_tool(
                "remember",
                {"fact": "likes spicy food", "short_term": False},
                tool_map,
            )
        assert result == "Saved."

    async def test_read_url_delegates(self):
        ctx = ToolContext(user_id="u1")
        tool_map = {t.name: t for t in build_research_tools(ctx)}
        with patch(
            "core.agent.tools.reader.read_url",
            new=AsyncMock(return_value="Page text"),
        ):
            result = await invoke_tool("read_url", {"url": "https://example.com"}, tool_map)
        assert result == "Page text"

    async def test_list_schedules_delegates(self):
        ctx = ToolContext(user_id="u1", pool=MagicMock())
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        with patch(
            "core.agent.tools.schedules.list_schedules",
            new=AsyncMock(return_value='[{"name": "morning-proactive"}]'),
        ) as mock_list:
            result = await invoke_tool("list_schedules", {}, tool_map)
        assert "morning-proactive" in result
        mock_list.assert_awaited_once()

    async def test_create_schedule_delegates(self):
        ctx = ToolContext(user_id="u1", pool=MagicMock(), bot=MagicMock())
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        with patch(
            "core.agent.tools.schedules.create_schedule",
            new=AsyncMock(return_value="Created schedule 'gym' (id=1)."),
        ) as mock_create:
            result = await invoke_tool(
                "create_schedule",
                {"name": "gym", "task": "remind me", "cron_schedule": "07:00"},
                tool_map,
            )
        assert "Created" in result
        mock_create.assert_awaited_once()

    async def test_edit_schedule_delegates(self):
        ctx = ToolContext(user_id="u1", pool=MagicMock(), bot=MagicMock())
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        with patch(
            "core.agent.tools.schedules.edit_schedule",
            new=AsyncMock(return_value="Updated schedule 'gym'."),
        ) as mock_edit:
            result = await invoke_tool(
                "edit_schedule",
                {"name": "gym", "cron_schedule": "08:00"},
                tool_map,
            )
        assert "Updated" in result
        mock_edit.assert_awaited_once()

    async def test_invoke_unknown_tool_returns_error(self):
        ctx = ToolContext(user_id="u1")
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        result = await invoke_tool("nonexistent_tool", {}, tool_map)
        assert "unknown tool" in result

    async def test_search_memory_empty_user_id_returns_empty_list(self):
        ctx = ToolContext(user_id="")
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        result = await invoke_tool("search_memory", {"query": "diet"}, tool_map)
        assert json.loads(result) == []

    async def test_delete_schedule_delegates(self):
        ctx = ToolContext(user_id="u1", pool=MagicMock())
        tool_map = {t.name: t for t in build_orchestrator_tools(ctx)}
        with patch(
            "core.agent.tools.schedules.delete_schedule",
            new=AsyncMock(return_value="Deleted schedule 'morning-proactive'."),
        ) as mock_delete:
            result = await invoke_tool(
                "delete_schedule",
                {"name": "morning-proactive"},
                tool_map,
            )
        assert "Deleted" in result
        mock_delete.assert_awaited_once_with(
            pool=ctx.pool,
            user_id="u1",
            name="morning-proactive",
        )

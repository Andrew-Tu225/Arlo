"""Tests for core/agent/tools.py — schemas and invocation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from core.agent.tools import (
    ToolContext,
    get_openai_tool_schemas,
    get_readonly_openai_tool_schemas,
    invoke_tool,
)


class TestToolMetadata:
    def test_orchestrator_tools(self):
        names = {s["function"]["name"] for s in get_openai_tool_schemas()}
        assert names == {
            "web_search",
            "read_url",
            "search_memory",
            "remember",
            "list_schedules",
            "create_schedule",
            "delete_schedule",
        }

    def test_readonly_subset(self):
        names = {s["function"]["name"] for s in get_readonly_openai_tool_schemas()}
        assert names == {"web_search", "search_memory"}


class TestInvokeTool:
    async def test_web_search_returns_json(self):
        with patch(
            "core.agent.tools.search.web_search",
            new=AsyncMock(return_value=[{"url": "https://a.com", "title": "A", "snippet": "s"}]),
        ):
            result = await invoke_tool(
                "web_search",
                {"query": "ai news"},
                ToolContext(user_id="u1"),
            )
        assert json.loads(result)[0]["url"] == "https://a.com"

    async def test_search_memory_injects_user_id(self):
        with patch(
            "core.agent.tools.store.search",
            new=AsyncMock(return_value=["vegetarian"]),
        ) as mock_search:
            result = await invoke_tool(
                "search_memory",
                {"query": "diet"},
                ToolContext(user_id="user-99"),
            )
        mock_search.assert_awaited_once_with("diet", "user-99")
        assert json.loads(result) == ["vegetarian"]

    async def test_remember_returns_saved(self):
        with patch("core.agent.tools.store.add", new=AsyncMock()):
            result = await invoke_tool(
                "remember",
                {"fact": "likes spicy food", "short_term": False},
                ToolContext(user_id="u1"),
            )
        assert result == "Saved."

    async def test_read_url_delegates(self):
        with patch(
            "core.agent.tools.reader.read_url",
            new=AsyncMock(return_value="Page text"),
        ):
            result = await invoke_tool(
                "read_url",
                {"url": "https://example.com"},
                ToolContext(user_id="u1"),
            )
        assert result == "Page text"

    async def test_list_schedules_delegates(self):
        ctx = ToolContext(user_id="u1", pool=MagicMock())
        with patch(
            "core.agent.tools.schedules.list_schedules",
            new=AsyncMock(return_value='[{"name": "morning-proactive"}]'),
        ) as mock_list:
            result = await invoke_tool("list_schedules", {}, ctx)
        assert "morning-proactive" in result
        mock_list.assert_awaited_once()

    async def test_create_schedule_delegates(self):
        ctx = ToolContext(user_id="u1", pool=MagicMock(), bot=MagicMock())
        with patch(
            "core.agent.tools.schedules.create_schedule",
            new=AsyncMock(return_value="Created schedule 'gym' (id=1)."),
        ) as mock_create:
            result = await invoke_tool(
                "create_schedule",
                {
                    "name": "gym",
                    "task": "remind me",
                    "cron_schedule": "07:00",
                },
                ctx,
            )
        assert "Created" in result
        mock_create.assert_awaited_once()

    async def test_delete_schedule_delegates_name(self):
        ctx = ToolContext(user_id="u1", pool=MagicMock())
        with patch(
            "core.agent.tools.schedules.delete_schedule",
            new=AsyncMock(return_value="Deleted schedule 'morning-proactive'."),
        ) as mock_delete:
            result = await invoke_tool(
                "delete_schedule",
                {"name": "morning-proactive"},
                ctx,
            )
        assert "Deleted" in result
        mock_delete.assert_awaited_once_with(
            pool=ctx.pool,
            user_id="u1",
            name="morning-proactive",
        )

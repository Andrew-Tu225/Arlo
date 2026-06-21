"""Tests for core/agent/researcher.py — research sub-agent."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.researcher import ResearchBrief, SourceItem, _SUMMARY_MAX_CHARS, run_research
from core.agent.tools import RESEARCH_FALLBACK


# ── ResearchBrief schema tests ─────────────────────────────────────────────


class TestResearchBrief:
    def test_serialises_complete_brief(self):
        brief = ResearchBrief(
            summary="Python 3.12 released October 2023.",
            sources=[SourceItem(url="https://python.org", title="Python")],
            complete=True,
        )
        data = json.loads(brief.model_dump_json())
        assert data["summary"] == "Python 3.12 released October 2023."
        assert data["sources"] == [{"url": "https://python.org", "title": "Python"}]
        assert data["complete"] is True
        assert "note" not in data or data["note"] is None

    def test_serialises_incomplete_brief_with_note(self):
        brief = ResearchBrief(
            summary="No reliable source found.",
            sources=[],
            complete=False,
            note="Search limit reached.",
        )
        data = json.loads(brief.model_dump_json())
        assert data["complete"] is False
        assert data["note"] == "Search limit reached."
        assert data["sources"] == []

    def test_validates_json_round_trip(self):
        brief = ResearchBrief(
            summary="Test summary.",
            sources=[SourceItem(url="https://a.com", title="A")],
            complete=True,
        )
        restored = ResearchBrief.model_validate_json(brief.model_dump_json())
        assert restored.summary == brief.summary
        assert restored.sources[0].url == "https://a.com"
        assert restored.complete is True

    def test_note_defaults_to_none(self):
        brief = ResearchBrief(summary="x", sources=[], complete=True)
        assert brief.note is None

    def test_validate_json_rejects_missing_required_fields(self):
        with pytest.raises(Exception):
            ResearchBrief.model_validate_json('{"summary": "only summary"}')


# ── run_research integration tests ────────────────────────────────────────


def _make_completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    total_tokens: int = 50,
) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        **({"tool_calls": tool_calls} if tool_calls else {}),
    }
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(total_tokens=total_tokens)
    return completion


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _make_valid_brief_json(
    summary: str = "Test finding.",
    sources: list[dict[str, str]] | None = None,
    complete: bool = True,
) -> str:
    return json.dumps({
        "summary": summary,
        "sources": sources or [{"url": "https://example.com", "title": "Example"}],
        "complete": complete,
    })


@pytest.mark.asyncio
async def test_run_research_returns_valid_brief_json():
    """Sub-agent returns valid JSON → parsed into ResearchBrief and returned."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(
            content=_make_valid_brief_json(
                summary="Python 3.12 was released in October 2023.",
                sources=[{"url": "https://python.org/downloads", "title": "Downloads"}],
            )
        )
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_research("When was Python 3.12 released?")

    brief = ResearchBrief.model_validate_json(result)
    assert "Python 3.12" in brief.summary
    assert brief.complete is True
    assert any(s.url == "https://python.org/downloads" for s in brief.sources)


@pytest.mark.asyncio
async def test_run_research_task_passed_as_user_message():
    """The task description is sent as the user message to the sub-agent."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=_make_valid_brief_json())
    )
    task = "Find the current pricing for OpenAI's GPT-4o API"
    with patch("core.agent.react.get_client", return_value=client):
        await run_research(task)

    first_call_messages = client.chat.completions.create.await_args_list[0].kwargs["messages"]
    user_msg = next(m for m in first_call_messages if m["role"] == "user")
    assert user_msg["content"] == task


@pytest.mark.asyncio
async def test_run_research_sub_agent_calls_web_search():
    """Sub-agent reasons and calls web_search before producing the brief."""
    search_tool_call = _tool_call("web_search", {"query": "Python 3.12 release date"})
    valid_brief = _make_valid_brief_json(
        summary="Python 3.12 shipped in October 2023.",
        sources=[{"url": "https://python.org/blog/release", "title": "Release"}],
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _make_completion(content=None, tool_calls=[search_tool_call]),
            _make_completion(content=valid_brief),
        ]
    )
    mock_search = AsyncMock(
        return_value=[{"url": "https://python.org/blog/release", "title": "Release", "snippet": "Oct 2023"}]
    )
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.search.web_search", new=mock_search),
    ):
        result = await run_research("Find when Python 3.12 was released and any notable features")

    brief = ResearchBrief.model_validate_json(result)
    assert brief.complete is True
    assert any(s.url == "https://python.org/blog/release" for s in brief.sources)
    assert len(brief.summary) <= _SUMMARY_MAX_CHARS


@pytest.mark.asyncio
async def test_run_research_ceiling_returns_incomplete_brief():
    """On iteration ceiling, run_research returns a complete=false brief."""
    search_tool_call = _tool_call("web_search", {"query": "q"})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=None, tool_calls=[search_tool_call])
    )
    mock_search = AsyncMock(return_value=[])
    with (
        patch("core.agent.react.get_client", return_value=client),
        patch("core.agent.tools.search.web_search", new=mock_search),
        patch("core.agent.react.get_settings") as mock_settings,
    ):
        mock_settings.return_value.research_max_react_iterations = 2
        mock_settings.return_value.research_task_token_budget = 99999
        mock_settings.return_value.tool_observation_max_chars = 2000
        result = await run_research("impossible query")

    brief = ResearchBrief.model_validate_json(result)
    assert brief.complete is False
    assert brief.sources == []
    assert brief.note is not None


@pytest.mark.asyncio
async def test_run_research_exception_returns_incomplete_brief():
    """If ainvoke raises unexpectedly, run_research returns a complete=false brief."""
    with patch(
        "core.agent.researcher.research_agent_graph.ainvoke",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        result = await run_research("any task")

    brief = ResearchBrief.model_validate_json(result)
    assert brief.complete is False
    assert brief.sources == []
    assert brief.note is not None


@pytest.mark.asyncio
async def test_run_research_non_json_response_wrapped_as_incomplete():
    """When the sub-agent returns non-JSON text, it is wrapped as complete=false."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content="plain text, not JSON at all")
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_research("test task")

    brief = ResearchBrief.model_validate_json(result)
    assert brief.complete is False
    assert "plain text" in brief.summary


@pytest.mark.asyncio
async def test_run_research_summary_capped_at_max_chars():
    """Summary in the brief is capped at _SUMMARY_MAX_CHARS on non-JSON fallback."""
    long_text = "x" * 2000
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion(content=long_text)
    )
    with patch("core.agent.react.get_client", return_value=client):
        result = await run_research("test task")

    brief = ResearchBrief.model_validate_json(result)
    assert len(brief.summary) <= _SUMMARY_MAX_CHARS

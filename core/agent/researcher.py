"""Research sub-agent — isolated web_search + read_url ReAct loop.

The orchestrator calls run_research() as a tool and receives a ResearchBrief
JSON string. This sub-agent runs its own ReAct graph (no checkpointer, no
interrupts) and returns only the findings — raw search payloads and intermediate
tool traffic never reach the orchestrator's context window.

Invocation pattern: call graph.ainvoke() directly and read the response from
the returned state dict. Do NOT use run_react_graph() — that function calls
aget_state() which requires a checkpointer (see react.py for details).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from openai.types.chat import ChatCompletionMessageParam

from core.agent.prompts import get_temporal_context
from core.agent.react import LLM_ERROR_FALLBACK, ReactGraphConfig, build_react_graph
from core.agent.tools import RESEARCH_FALLBACK, build_research_tools, get_research_schemas
from core.settings import get_settings

logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 800


class SourceItem(BaseModel):
    url: str = Field(description="URL of the page consulted")
    title: str = Field(description="Title of the page")


class ResearchBrief(BaseModel):
    """Structured result returned by the research sub-agent to the orchestrator."""

    summary: str = Field(
        description=(
            "Research findings — depth matched to task type. "
            "Don't pad simple facts; don't compress detailed analysis."
        )
    )
    sources: list[SourceItem] = Field(
        description="Sources consulted during research (up to 5 URLs)"
    )
    complete: bool = Field(
        description=(
            "True if research was successful. "
            "False if results were limited, no reliable source was found, "
            "or the search ceiling was hit."
        )
    )
    note: str | None = Field(
        default=None,
        description="Explanation of incompleteness; only present when complete is false",
    )


RESEARCH_SYSTEM_PROMPT = """\
You are a research engine. The orchestrator tells you WHAT it needs — you decide \
HOW to find it. Neutral factual tone; you are not the user-facing voice.

TASK TYPES — identify before searching, then match your approach and response:
A. News/events    → search with time signals ("2025", "latest"); \
response: 3–5 items, one source each
B. Factual        → 1 targeted search, read_url if snippet thin; \
response: 2–3 direct sentences
C. Analysis/comparison → search each subject then "X vs Y"; \
response: cover key angles fully, don't compress
D. Technical/how-to   → search official docs first, read_url for depth; \
response: steps or specifics
E. Person/entity      → bio search + recent news search; \
response: key facts then recent activity
F. Price/availability → search official page, read_url it; \
response: exact numbers + note source date

TOOLS
web_search — always first. Craft targeted queries; don't pass the task description \
verbatim. Rephrase once if first results miss. Max 3 searches.
read_url — only after web_search finds a promising URL. \
Use for thin snippets, official docs, exact specs. \
Skip homepages, login walls, video URLs. Max 2 per task.

OUTPUT — respond only with valid JSON, no markdown fences or extra prose:
{
  "summary": "findings (str) — depth matched to task type; \
don't pad facts, don't compress analysis",
  "sources": [{"url": "https://...", "title": "page title"}],
  "complete": true,
  "note": "str — only include when complete is false; explain why results are limited"
}
Never hallucinate. When results are limited: set complete to false, \
include the best you found in summary, add note."""


def _build_sub_agent_config() -> ReactGraphConfig:
    settings = get_settings()

    def build_prompt(_: dict[str, Any]) -> str:
        return RESEARCH_SYSTEM_PROMPT + "\n" + get_temporal_context()

    return ReactGraphConfig(
        build_system_prompt=build_prompt,
        tool_schemas=get_research_schemas(),
        tool_builder=build_research_tools,
        max_react_iterations=settings.research_max_react_iterations,
        task_token_budget=settings.research_task_token_budget,
    )


research_agent_graph = build_react_graph(_build_sub_agent_config(), checkpointer=None)


async def run_research(
    task: str,
    *,
    user_id: str | None = None,
) -> str:
    """Run the research sub-agent and return a ResearchBrief JSON string.

    The sub-agent receives the task description, reasons about how to search,
    and returns findings as structured JSON. Calls graph.ainvoke() directly — no
    checkpointer, no interrupts, no aget_state (see react.py for why).

    Args:
        task: Plain-language description of what information is needed.
              The sub-agent decides the search strategy; do not pass a raw
              search query string.
        user_id: Passed through for logging; the sub-agent does not write memory.

    Returns:
        ResearchBrief serialised as a JSON string. Falls back to a complete=false
        brief on sub-agent failure or non-JSON response.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": task.strip()},
    ]

    # Sub-agents use ainvoke directly. thread_id is set per-call to avoid any
    # cross-call state leakage inside LangGraph internals, even without a
    # checkpointer. pool/bot are None — research tools don't need DB or Discord.
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": f"research-{uuid.uuid4()}",
            "pool": None,
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
        final_state: dict[str, Any] = await research_agent_graph.ainvoke(
            initial_state, config=config
        )
    except Exception:
        logger.exception("Research sub-agent failed for task=%r", task)
        return ResearchBrief(
            summary=RESEARCH_FALLBACK,
            sources=[],
            complete=False,
            note="Sub-agent failed unexpectedly.",
        ).model_dump_json()

    raw = final_state.get("response") or ""
    try:
        brief = ResearchBrief.model_validate_json(raw)
    except Exception:
        # Ceiling hit returns plain-text RESEARCH_FALLBACK, not JSON
        note = (
            "Search limit reached."
            if raw in (RESEARCH_FALLBACK, LLM_ERROR_FALLBACK)
            else "Response was not valid JSON."
        )
        brief = ResearchBrief(
            summary=raw[:_SUMMARY_MAX_CHARS] if raw else RESEARCH_FALLBACK,
            sources=[],
            complete=False,
            note=note,
        )
    return brief.model_dump_json()

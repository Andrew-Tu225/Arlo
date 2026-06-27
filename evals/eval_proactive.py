"""Proactive agent eval — measures Discord message quality for scheduled tasks.

Runs each scenario through the real proactive agent graph with mocked
search_memory, run_research, and get_recent_sends. The LLM composes the
message; the judge scores it against 8 quality criteria.

Scenarios cover: reminders, research digests, anti-repetition, casual outreach,
research failure fallback, channel-topic context, and memory-driven personalisation.

Usage:
    python -m evals.eval_proactive

Requires OPENAI_API_KEY (or OPENROUTER_API_KEY) in .env.
Does NOT require Discord, a database, or Tavily.
"""

import asyncio
import json
import textwrap
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

from core.agent.proactive import run_proactive_agent
from core.llm import get_client, get_default_model

# ---------------------------------------------------------------------------
# Criteria scored per scenario (1 = pass, 0 = fail)
# ---------------------------------------------------------------------------

CRITERIA = {
    "message_produced": (
        "Output is a non-empty string — the agent always produces something, never goes silent."
    ),
    "task_addressed": (
        "Message content relates to the scheduled task instruction. "
        "A gym reminder should mention gym/exercise; a news digest should contain news items."
    ),
    "personalized": (
        "Message references at least one specific detail from the mocked memory. "
        "Generic messages with no memory detail fail. "
        "N/A (score 1) when mock_memory is empty."
    ),
    "tone_appropriate": (
        "Tone matches the task type: "
        "Reminder = direct, brief, no-frills. "
        "Outreach/check-in = casual, warm, friend-like. "
        "Digest/summary = structured but conversational. "
        "No generic openers like 'Good morning!' or 'Hope you're having a great day!'."
    ),
    "not_repetitive": (
        "Content meaningfully differs from the mocked recent_sends output — "
        "no copy-paste of a prior message's specific phrasing or angle. "
        "N/A (score 1) when mock_recent_sends is empty."
    ),
    "no_preamble": (
        "Message does NOT open with meta-commentary: "
        "'Here is your reminder…', 'Here's the message:', 'Task completed:', etc. "
        "The message IS the output — no framing around it."
    ),
    "research_integrated": (
        "When mock_research contains content, that content (or a close paraphrase) "
        "appears in the message. The agent should weave in research findings, not ignore them. "
        "N/A (score 1) when mock_research is None."
    ),
    "fallback_graceful": (
        "When mock_research is an empty string (research failed), the message still "
        "references user context from memory or pivots to a friendly check-in. "
        "It does NOT say nothing or output an error. "
        "N/A (score 1) when mock_research is not empty string."
    ),
}

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "gym_reminder",
        "task": "Send user a gym reminder",
        "mock_memory": ["user is training for a half marathon", "user prefers morning workouts"],
        "mock_recent_sends": [],
        "mock_research": None,
        "channel_topic": None,
        "note": "Simple personalized reminder — should reference marathon training; direct and brief.",
    },
    {
        "id": "morning_ai_news",
        "task": "Send user their morning AI news digest",
        "mock_memory": ["user is very interested in LLMs and AI agents"],
        "mock_recent_sends": [],
        "mock_research": (
            "OpenAI released GPT-5 today with a 2M context window. "
            "Anthropic announced Claude 4 with improved tool use. "
            "Google DeepMind published a paper on multi-step reasoning."
        ),
        "channel_topic": None,
        "note": "Research digest — research findings should be woven into the message, not in bullet list form.",
    },
    {
        "id": "weekly_finance_digest",
        "task": "Send user a weekly summary of fintech and market news",
        "mock_memory": ["user works in fintech", "user invests in index funds"],
        "mock_recent_sends": [],
        "mock_research": (
            "S&P 500 rose 2.1% this week. "
            "The CFPB finalized new open banking rules. "
            "Stripe announced a $1B secondary round at a $70B valuation."
        ),
        "channel_topic": None,
        "note": "Finance summary — structured but conversational; references user's fintech background.",
    },
    {
        "id": "anti_repeat_gym",
        "task": "Send user a gym reminder",
        "mock_memory": ["user is training for a half marathon", "user is focusing on speed work this month"],
        "mock_recent_sends": [
            "Don't forget gym day! You've got that half marathon to train for. Legs day — going heavy or keeping it light?"
        ],
        "mock_research": None,
        "channel_topic": None,
        "note": (
            "Anti-repetition — prior send was about legs day and the marathon. "
            "New message should take a different angle (e.g. speed work, rest day, nutrition)."
        ),
    },
    {
        "id": "casual_checkin",
        "task": "Check in with the user about how their week is going",
        "mock_memory": ["user has been stressed at work lately", "user is a software engineer at a fintech startup"],
        "mock_recent_sends": [],
        "mock_research": None,
        "channel_topic": None,
        "note": "Outreach tone — warm, casual, references work stress; not a formal template check-in.",
    },
    {
        "id": "research_failure_fallback",
        "task": "Send user the latest space exploration news",
        "mock_memory": ["user loves space exploration and astronomy", "user follows NASA launches closely"],
        "mock_recent_sends": [],
        "mock_research": "",  # Research returned empty — simulating failure
        "channel_topic": None,
        "note": (
            "Graceful fallback — research failed (empty string). "
            "Message should still reference space interest from memory or ask how user is doing."
        ),
    },
    {
        "id": "channel_topic_context",
        "task": "Send daily inspirational quote",
        "mock_memory": [],
        "mock_recent_sends": [],
        "mock_research": None,
        "channel_topic": "morning-motivation",
        "note": (
            "Channel context — 'morning-motivation' should influence tone and content. "
            "Message fits the channel's purpose. No generic 'Good morning!' opener."
        ),
    },
    {
        "id": "upcoming_trip",
        "task": "Remind user about their upcoming trip and suggest packing tips",
        "mock_memory": [
            "user is traveling to Tokyo next week",
            "user prefers minimal packing — carry-on only",
            "user is vegetarian",
        ],
        "mock_recent_sends": [],
        "mock_research": None,
        "channel_topic": None,
        "note": "Memory-rich reminder — references Tokyo trip; packing tips tailored to minimal packer.",
    },
    {
        "id": "meal_suggestion",
        "task": "Ask user what they feel like eating tonight and suggest something",
        "mock_memory": ["user is vegetarian", "user likes spicy food", "user lives in Toronto"],
        "mock_recent_sends": [],
        "mock_research": None,
        "channel_topic": None,
        "note": "Personalized suggestion — vegetarian + spicy; casual; not a generic dinner prompt.",
    },
    {
        "id": "no_memory_fact",
        "task": "Send user a random interesting fact",
        "mock_memory": [],
        "mock_recent_sends": [],
        "mock_research": (
            "Octopuses have three hearts: two pump blood to the gills, one pumps it to the body. "
            "Their blood is blue due to hemocyanin. They can also edit their own RNA in real time, "
            "allowing rapid adaptation to temperature changes."
        ),
        "channel_topic": None,
        "note": "No memory — relies on research; the fact should appear in the message; no generic opener.",
    },
]

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating the output of a proactive Discord message agent.

Scheduled task instruction: {task}
Channel topic (if any): {channel_topic}

Mocked memory facts available to the agent:
{mock_memory}

Mocked recent sends (prior messages for this schedule):
{mock_recent_sends}

Mocked research result (None = not called, empty string = research failed):
{mock_research}

Agent's output message:
{output}

Score each criterion 1 (pass) or 0 (fail):
- message_produced: Output is a non-empty string — agent produced something.
- task_addressed: Message content relates to the scheduled task (gym = exercise, news = news items, etc.)
- personalized: Message references at least one specific detail from the mocked memory. Score 1 (N/A) if mock_memory is empty.
- tone_appropriate: Tone matches task type — reminder=brief/direct, outreach=casual/warm, digest=structured-but-conversational. No generic openers like 'Good morning!' or 'Hope you're having a great day!'
- not_repetitive: Content meaningfully differs from recent_sends — different angle or topic. Score 1 (N/A) if recent_sends is empty.
- no_preamble: Message does NOT open with meta-commentary ('Here is your reminder:', 'Here's the message:', 'Task completed:', etc.)
- research_integrated: Research content appears in the message (paraphrase OK). Score 1 (N/A) if mock_research is None.
- fallback_graceful: When mock_research is empty string (failure), message still uses memory or pivots — not an error or silence. Score 1 (N/A) if mock_research is not empty string.

Return ONLY this JSON (no markdown):
{{
  "message_produced": <0 or 1>,
  "task_addressed": <0 or 1>,
  "personalized": <0 or 1>,
  "tone_appropriate": <0 or 1>,
  "not_repetitive": <0 or 1>,
  "no_preamble": <0 or 1>,
  "research_integrated": <0 or 1>,
  "fallback_graceful": <0 or 1>,
  "notes": "<one sentence: the single most important failure, or 'all good'>"
}}
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    scenario_id: str
    task: str
    output: str
    scores: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    @property
    def total(self) -> int:
        return sum(self.scores.values())

    @property
    def max_score(self) -> int:
        return len(CRITERIA)

    @property
    def pct(self) -> float:
        return self.total / self.max_score * 100


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def _run_scenario(scenario: dict) -> str:
    mock_memory = scenario["mock_memory"]
    mock_recent_sends = scenario["mock_recent_sends"]
    mock_research = scenario["mock_research"]

    async def fake_search_memory(query: str, user_id: str) -> list[str]:
        return list(mock_memory)

    async def fake_run_research(task: str, user_id: str | None = None) -> str:
        from core.agent.researcher import ResearchBrief
        if mock_research is None:
            # Scenario doesn't need research; return a minimal brief if somehow called
            return ResearchBrief(
                summary="No specific research content available for this task.",
                sources=[],
                complete=True,
            ).model_dump_json()
        if mock_research == "":
            # Research explicitly failed — simulate empty/no-results scenario
            return ResearchBrief(
                summary="",
                sources=[],
                complete=False,
                note="No results found.",
            ).model_dump_json()
        return ResearchBrief(
            summary=mock_research,
            sources=[],
            complete=True,
        ).model_dump_json()

    async def fake_get_recent_runs(pool, *, schedule_id: int, limit: int = 7) -> list[dict]:
        return [{"message_preview": msg} for msg in mock_recent_sends]

    with (
        patch("core.memory.store.search", new=AsyncMock(side_effect=fake_search_memory)),
        patch("core.agent.researcher.run_research", new=AsyncMock(side_effect=fake_run_research)),
        patch("core.db.get_recent_runs", new=AsyncMock(side_effect=fake_get_recent_runs)),
    ):
        return await run_proactive_agent(
            scenario["task"],
            user_id="eval-user",
            schedule_id=1,
            channel_topic=scenario.get("channel_topic"),
            pool=object(),  # non-None so get_recent_sends doesn't short-circuit
        )


async def _judge(result: EvalResult, scenario: dict) -> tuple[dict[str, int], str]:
    prompt = JUDGE_PROMPT.format(
        task=result.task,
        channel_topic=scenario.get("channel_topic") or "none",
        mock_memory=json.dumps(scenario["mock_memory"], indent=2),
        mock_recent_sends=json.dumps(scenario["mock_recent_sends"], indent=2),
        mock_research=repr(scenario["mock_research"]),
        output=result.output,
    )
    response = await get_client().chat.completions.create(
        model=get_default_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    scores = {k: int(data.get(k, 0)) for k in CRITERIA}
    notes = data.get("notes", "")
    return scores, notes


async def run_eval() -> list[EvalResult]:
    results = []
    for scenario in SCENARIOS:
        print(f"  [{scenario['id']}] {scenario['task']!r:.80}")
        output = await _run_scenario(scenario)
        result = EvalResult(
            scenario_id=scenario["id"],
            task=scenario["task"],
            output=output,
        )
        result.scores, result.notes = await _judge(result, scenario)
        status = "PASS" if result.pct >= 80 else "WARN" if result.pct >= 60 else "FAIL"
        print(f"         -> {status} {result.total}/{result.max_score}  {result.notes}")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[EvalResult]) -> None:
    sep = "-" * 72
    print(f"\n{sep}")
    print("PROACTIVE AGENT EVAL REPORT")
    print(sep)

    for r in results:
        status = "+" if r.pct >= 80 else "~" if r.pct >= 60 else "x"
        print(f"\n{status} [{r.scenario_id}]  {r.pct:.0f}%  \"{r.task[:60]}\"")
        print(f"  Output:   {textwrap.shorten(r.output, 140)!r}")
        failures = [k for k, v in r.scores.items() if v == 0]
        if failures:
            print(f"  Failed:   {', '.join(failures)}")
        if r.notes and r.notes.lower() != "all good":
            print(f"  Judge:    {r.notes}")

    print(f"\n{sep}")
    print("CRITERION BREAKDOWN")
    print(sep)
    n = len(results)
    for criterion in CRITERIA:
        passed = sum(r.scores.get(criterion, 0) for r in results)
        bar = "#" * passed + "." * (n - passed)
        print(f"  {criterion:<22} {bar}  {passed}/{n}")

    overall = sum(r.total for r in results)
    max_overall = sum(r.max_score for r in results)
    pct = overall / max_overall * 100
    grade = "PASS" if pct >= 80 else "WARN" if pct >= 65 else "FAIL"
    print(f"\n{sep}")
    print(f"OVERALL: {grade}  {overall}/{max_overall}  ({pct:.1f}%)")
    print(sep)


async def main() -> None:
    print("Running proactive agent eval...\n")
    results = await run_eval()
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())

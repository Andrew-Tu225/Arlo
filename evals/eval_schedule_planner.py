"""Schedule planner sub-agent eval — measures SchedulePlan output quality.

Every scenario requires the agent to call at least one tool (list_schedules or
search_memory) to produce a correct answer. Scenarios where the output could be
derived from the request alone without tool calls are excluded by design.

Mocks list_schedules (DB) and search_memory (mem0) so no database is needed.
The real LLM runs the planner graph and the judge scores the output.

Usage:
    python -m evals.eval_schedule_planner

Requires OPENAI_API_KEY (or OPENROUTER_API_KEY) in .env.
Does NOT require Discord, a database, or Tavily.
"""

import asyncio
import json
import textwrap
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

from core.agent.schedule_planner import run_schedule_planner
from core.llm import get_client, get_default_model

# ---------------------------------------------------------------------------
# Criteria scored per scenario (1 = pass, 0 = fail)
# ---------------------------------------------------------------------------

CRITERIA = {
    "output_valid": (
        "Output is either valid SchedulePlan JSON or a plain-text clarifying question. "
        "Not malformed JSON, not empty, not a generic error string."
    ),
    "action_correct": (
        "The action field (create/edit/delete) matches the intent of the request. "
        "N/A (score 1) when output is a clarifying question."
    ),
    "cron_parseable": (
        "The cron_schedule field is a valid 5-field cron expression or HH:MM shorthand. "
        "N/A (score 1) for delete actions and clarifying questions."
    ),
    "name_matches_existing": (
        "For edit/delete: name matches the exact schedule name from list_schedules. "
        "For create: name is a concise 2–4 word slug describing the schedule. "
        "N/A (score 1) for clarifying questions."
    ),
    "task_captured": (
        "For create/edit: the task field accurately captures what the scheduled job should do. "
        "N/A (score 1) for delete and clarifying questions."
    ),
    "ambiguity_detected": (
        "When the request is genuinely ambiguous (cannot be resolved from tools), "
        "the agent returns a plain-text clarifying question — not JSON. "
        "N/A (score 1) when the request is not ambiguous."
    ),
    "tool_was_called": (
        "The agent called list_schedules and/or search_memory as required by this scenario. "
        "Scenarios are designed so correct output requires at least one tool call."
    ),
}

# ---------------------------------------------------------------------------
# Test scenarios — all require at least one tool call for correct output
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- Requires list_schedules ---
    {
        "id": "shift_earlier",
        "request": "move my gym reminder 30 minutes earlier",
        "mock_schedules": [{"name": "gym reminder", "cron": "0 7 * * 1-5", "task": "Send gym reminder", "enabled": True}],
        "mock_memory": [],
        "expected_action": "edit",
        "expect_json": True,
        "note": (
            "Agent must call list_schedules to learn cron is '0 7 * * 1-5' (7am), "
            "then output '30 6 * * 1-5' (6:30am). Cannot produce correct cron without the tool."
        ),
    },
    {
        "id": "weekend_duplicate",
        "request": "my gym reminder is only weekdays — add the same one for weekends",
        "mock_schedules": [{"name": "gym reminder", "cron": "0 7 * * 1-5", "task": "Send gym reminder", "enabled": True}],
        "mock_memory": [],
        "expected_action": "create",
        "expect_json": True,
        "note": (
            "Agent must call list_schedules to read the weekday cron (7am) and task, "
            "then create a weekend version with cron '0 7 * * 6,0'. Task and time unknown without tool."
        ),
    },
    {
        "id": "edit_existing_name_match",
        "request": "change my gym reminder to 6am instead",
        "mock_schedules": [{"name": "gym reminder", "cron": "0 7 * * 1-5", "task": "Send gym reminder", "enabled": True}],
        "mock_memory": [],
        "expected_action": "edit",
        "expect_json": True,
        "note": (
            "Agent must call list_schedules to confirm exact name 'gym reminder' exists. "
            "Name in output must match exactly — no paraphrase."
        ),
    },
    {
        "id": "edit_task_content",
        "request": "update my daily standup to also include the sprint backlog",
        "mock_schedules": [{"name": "daily standup", "cron": "0 9 * * *", "task": "Send standup summary", "enabled": True}],
        "mock_memory": [],
        "expected_action": "edit",
        "expect_json": True,
        "note": (
            "Agent must call list_schedules to get the current task text ('Send standup summary'), "
            "then produce an updated task that preserves the original and adds sprint backlog. Cron unchanged."
        ),
    },
    {
        "id": "conflict_rescheduler",
        "request": "I have a new meeting every Tuesday at 9am — do any of my reminders clash? Move the one that does.",
        "mock_schedules": [
            {"name": "daily standup", "cron": "0 9 * * *", "task": "Send standup summary", "enabled": True},
            {"name": "gym reminder", "cron": "0 7 * * 1-5", "task": "Send gym reminder", "enabled": True},
        ],
        "mock_memory": [],
        "expected_action": "edit",
        "expect_json": True,
        "note": (
            "Agent must call list_schedules to find 'daily standup' fires at 9am (clashes with Tuesday meeting). "
            "gym reminder at 7am does not clash. Output: edit daily standup to a different time."
        ),
    },
    {
        "id": "delete_by_verify",
        "request": "delete my water reminder",
        "mock_schedules": [{"name": "water reminder", "cron": "0 * * * *", "task": "Remind user to drink water", "enabled": True}],
        "mock_memory": [],
        "expected_action": "delete",
        "expect_json": True,
        "note": (
            "Agent must call list_schedules to confirm 'water reminder' exists. "
            "Name in delete output must match exactly."
        ),
    },
    {
        "id": "delete_nonexistent",
        "request": "delete my reading reminder",
        "mock_schedules": [{"name": "gym reminder", "cron": "0 7 * * 1-5", "task": "Send gym reminder", "enabled": True}],
        "mock_memory": [],
        "expected_action": None,
        "expect_json": False,
        "note": (
            "Agent calls list_schedules — 'reading reminder' not found. "
            "Must return a clarifying question or inform user it doesn't exist. Not a delete JSON."
        ),
    },
    {
        "id": "ambiguous_which_schedule",
        "request": "change my reminder",
        "mock_schedules": [
            {"name": "gym reminder", "cron": "0 7 * * 1-5", "task": "Send gym reminder", "enabled": True},
            {"name": "water reminder", "cron": "0 * * * *", "task": "Remind user to drink water", "enabled": True},
        ],
        "mock_memory": [],
        "expected_action": None,
        "expect_json": False,
        "note": (
            "Agent calls list_schedules — finds 2 schedules, can't determine which. "
            "Must ask a clarifying question that names both options."
        ),
    },
    {
        "id": "collision_detect",
        "request": "create a gym reminder at 8am",
        "mock_schedules": [{"name": "gym reminder", "cron": "0 7 * * 1-5", "task": "Send gym reminder", "enabled": True}],
        "mock_memory": [],
        "expected_action": None,
        "expect_json": False,
        "note": (
            "Agent calls list_schedules — sees 'gym reminder' already exists. "
            "Must not blindly create a duplicate. Should suggest editing the existing one or ask."
        ),
    },
    # --- Requires search_memory ---
    {
        "id": "memory_medication_timing",
        "request": "remind me to take my medication",
        "mock_schedules": [],
        "mock_memory": ["user usually wakes up at 7am and takes medication with breakfast"],
        "expected_action": "create",
        "expect_json": True,
        "note": (
            "Agent must call search_memory to learn wakeup time (7am), "
            "then schedule at or around 7am. A generic '09:00' without memory call is a fail."
        ),
    },
    {
        "id": "memory_workout_schedule",
        "request": "set up a workout reminder that fits how I usually exercise",
        "mock_schedules": [],
        "mock_memory": ["user does HIIT workouts on Monday, Wednesday, and Friday mornings at 6am"],
        "expected_action": "create",
        "expect_json": True,
        "note": (
            "Agent must call search_memory to discover Mon/Wed/Fri 6am pattern, "
            "then output cron '0 6 * * 1,3,5'. Cannot produce correct cron without the memory call."
        ),
    },
    # --- Requires BOTH list_schedules and search_memory ---
    {
        "id": "smart_no_conflict",
        "request": "add a meditation reminder but make sure it doesn't clash with any of my existing schedules",
        "mock_schedules": [
            {"name": "daily standup", "cron": "0 9 * * *", "task": "Send standup summary", "enabled": True},
            {"name": "gym reminder", "cron": "30 6 * * 1-5", "task": "Send gym reminder", "enabled": True},
        ],
        "mock_memory": ["user prefers to meditate in the evening around 8pm"],
        "expected_action": "create",
        "expect_json": True,
        "note": (
            "Agent must call list_schedules (6:30am and 9am taken) AND search_memory (user prefers 8pm). "
            "Since 8pm is free, cron should be '0 20 * * *'. Requires synthesising both tool results."
        ),
    },
]

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating the output of a schedule planner sub-agent.

User request: {request}

What list_schedules returned (mock): {mock_schedules}
What search_memory returned (mock): {mock_memory}

Did the agent call list_schedules? {called_list_schedules}
Did the agent call search_memory? {called_search_memory}

Expected output type: {expected_type}
Expected action (if JSON): {expected_action}

Agent output:
{output}

Score each criterion 1 (pass) or 0 (fail):
- output_valid: Output is valid SchedulePlan JSON or a plain-text clarifying question. Not empty or a generic error.
- action_correct: action field matches expected_action (create/edit/delete). Score 1 (N/A) if output is a clarifying question.
- cron_parseable: cron_schedule is a valid 5-field cron or HH:MM shorthand. Score 1 (N/A) for delete actions and clarifying questions.
- name_matches_existing: For edit/delete: name exactly matches a name from list_schedules. For create: name is a descriptive slug. Score 1 (N/A) for clarifying questions.
- task_captured: For create/edit: task field accurately describes what the job should do. Score 1 (N/A) for delete and clarifying questions.
- ambiguity_detected: When request could not be resolved from tool results alone, output is a plain-text clarifying question (not JSON). Score 1 (N/A) when request was clear.
- tool_was_called: Agent called list_schedules and/or search_memory as the scenario requires. Score 1 if the required tool was called. Score 0 if neither was called.

Return ONLY this JSON (no markdown):
{{
  "output_valid": <0 or 1>,
  "action_correct": <0 or 1>,
  "cron_parseable": <0 or 1>,
  "name_matches_existing": <0 or 1>,
  "task_captured": <0 or 1>,
  "ambiguity_detected": <0 or 1>,
  "tool_was_called": <0 or 1>,
  "notes": "<one sentence: the single most important failure, or 'all good'>"
}}
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    scenario_id: str
    request: str
    output: str
    scores: dict[str, int] = field(default_factory=dict)
    notes: str = ""
    called_list_schedules: bool = False
    called_search_memory: bool = False

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


async def _run_scenario(scenario: dict) -> EvalResult:
    mock_schedules = scenario["mock_schedules"]
    mock_memory = scenario["mock_memory"]

    list_schedules_called = False
    search_memory_called = False

    async def fake_list_schedules(pool, user_id: str) -> str:
        nonlocal list_schedules_called
        list_schedules_called = True
        return json.dumps(mock_schedules)

    async def fake_search_memory(query: str, user_id: str) -> list[str]:
        nonlocal search_memory_called
        search_memory_called = True
        # store.search returns list[str] — extract "memory" field if dict, else pass through
        return [m["memory"] if isinstance(m, dict) else m for m in mock_memory]

    with (
        patch("core.tools.schedules.list_schedules", new=AsyncMock(side_effect=fake_list_schedules)),
        patch("core.memory.store.search", new=AsyncMock(side_effect=fake_search_memory)),
    ):
        output = await run_schedule_planner(
            scenario["request"],
            user_id="eval-user",
            pool=object(),  # non-None pool so list_schedules is not skipped
        )

    return EvalResult(
        scenario_id=scenario["id"],
        request=scenario["request"],
        output=output,
        called_list_schedules=list_schedules_called,
        called_search_memory=search_memory_called,
    )


async def _judge(result: EvalResult, scenario: dict) -> tuple[dict[str, int], str]:
    expected_type = "JSON" if scenario["expect_json"] else "clarifying question"
    prompt = JUDGE_PROMPT.format(
        request=result.request,
        mock_schedules=json.dumps(scenario["mock_schedules"], indent=2),
        mock_memory=json.dumps(scenario["mock_memory"], indent=2),
        called_list_schedules=result.called_list_schedules,
        called_search_memory=result.called_search_memory,
        expected_type=expected_type,
        expected_action=scenario.get("expected_action", "N/A"),
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
        print(f"  [{scenario['id']}] {scenario['request']!r:.80}")
        result = await _run_scenario(scenario)
        result.scores, result.notes = await _judge(result, scenario)
        status = "PASS" if result.pct >= 80 else "WARN" if result.pct >= 60 else "FAIL"
        tools_called = []
        if result.called_list_schedules:
            tools_called.append("list_schedules")
        if result.called_search_memory:
            tools_called.append("search_memory")
        tools_str = ", ".join(tools_called) if tools_called else "none"
        print(f"         -> {status} {result.total}/{result.max_score}  tools=[{tools_str}]  {result.notes}")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[EvalResult]) -> None:
    sep = "-" * 72
    print(f"\n{sep}")
    print("SCHEDULE PLANNER EVAL REPORT")
    print(sep)

    for r in results:
        status = "+" if r.pct >= 80 else "~" if r.pct >= 60 else "x"
        tools = []
        if r.called_list_schedules:
            tools.append("list_schedules")
        if r.called_search_memory:
            tools.append("search_memory")
        print(f"\n{status} [{r.scenario_id}]  {r.pct:.0f}%  \"{r.request[:60]}\"")
        print(f"  Output:   {textwrap.shorten(r.output, 120)!r}")
        print(f"  Tools:    {', '.join(tools) if tools else 'none called'}")
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
        print(f"  {criterion:<26} {bar}  {passed}/{n}")

    overall = sum(r.total for r in results)
    max_overall = sum(r.max_score for r in results)
    pct = overall / max_overall * 100
    grade = "PASS" if pct >= 80 else "WARN" if pct >= 65 else "FAIL"
    print(f"\n{sep}")
    print(f"OVERALL: {grade}  {overall}/{max_overall}  ({pct:.1f}%)")
    print(sep)


async def main() -> None:
    print("Running schedule planner eval...\n")
    results = await run_eval()
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())

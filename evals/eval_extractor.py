"""Extractor eval — measures how well _extract_facts pulls relevant facts from real conversations.

Each scenario is a ~10-message multi-topic window (the actual input size the extractor
sees at PROFILE_EXTRACTION_INTERVAL=10). Topics shift naturally within each scenario —
the extractor must pull facts across all topic areas, not just the dominant thread.

Scoring uses an LLM judge (same pattern as eval_persona.py). Results are printed for
human review plus summarised in a criterion breakdown table.

Usage:
    python -m evals.eval_extractor

Requires OPENAI_API_KEY (or OPENROUTER_API_KEY) in .env.
Does NOT require Discord or a database.
"""

import asyncio
import json
from dataclasses import dataclass, field

from core.llm import get_client, get_default_model
from core.memory.extractor import _extract_facts

# ---------------------------------------------------------------------------
# Criteria scored per scenario (1 = pass, 0 = fail)
# ---------------------------------------------------------------------------

CRITERIA = {
    "recall": (
        "All significant facts about the user were captured — no major fact a perceptive "
        "friend would notice was missed."
    ),
    "no_hallucination": (
        "Every extracted fact is clearly supported by the conversation. "
        "Nothing was invented or inferred far beyond what the conversation reasonably implies."
    ),
    "no_third_party": (
        "Facts about other people (partner, friends, family, colleagues) are NOT attributed "
        "to the user. Only facts about the user are stored."
    ),
    "short_term_tagging": (
        "is_short_term=true for time-bound or situation-specific facts (upcoming events, "
        "current mood). is_short_term=false for stable traits and ongoing characteristics. "
        "PASS if tagging is consistently accurate. FAIL if multiple tags are wrong."
    ),
    "multi_topic_coverage": (
        "When the conversation shifts between multiple topics, facts are extracted from ALL "
        "topic areas — not just the first or most dominant thread."
    ),
    "value_quality": (
        "Values are complete, self-contained phrases rather than bare nouns. "
        "e.g. 'follows a vegetarian diet' not just 'vegetarian'. "
        "PASS if most values read as complete statements. FAIL if many are single words."
    ),
    "usefulness": (
        "The extracted facts are genuinely useful for personalising Arlo's future responses, "
        "recommendations, and proactive content. "
        "PASS: facts would meaningfully change how Arlo responds or what it recommends "
        "(e.g. 'hates cilantro', 'training for a half marathon', 'works in fintech'). "
        "FAIL: facts are too vague to act on, trivially obvious, or wouldn't change anything "
        "(e.g. 'user said something', 'user is a person', 'user communicated with Arlo')."
    ),
}

# ---------------------------------------------------------------------------
# Scenarios — ~10-message multi-topic windows
# Each represents a different type of user and covers 3-4 distinct topic shifts.
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "developer_winding_down",
        "description": "Developer finishing a long bug-fix day — work → food preference → entertainment → gym habit → tech decision",
        "conversation": (
            "[user] finally fixed that auth bug, been on it since 9am\n"
            "[assistant] ugh prod-only auth bugs are the absolute worst, glad that's done\n"
            "[user] jwt tokens weren't refreshing in prod only, took ages to even reproduce\n"
            "[assistant] classic. ok you need to do absolutely nothing tonight\n"
            "[user] yeah ordering thai and watching something, too tired to think\n"
            "[assistant] solid call. Succession if you haven't finished it, those last episodes are unreal\n"
            "[user] been meaning to finish it, literally 3 episodes left\n"
            "[assistant] do not get spoiled you're so close\n"
            "[user] skipped the gym this morning too, just needed sleep more than anything\n"
            "[assistant] rest day was the right move, a full day on one bug earns that\n"
            "[user] yeah, also been going back and forth on switching the backend to Rust but not sure yet"
        ),
    },
    {
        "id": "social_planner_trip",
        "description": "Planning a trip — relationship + partner's diet → personal interests → work stress → travel style",
        "conversation": (
            "[user] trying to plan a weekend trip to Montreal with my girlfriend for early July\n"
            "[assistant] Montreal in summer is actually perfect timing, good call\n"
            "[user] she's vegan so finding restaurants that work for both of us is always a mission\n"
            "[assistant] surprisingly solid vegan scene there honestly, easier than most cities\n"
            "[user] good to know. I'm more into architecture and food spots so hopefully we can make both work\n"
            "[assistant] Old Montreal alone is worth the trip for that, great combo\n"
            "[user] yeah mid-range budget, not trying to go bougie but not backpacking either\n"
            "[assistant] plenty of good options in that range\n"
            "[user] I usually like having a rough plan, she's more spontaneous so we compromise\n"
            "[assistant] been a while coming though — you two have been slammed at work?\n"
            "[user] yeah both of us kept pushing it, finally just booking it"
        ),
    },
    {
        "id": "fitness_tracker",
        "description": "Fitness-focused person — training goal → nutrition tracking → podcast opinions → reading goal",
        "conversation": (
            "[user] just got back from a 10k, legs are completely done\n"
            "[assistant] 10k is no joke, legs should be done\n"
            "[user] building up for a half marathon in October, third one\n"
            "[assistant] ok veteran status, third is where it starts to feel dialled in\n"
            "[user] yeah and been more intentional with nutrition too, hitting 150g protein a day\n"
            "[assistant] that's a real commitment, tracking it or estimating\n"
            "[user] Cronometer, been using it for like 2 years now, really helps\n"
            "[assistant] lol Huberman would be proud\n"
            "[user] used to listen to him constantly but burned out honestly, switched to Lex Fridman\n"
            "[assistant] very different vibe but understandable, Huberman can be a lot\n"
            "[user] yeah. also trying to hit 20 books this year, on number 7 right now, mostly non-fiction"
        ),
    },
    {
        "id": "creative_day_job_tension",
        "description": "Creative with a day job — side project + taste → job stress → upcoming milestone → comfort media",
        "conversation": (
            "[user] finally finished the first draft of my short film script, took months\n"
            "[assistant] that's a real milestone, first draft is the hardest part out of the way\n"
            "[user] slow burn psychological horror, kind of inspired by Hereditary\n"
            "[assistant] excellent taste, that film is terrifying in exactly the right way\n"
            "[user] yeah that's the feeling I want. my day job is UX at a fintech startup, totally different world\n"
            "[assistant] the classic arrangement lol, fintech deadlines are brutal I imagine\n"
            "[user] yeah got a product launch in two weeks, pretty chaotic right now\n"
            "[assistant] decompress mode is essential then, what are you watching\n"
            "[user] rewatching The Sopranos for the third time, still absolutely holds up\n"
            "[assistant] third rewatch is where you really start catching everything\n"
            "[user] yeah and turning 30 next month so slightly spiraling about life in general lol"
        ),
    },
    {
        "id": "tech_curious_traveller",
        "description": "Tech-curious traveller — new hardware → creative hobby → travel depth → AI tools → food quirk",
        "conversation": (
            "[user] just got the M4 MacBook Pro, upgrading from 2019\n"
            "[assistant] that's going to feel like a different planet honestly\n"
            "[user] yeah was bottlenecking my video editing, doing travel vlogs on YouTube\n"
            "[assistant] 14 countries of backlog sounds like solid content\n"
            "[user] lol yeah been to 14, Japan twice — honestly would move there if I could\n"
            "[assistant] Japan is in a different tier, completely get it\n"
            "[user] obsessed. also started using Cursor for my coding side projects\n"
            "[assistant] Cursor is genuinely good once you get used to it\n"
            "[user] feels like cheating still but I'm into it, use ChatGPT for scripting too\n"
            "[assistant] that feeling disappears fast, just becomes the workflow\n"
            "[user] lol true. btw if you ever recommend food spots I love street food, only rule is absolutely no cilantro"
        ),
    },
]

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating how well an AI system extracted user facts from a conversation.

Context: These extracted facts become the long-term memory of Arlo, a personal AI \
companion designed to act like a smart, perceptive friend. The facts are used to \
personalise responses, give better recommendations, and proactively surface content \
relevant to the user. Good extraction must capture facts across ALL topic areas \
the conversation touched — conversations jump topics just like real texting does.

Conversation:
{conversation}

Extracted facts:
{facts_formatted}

Score each criterion as 1 (pass) or 0 (fail):

- recall: all significant user facts captured; no major fact a perceptive friend would notice was missed
- no_hallucination: every fact clearly supported by the conversation; nothing invented
- no_third_party: facts about others (partner, friends, family) NOT attributed to the user
- short_term_tagging: is_short_term tags accurate (true=time-bound, false=stable trait)
- multi_topic_coverage: facts extracted from ALL topic areas, not just the first/dominant thread
- value_quality: values are complete phrases not bare nouns (e.g. "follows a vegetarian diet" not "vegetarian")
- usefulness: facts would meaningfully change how Arlo responds or what it recommends; FAIL if most facts are too vague or trivial to act on

Return ONLY this JSON:
{{
  "recall": <0 or 1>,
  "no_hallucination": <0 or 1>,
  "no_third_party": <0 or 1>,
  "short_term_tagging": <0 or 1>,
  "multi_topic_coverage": <0 or 1>,
  "value_quality": <0 or 1>,
  "usefulness": <0 or 1>,
  "missed_facts": "<comma-separated list of significant facts that were missed, or 'none'>",
  "hallucinated_facts": "<comma-separated list of invented or unsupported facts, or 'none'>",
  "useless_facts": "<comma-separated list of extracted facts that are too vague or trivial to be useful, or 'none'>",
  "notes": "<one sentence: the single most important thing the extraction got wrong, or 'all good'>"
}}
"""


# ---------------------------------------------------------------------------
# Data model + runner
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    scenario_id: str
    description: str
    facts: list[dict]
    scores: dict[str, int] = field(default_factory=dict)
    missed_facts: str = ""
    hallucinated_facts: str = ""
    useless_facts: str = ""
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


def _format_facts(facts: list[dict]) -> str:
    if not facts:
        return "  (no facts extracted)"
    lines = []
    for f in facts:
        tag = "[ST]" if f.get("is_short_term") else "[LT]"
        lines.append(f"  {tag}  {f.get('dimension', '?'):<16} {f.get('value', '?')}")
    return "\n".join(lines)


async def _judge(conversation: str, facts: list[dict]) -> tuple[dict[str, int], str, str, str]:
    facts_formatted = _format_facts(facts)
    prompt = JUDGE_PROMPT.format(conversation=conversation, facts_formatted=facts_formatted)
    result = await get_client().chat.completions.create(
        model=get_default_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = result.choices[0].message.content or "{}"
    data = json.loads(raw)
    scores = {k: int(data.get(k, 0)) for k in CRITERIA}
    return (
        scores,
        data.get("missed_facts", "none"),
        data.get("hallucinated_facts", "none"),
        data.get("useless_facts", "none"),
        data.get("notes", ""),
    )


_RESULTS_DIR = "evals/results"


async def run_print_only() -> None:
    import os
    from datetime import datetime

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{_RESULTS_DIR}/extractor_{timestamp}.txt"

    sep = "-" * 72
    lines: list[str] = []

    for s in SCENARIOS:
        print(f"  extracting [{s['id']}]...")
        facts = await _extract_facts(s["conversation"])
        block = [
            f"\n{sep}",
            f"[{s['id']}]",
            f"  {s['description']}",
            f"\n  Extracted facts ({len(facts)}):",
            _format_facts(facts),
        ]
        lines.extend(block)

    lines.append(f"\n{sep}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResults written to {out_path}")


async def run_with_judge() -> list[EvalResult]:
    results = []
    for s in SCENARIOS:
        print(f"  [{s['id']}]")
        facts = await _extract_facts(s["conversation"])
        scores, missed, hallucinated, useless, notes = await _judge(s["conversation"], facts)
        result = EvalResult(
            scenario_id=s["id"],
            description=s["description"],
            facts=facts,
            scores=scores,
            missed_facts=missed,
            hallucinated_facts=hallucinated,
            useless_facts=useless,
            notes=notes,
        )
        results.append(result)
        status = "PASS" if result.pct >= 80 else "WARN" if result.pct >= 60 else "FAIL"
        print(f"         -> {status} {result.total}/{result.max_score}  {notes}")
    return results


def print_judge_report(results: list[EvalResult]) -> None:
    sep = "-" * 72

    for r in results:
        status = "+" if r.pct >= 80 else "~" if r.pct >= 60 else "x"
        print(f"\n{sep}")
        print(f"{status} [{r.scenario_id}]  {r.pct:.0f}%")
        print(f"  {r.description}")
        print(f"\n  Extracted facts ({len(r.facts)}):")
        print(_format_facts(r.facts))
        failures = [k for k, v in r.scores.items() if v == 0]
        if failures:
            print(f"\n  Failed:       {', '.join(failures)}")
        if r.missed_facts and r.missed_facts.lower() != "none":
            print(f"  Missed:       {r.missed_facts}")
        if r.hallucinated_facts and r.hallucinated_facts.lower() != "none":
            print(f"  Hallucinated: {r.hallucinated_facts}")
        if r.useless_facts and r.useless_facts.lower() != "none":
            print(f"  Useless:      {r.useless_facts}")
        if r.notes and r.notes.lower() != "all good":
            print(f"  Judge note:   {r.notes}")

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
    import argparse
    parser = argparse.ArgumentParser(description="Extractor eval")
    parser.add_argument("--judge", action="store_true", help="Run LLM judge scoring after extraction")
    args = parser.parse_args()

    if args.judge:
        print("Running extractor eval with judge...\n")
        results = await run_with_judge()
        print_judge_report(results)
    else:
        print("Running extractor eval (results written to file)...\n")
        await run_print_only()


if __name__ == "__main__":
    asyncio.run(main())

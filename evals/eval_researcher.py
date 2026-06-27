"""Researcher sub-agent eval — measures ResearchBrief output quality.

Runs each scenario through the real LLM (researcher graph) with mocked
web_search and read_url tools. Mocking keeps results deterministic: the eval
tests LLM reasoning and synthesis, not Tavily availability. The judge also
receives the mock data so it can score sources_grounded and no_hallucination
authoritatively.

The tool_discipline criterion is evaluated by wrapping the underlying search
and reader modules with call counters before each scenario runs.

Usage:
    python -m evals.eval_researcher

Requires OPENAI_API_KEY (or OPENROUTER_API_KEY) in .env.
Does NOT require Discord, a database, or a Tavily key.
"""

import asyncio
import json
import textwrap
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

from core.agent.researcher import run_research
from core.llm import get_client, get_default_model

# ---------------------------------------------------------------------------
# Criteria scored per scenario (1 = pass, 0 = fail)
# ---------------------------------------------------------------------------

CRITERIA = {
    "schema_valid": (
        "Output parses as valid JSON with summary (str), sources (list), complete (bool). "
        "SourceItems may be dicts with url+title or plain strings."
    ),
    "summary_substantive": (
        "Summary is >60 chars and contains specific, concrete information — "
        "not 'I searched and found…' or vague filler."
    ),
    "task_addressed": (
        "Summary content directly addresses what was asked — not tangential or off-topic."
    ),
    "sources_grounded": (
        "Every URL cited in sources appears in the mocked search results or read_url data. "
        "No invented URLs. At least one source present when complete=true."
    ),
    "tool_discipline": (
        "web_search was called at least once. read_url called at most 2 times. "
        "No sign of excessive or redundant identical queries."
    ),
    "complete_flag_accurate": (
        "complete=true when task had a findable answer from the mocked data. "
        "complete=false when mocked search returned empty or ceiling was forced."
    ),
    "no_hallucination": (
        "No claims in the summary contradict or go far beyond what the mocked "
        "search results and read_url content actually say."
    ),
}

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "news_ai",
        "task": "What are the top AI news stories today?",
        "mock_search": [
            {"url": "https://techcrunch.com/2026/06/ai-roundup", "title": "AI Roundup: GPT-5 Released", "snippet": "OpenAI released GPT-5 today with a 2M context window and significantly improved reasoning."},
            {"url": "https://arstechnica.com/ai/2026/06/anthropic", "title": "Anthropic Raises $3B Series E", "snippet": "Anthropic secured a $3B round led by Google, valuing the company at $45B."},
            {"url": "https://wired.com/story/llm-benchmarks-2026", "title": "New LLM Benchmarks Drop", "snippet": "MMLU-Pro scores show Claude 4 and GPT-5 trading the top spot on reasoning tasks."},
        ],
        "mock_read": None,
        "note": "News type — should synthesize 3 results into a multi-item summary with all 3 URLs.",
    },
    {
        "id": "factual_simple",
        "task": "What year was Python first released?",
        "mock_search": [
            {"url": "https://python.org/doc/history", "title": "Python History", "snippet": "Python 0.9.0 was released in February 1991 by Guido van Rossum."},
        ],
        "mock_read": None,
        "note": "Simple factual — one search result is enough; concise summary with 1991 and Guido van Rossum.",
    },
    {
        "id": "technical_deep",
        "task": "How does the attention mechanism in transformers work?",
        "mock_search": [
            {"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need", "snippet": "Queries, keys, and values are used to compute attention weights via softmax over dot products."},
            {"url": "https://jalammar.github.io/illustrated-transformer", "title": "The Illustrated Transformer", "snippet": "Each word in the sequence attends to every other word using learned Q, K, V projections."},
        ],
        "mock_read": {
            "url": "https://jalammar.github.io/illustrated-transformer",
            "content": (
                "The attention score is computed as softmax(QK^T / sqrt(d_k)) * V. "
                "Multi-head attention runs this operation h times in parallel with different projections, "
                "then concatenates and linearly projects the results. This allows the model to jointly "
                "attend to information from different representation subspaces."
            ),
        },
        "note": "Technical type — should call read_url on the detailed source; summary explains QKV and softmax.",
    },
    {
        "id": "price_check",
        "task": "What is the current GPT-4o pricing per 1M tokens?",
        "mock_search": [
            {"url": "https://openai.com/api/pricing", "title": "OpenAI API Pricing", "snippet": "GPT-4o: $5.00 per 1M input tokens, $15.00 per 1M output tokens. Batch API: 50% discount."},
        ],
        "mock_read": None,
        "note": "Price type — specific dollar amounts must appear in summary; source must be the pricing URL.",
    },
    {
        "id": "person_entity",
        "task": "Who is Andrej Karpathy and what is he known for professionally?",
        "mock_search": [
            {"url": "https://en.wikipedia.org/wiki/Andrej_Karpathy", "title": "Andrej Karpathy - Wikipedia", "snippet": "Andrej Karpathy is a Slovak-Canadian computer scientist. He was Director of AI at Tesla and a founding member of OpenAI. Known for his Stanford CS231n course and the micrograd/nanoGPT educational projects."},
        ],
        "mock_read": None,
        "note": "Entity type — summary should name Tesla AI, OpenAI, and CS231n from the mock data.",
    },
    {
        "id": "analysis_compare",
        "task": "Compare React vs Vue.js for building a small startup's landing page — which should they pick?",
        "mock_search": [
            {"url": "https://dev.to/react-vs-vue-landing", "title": "React vs Vue for Landing Pages", "snippet": "React has a larger ecosystem and more job market demand. Vue has a gentler learning curve and simpler single-file component model."},
            {"url": "https://medium.com/startup-stack-2026", "title": "Framework Choices for Startups", "snippet": "For landing pages with limited dev time, Vue or even plain HTML/CSS often beats React. React shines when you need a full SPA with complex state."},
        ],
        "mock_read": None,
        "note": "Analysis type — should synthesize both sources and commit to a recommendation for a landing page.",
    },
    {
        "id": "multi_step",
        "task": "What are the best open-source Python PDF parsing libraries? Give a code example.",
        "mock_search": [
            {"url": "https://github.com/jsvine/pdfplumber", "title": "pdfplumber — GitHub", "snippet": "pdfplumber: Plumb a PDF for detailed information about each text character, rectangle, and line. Excellent for table extraction."},
            {"url": "https://pypdf.readthedocs.io", "title": "pypdf Documentation", "snippet": "pypdf (formerly PyPDF2): pure-Python library for reading and writing PDF files. Handles text extraction, page splitting, merging."},
        ],
        "mock_read": {
            "url": "https://github.com/jsvine/pdfplumber",
            "content": (
                "Installation: pip install pdfplumber\n\n"
                "Basic usage:\n"
                "import pdfplumber\n"
                "with pdfplumber.open('document.pdf') as pdf:\n"
                "    first_page = pdf.pages[0]\n"
                "    text = first_page.extract_text()\n"
                "    tables = first_page.extract_tables()\n\n"
                "For table extraction, pdfplumber is generally superior to pypdf."
            ),
        },
        "note": "Multi-step — expects web_search first, then read_url on pdfplumber for the code example.",
    },
    {
        "id": "empty_results",
        "task": "What are the latest earnings results for Globex Corporation Q1 2026?",
        "mock_search": [],
        "mock_read": None,
        "note": "No results — complete=false expected; summary honestly states nothing was found.",
    },
    {
        "id": "ceiling_behavior",
        "task": "Write a comprehensive history of every major programming language created since 1950.",
        "mock_search": [
            {"url": "https://computerhistory.org/programming-languages", "title": "History of Programming Languages", "snippet": "Fortran (1957), COBOL (1959), LISP (1958), ALGOL (1960), BASIC (1964), C (1972)..."},
        ],
        "mock_read": None,
        "max_iterations": 2,  # Force ceiling hit to test complete=false
        "note": "Ceiling test — task too broad for 2 iterations; complete=false expected.",
    },
    {
        "id": "ambiguous_but_tries",
        "task": "Tell me about it",
        "mock_search": [
            {"url": "https://en.wikipedia.org/wiki/It_(pronoun)", "title": "It (pronoun) - Wikipedia", "snippet": "In English, 'it' is a third-person singular neuter pronoun used to refer to inanimate objects or concepts."},
        ],
        "mock_read": None,
        "note": "Vague task — agent should attempt a search and return whatever it finds. complete=false or partial acceptable.",
    },
]

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating the output of a research sub-agent (ResearchBrief).

Task given to the agent: {task}

Mocked search results the agent had access to:
{mock_search}

Mocked read_url content (if any): {mock_read}

Tool call log:
- web_search called: {web_search_calls} time(s)
- read_url called: {read_url_calls} time(s)

Ceiling forced (max_iterations=2): {ceiling_forced}

Agent ResearchBrief output (JSON string):
{output}

Score each criterion 1 (pass) or 0 (fail):
- schema_valid: Output parses as JSON with summary (str), sources (list), complete (bool). SourceItems may be dicts or strings.
- summary_substantive: Summary is >60 chars and contains specific, concrete details — not vague filler like "I searched and found some results"
- task_addressed: Summary directly addresses the task — not tangential info
- sources_grounded: Every source URL cited in the output appears in the mocked search results or read_url data. No invented URLs. At least one source when complete=true.
- tool_discipline: web_search was called at least once (from log). read_url called at most 2 times. No excessive identical queries.
- complete_flag_accurate: complete=true if mocked data contained a clear answer; complete=false if ceiling was forced or search returned no results
- no_hallucination: No claims contradict or go far beyond what the mocked search results and read_url content actually say

Return ONLY this JSON (no markdown):
{{
  "schema_valid": <0 or 1>,
  "summary_substantive": <0 or 1>,
  "task_addressed": <0 or 1>,
  "sources_grounded": <0 or 1>,
  "tool_discipline": <0 or 1>,
  "complete_flag_accurate": <0 or 1>,
  "no_hallucination": <0 or 1>,
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
    web_search_calls: int = 0
    read_url_calls: int = 0

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
    mock_search_results = scenario["mock_search"]
    mock_read_content = scenario.get("mock_read")
    max_iterations = scenario.get("max_iterations")

    web_search_call_count = 0
    read_url_call_count = 0

    async def fake_web_search(query: str) -> list[dict]:
        nonlocal web_search_call_count
        web_search_call_count += 1
        return mock_search_results

    async def fake_read_url(url: str) -> str:
        nonlocal read_url_call_count
        read_url_call_count += 1
        if mock_read_content and url == mock_read_content["url"]:
            return mock_read_content["content"]
        return f"[No content mocked for {url}]"

    search_patch = patch("core.tools.search.web_search", new=AsyncMock(side_effect=fake_web_search))
    reader_patch = patch("core.tools.reader.read_url", new=AsyncMock(side_effect=fake_read_url))

    with search_patch, reader_patch:
        if max_iterations is not None:
            # Build a capped graph and swap in for this scenario only
            from core.agent import researcher
            from core.agent.react import ReactGraphConfig, build_react_graph
            from core.agent.prompts import get_temporal_context
            from core.agent.tools import build_research_tools, get_research_schemas

            def _build_capped_prompt(_: dict) -> str:
                return researcher.RESEARCH_SYSTEM_PROMPT + "\n" + get_temporal_context()

            capped_graph = build_react_graph(
                ReactGraphConfig(
                    build_system_prompt=_build_capped_prompt,
                    tool_schemas=get_research_schemas(),
                    tool_builder=build_research_tools,
                    max_react_iterations=max_iterations,
                    task_token_budget=12000,
                ),
                checkpointer=None,
            )
            with patch("core.agent.researcher.research_agent_graph", new=capped_graph):
                output = await run_research(scenario["task"], user_id="eval-user")
        else:
            output = await run_research(scenario["task"], user_id="eval-user")

    return EvalResult(
        scenario_id=scenario["id"],
        task=scenario["task"],
        output=output,
        web_search_calls=web_search_call_count,
        read_url_calls=read_url_call_count,
    )


async def _judge(result: EvalResult, scenario: dict) -> tuple[dict[str, int], str]:
    mock_search_str = json.dumps(scenario["mock_search"], indent=2)
    mock_read_str = json.dumps(scenario.get("mock_read"), indent=2)
    ceiling_forced = scenario.get("max_iterations") is not None

    prompt = JUDGE_PROMPT.format(
        task=result.task,
        mock_search=mock_search_str,
        mock_read=mock_read_str,
        web_search_calls=result.web_search_calls,
        read_url_calls=result.read_url_calls,
        ceiling_forced=ceiling_forced,
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
        result = await _run_scenario(scenario)
        result.scores, result.notes = await _judge(result, scenario)
        status = "PASS" if result.pct >= 80 else "WARN" if result.pct >= 60 else "FAIL"
        print(f"         -> {status} {result.total}/{result.max_score}  (web_search×{result.web_search_calls} read_url×{result.read_url_calls})  {result.notes}")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[EvalResult]) -> None:
    sep = "-" * 72
    print(f"\n{sep}")
    print("RESEARCHER EVAL REPORT")
    print(sep)

    for r in results:
        status = "+" if r.pct >= 80 else "~" if r.pct >= 60 else "x"
        print(f"\n{status} [{r.scenario_id}]  {r.pct:.0f}%  \"{r.task[:60]}\"")
        print(f"  Output:   {textwrap.shorten(r.output, 120)!r}")
        print(f"  Calls:    web_search×{r.web_search_calls}  read_url×{r.read_url_calls}")
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
    print("Running researcher eval...\n")
    results = await run_eval()
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())

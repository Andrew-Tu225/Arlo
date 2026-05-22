# Arlo — CLAUDE.md

Arlo is a persistent AI companion Discord bot. It learns the user over time, proactively surfaces relevant content, and executes real-world tasks via a ReAct web-search loop — designed to feel like texting a smart friend, not a chatbot.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in env vars
cp .env.example .env

# Run the bot (development)
python -m core.interfaces.discord.bot

# Run with FastAPI health server
uvicorn core.api:app --reload
```

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | OpenAI (GPT-4o) or OpenRouter models | Abstracted behind `core/llm.py` — pick one per deployment |
| Agent orchestration | LangGraph | ReAct loop; max 8 iterations |
| Memory abstraction | mem0 | Self-hosted library backed by PostgreSQL + pgvector |
| Database | PostgreSQL + pgvector | Run locally via Docker Compose; pgvector extension for semantic recall |
| Web search | Tavily API | Primary search tool in ReAct loop |
| URL reading | Jina Reader (`r.jina.ai/{url}`) | Lightweight page content fetcher |
| Discord interface | discord.py | Single-server bot for MVP |
| Proactive scheduler | APScheduler | Daily digest; in-process, not distributed |
| Backend | FastAPI | Health check only for MVP; hook for future web dashboard |
| Deployment | Docker Compose | Default for self-hosters; works on any VPS or local machine |

**Not in MVP:** Firecrawl (Jina handles enough), Telegram, browser automation, any managed cloud service as a hard dependency.

## Architecture

Two parallel subsystems run inside the same process: **Reactive** (user-triggered) and **Proactive** (scheduler-triggered).

### Subsystem A — Reactive (user-triggered)

```
User sends Discord message
    ↓
bot.py: on_message fires
    ↓
handlers.py:
  - Drop bot's own messages, messages outside DISCORD_GUILD_ID, empty messages
  - Build context window: fetch last CONTEXT_WINDOW_SIZE (default: 12) messages
  - Start Discord typing indicator
    ↓
classifier.py — ONE LLM call (structured output)
  → { tone: casual|task|venting|excited, intent: chat|task|memory_update }
  → on parse failure: default to { casual, chat } — never crash
    ↓
Route by intent (tone passes through to persona builder, does not change routing):

  [chat]
    store.py: embed message → pgvector similarity search → top-K relevant memories
    persona.py: build system prompt (anti-bot persona rules + injected memories + tone hint)
    llm.py: generate reply
    → Discord: send plain text reply

  [task]
    guardrail check: reject harmful/impossible requests before the loop starts
    planner.py: ONE LLM call decomposes goal into ordered step list
    orchestrator.py: LangGraph ReAct loop (hard ceiling MAX_REACT_ITERATIONS=8)
      Reason node → choose next action
        web_search() via Tavily
        read_url() via Jina Reader
          ↑ URL validated here: reject non-http(s), private IPs, malformed — no raw LLM URLs
      Synthesize node → LLM compiles final answer + inline source URLs
      On cap/budget hit without answer → "couldn't find a reliable source" (no hallucination)
    → Discord: send answer with sources

  [memory_update]
    store.py: mem0.add() — contradiction resolution, short-term vs long-term tag
    → Discord: short natural ack ("got it")

    ↓
[after reply — non-blocking]
  message_count % PROFILE_EXTRACTION_INTERVAL == 0?
    → asyncio.create_task(extractor.py):
        pull last N messages from episodic log
        ONE LLM call extracts facts (interests, preferences, opinions, habits)
        mem0.add() for each fact with short-term/long-term tag
```

### Subsystem B — Proactive (APScheduler-triggered)

The Discord bot is a long-running process. APScheduler runs jobs inside that process and can send Discord messages at any time without a user prompt. This is how all proactive outreach works — no MCP or webhooks needed for MVP.

```
APScheduler daily job fires (user-configurable time, default: 9am)
    ↓
digest.py:
  1. mem0.search("interests preferences habits") → user profile snapshot
  2. Build Tavily queries from profile interests
     e.g. "AI news today", "NBA highlights", "r/ProgrammerHumor top posts"
  3. Tavily search → news, Reddit posts, trending topics in user's niches
  4. LLM composes casual digest — 2–4 items, references ≥1 profile fact
  5. discord.get_channel(channel_id).send(message)

Job config (schedule, channel, on/off state) persisted in PostgreSQL.
On bot restart: re-read config from DB and re-register the job.
/digest off → pause; /digest on → resume.
```

### Subsystem C — Slash Commands

```
/start      → onboarding: 5–7 questions, answers saved directly to mem0
/profile    → mem0.search("*") → formatted readable summary → Discord reply
/forget X   → mem0.delete() for facts matching topic X → ack
/digest on|off → toggle APScheduler digest job, persist state to DB
```

### ReAct Loop Detail

```
planner.py: goal → ordered step list (ONE LLM call)
    ↓
orchestrator.py: LangGraph graph
  ┌─ Reason node: LLM decides next tool call
  │
  ├─ web_search(query) → Tavily API → list of {url, snippet}
  │
  ├─ read_url(url)     → validate URL → Jina Reader (r.jina.ai/{url}) → page text
  │                      (reject if: not http(s), private IP, malformed, unresolvable)
  │
  └─ synthesize()      → LLM compiles all observations into final answer + source URLs
    ↓
Exit conditions (whichever comes first):
  - synthesize reached
  - MAX_REACT_ITERATIONS=8 hit → honest "couldn't find a reliable source"
  - TASK_TOKEN_BUDGET exceeded → same honest fallback
```

### Memory Layers

| Layer | What's stored | Storage | How retrieved |
|---|---|---|---|
| Short-term | Last CONTEXT_WINDOW_SIZE Discord messages | In-context (deque) | Passed directly in prompt |
| Long-term semantic | Facts, preferences, traits (vector) | PostgreSQL + pgvector | mem0 similarity search — top-K per message |
| Long-term structured | Same facts in structured form | PostgreSQL | mem0 query by dimension |
| Episodic | Raw interaction log | PostgreSQL | Read by extraction job after N messages |

**How embeddings work:** mem0 converts each stored fact into a vector (array of numbers) representing its meaning. Similar texts get nearby vectors. When Arlo receives a message, it embeds the query and retrieves the most semantically similar stored facts — so "what should I eat?" surfaces "user is vegetarian" and "user loves spicy food" without scanning every stored memory.

## Repo Structure

```
arlo/
├── core/
│   ├── llm.py                   # LLM client abstraction (OpenAI or OpenRouter)
│   ├── memory/
│   │   ├── extractor.py         # Passive profile extraction from messages
│   │   ├── store.py             # Read/write/update via mem0 + Supabase
│   │   └── models.py            # UserProfile, MemoryEntry schemas
│   ├── agent/
│   │   ├── orchestrator.py      # LangGraph ReAct loop
│   │   ├── planner.py           # Task decomposition
│   │   ├── classifier.py        # Tone + intent classification (single call)
│   │   └── persona.py           # System prompt builder with memory injection
│   ├── tools/
│   │   ├── search.py            # Tavily web search wrapper
│   │   └── reader.py            # Jina Reader URL fetcher
│   ├── scheduler/
│   │   └── digest.py            # Daily proactive suggestion engine (APScheduler)
│   ├── interfaces/
│   │   └── discord/
│   │       ├── bot.py           # discord.py bot setup
│   │       ├── handlers.py      # Message event handlers
│   │       └── commands.py      # /start, /profile, /forget
│   └── api.py                   # FastAPI app (health check + future hooks)
├── tests/
├── docs/
│   └── prd.md
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Key Conventions

- **No mutations** — always return new objects; don't modify in place
- **Classifier is one call** — tone + intent resolved in a single LLM call, not two
- **Memory extraction is background** — runs after every `PROFILE_EXTRACTION_INTERVAL` messages, never blocking the response path
- **ReAct loop has a hard ceiling** — `MAX_REACT_ITERATIONS=8`, enforced in `orchestrator.py`
- **Persona is a prompt problem** — `persona.py` builds the system prompt; no special runtime logic
- **Validate URLs before fetching** — never pass LLM-generated URLs directly to `read_url()` without validation
- **No Firecrawl in MVP** — Jina Reader only; add Firecrawl post-MVP if needed

## Environment Variables

```bash
# LLM — pick one; set LLM_PROVIDER accordingly
OPENAI_API_KEY=
OPENROUTER_API_KEY=
LLM_PROVIDER=openai              # openai | openrouter

# Database (PostgreSQL + pgvector — run via docker-compose)
# Format: postgresql://<user>:<password>@<host>:<port>/<dbname>
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/arlo

# Search
TAVILY_API_KEY=

# Discord
DISCORD_BOT_TOKEN=
DISCORD_GUILD_ID=

# App config
ENVIRONMENT=development
LOG_LEVEL=info
PROFILE_EXTRACTION_INTERVAL=10   # extract profile every N messages
MAX_REACT_ITERATIONS=8
```

> No `MEM0_API_KEY` needed — mem0 runs in self-hosted mode against `DATABASE_URL`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=term-missing

# Run a specific module
pytest tests/test_classifier.py
```

Minimum coverage: **80%**. Unit tests for tools, memory, and classifier. Integration tests for the ReAct loop end-to-end (use recorded Tavily fixtures to avoid live API calls in CI).

## Build Sequence

| Week | Milestone |
|---|---|
| 1–2 | Discord bot + basic LLM conversation end-to-end |
| 3–4 | Memory: passive extraction + PostgreSQL + pgvector storage |
| 5–6 | Daily proactive digest via APScheduler |
| 7–8 | ReAct task loop (Tavily + Jina + LangGraph) |
| 9–10 | Persona tuning, onboarding `/start`, profile commands, guardrails |
| 11–12 | Polish, testing, public GitHub launch |

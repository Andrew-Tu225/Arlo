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
handlers.py — Filter gate (drop silently if any match):
  - message.author == bot itself
  - message.guild.id != DISCORD_GUILD_ID
  - message.author.id != DISCORD_USER_ID   ← single-user enforcement
  - message.content is empty
    ↓
INSERT into episodic_messages (async, non-blocking)
Build context window: last CONTEXT_WINDOW_SIZE rows from episodic_messages (default: 12)
    ↓
Unified LangGraph Agent (orchestrator.py)
  System prompt (persona.py):
    - Persona rules (anti-bot tone, has opinions, matches energy)
    - Content guardrails (refuses harmful requests, won't impersonate, stays honest)
    - Basic profile summary (location, key interests injected from mem0)
    - Tone guidance
  Tools:
    - web_search(query)   → Tavily API
    - read_url(url)       → URL validation → Jina Reader (r.jina.ai/{url})
    - search_memory(q)    → mem0 semantic search (on-demand user context)
    - remember(fact)      → mem0.add() with contradiction handling
    ↓
  The model decides (no separate classifier call):
    Casual message       → respond directly, no tool calls
    Task request         → calls web_search / read_url (ReAct loop)
    Memory-worthy stmt   → calls remember()
    Needs user context   → calls search_memory() mid-reasoning
    ↓
  ReAct loop ceiling: MAX_REACT_ITERATIONS (default: 8)
    → On ceiling hit or TASK_TOKEN_BUDGET exceeded:
       honest "couldn't find a reliable source" reply, no hallucination
    ↓
Discord: send reply (plain text)
    ↓
[after reply — non-blocking]
  message_count % PROFILE_EXTRACTION_INTERVAL == 0?
    → asyncio.create_task(extractor.py):
        read last N rows from episodic_messages
        ONE LLM call: extract facts (interests, preferences, opinions, habits)
        mem0.add() for each fact with short-term/long-term tag
```

### Subsystem B — Proactive (APScheduler-triggered)

The Discord bot is a long-running process. APScheduler runs jobs inside that process and can send Discord messages at any time without a user prompt. This is how all proactive outreach works — no MCP or webhooks needed for MVP.

```
APScheduler daily job fires at DIGEST_TIME in DIGEST_TIMEZONE (default: 09:00 America/Toronto)
    ↓
digest.py:
  1. mem0.search("interests preferences habits") → user profile snapshot
  2. Build Tavily queries from profile interests
     e.g. "AI news today", "NBA highlights", "r/ProgrammerHumor top posts"
  3. Tavily search → news, Reddit posts, trending topics in user's niches
  4. LLM composes casual digest — 2–4 items, references ≥1 profile fact
  5. discord.get_channel(channel_id).send(message)

Job config (schedule, channel, on/off state) persisted in PostgreSQL.
On bot restart: re-read config from DB, re-register with misfire_grace_time=3600.
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

No separate planner — the reason node handles step-by-step decomposition inline.

```
orchestrator.py: LangGraph graph
  ┌─ Reason node: LLM decides next tool call (or whether to synthesize)
  │
  ├─ web_search(query) → Tavily API → list of {url, title, snippet}
  │
  ├─ read_url(url)     → SSRF validation → Jina Reader (r.jina.ai/{url}) → page text
  │                      (reject: non-http(s), RFC 1918, loopback, link-local, malformed)
  │
  ├─ search_memory(q)  → mem0 semantic search → relevant user facts
  │
  └─ synthesize()      → LLM compiles all observations into final answer + source URLs
    ↓
Exit conditions (first one wins):
  - synthesize reached
  - MAX_REACT_ITERATIONS=8 hit → honest "couldn't find a reliable source"
  - TASK_TOKEN_BUDGET exceeded → same honest fallback
```

### Memory Layers

| Layer | What's stored | Storage | How retrieved |
|---|---|---|---|
| Short-term context | Last CONTEXT_WINDOW_SIZE messages | PostgreSQL `episodic_messages` | SELECT last N rows, passed directly in prompt |
| Long-term semantic | Facts, preferences, traits (vector) | PostgreSQL + pgvector | `search_memory` tool — semantic similarity search |
| Long-term structured | Same facts, structured form | PostgreSQL (via mem0) | mem0 query by dimension |
| Episodic log | Raw interaction history | PostgreSQL `episodic_messages` | Read by extraction job every N messages |

**How embeddings work:** mem0 converts each stored fact into a vector (array of numbers) representing its meaning. Similar texts get nearby vectors. When Arlo receives a message, it embeds the query and retrieves the most semantically similar stored facts — so "what should I eat?" surfaces "user is vegetarian" and "user loves spicy food" without scanning every stored memory.

## Repo Structure

```
arlo/
├── core/
│   ├── llm.py                   # LLM client abstraction (OpenAI or OpenRouter)
│   ├── memory/
│   │   ├── extractor.py         # Passive profile extraction from episodic_messages
│   │   ├── store.py             # Read/write/update via mem0 (self-hosted)
│   │   └── models.py            # UserProfile, MemoryEntry schemas
│   ├── agent/
│   │   ├── orchestrator.py      # Unified LangGraph agent (4 tools: web_search, read_url, search_memory, remember)
│   │   └── persona.py           # System prompt builder (persona rules + guardrails + memory injection)
│   ├── tools/
│   │   ├── search.py            # web_search tool: Tavily wrapper
│   │   └── reader.py            # read_url tool: SSRF validation + Jina Reader
│   ├── scheduler/
│   │   └── digest.py            # Daily proactive digest (APScheduler, DIGEST_TIME + DIGEST_TIMEZONE)
│   ├── interfaces/
│   │   └── discord/
│   │       ├── bot.py           # discord.py bot setup + startup validation
│   │       ├── handlers.py      # on_message: DISCORD_USER_ID filter, episodic INSERT, agent dispatch
│   │       └── commands.py      # /start, /profile, /forget, /digest
│   └── api.py                   # FastAPI app (health check + future hooks)
├── tests/
├── docs/
│   ├── prd.md
│   └── architecture.md
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**Not in codebase (removed post eng-review):**
- `core/agent/classifier.py` — Removed. The unified agent routes implicitly via tool-calling behavior; no separate classifier call.
- `core/agent/planner.py` — Removed. The ReAct reason node handles step-by-step planning inline; separate planner = redundant LLM call.

## Key Conventions

- **No mutations** — always return new objects; don't modify in place
- **No classifier call** — the unified LangGraph agent routes implicitly via which tools the model calls; no separate tone/intent classification
- **No planner call** — the ReAct reason node handles decomposition inline; no separate planner LLM call
- **Memory extraction is background** — `asyncio.create_task` after every `PROFILE_EXTRACTION_INTERVAL` messages, never blocking the response path
- **ReAct loop has a hard ceiling** — `MAX_REACT_ITERATIONS=8` and `TASK_TOKEN_BUDGET=8000`, both enforced in `orchestrator.py`
- **Persona is a prompt problem** — `persona.py` builds the system prompt including guardrails; no special runtime logic
- **Validate URLs before fetching** — `ipaddress` + `validators` libraries in `reader.py`; never pass LLM-generated URLs directly to Jina Reader
- **Single-user enforcement in handlers** — `DISCORD_USER_ID` checked on every message in `handlers.py`; missing at startup = halt
- **Episodic log is the source of truth** — context window reads from `episodic_messages` table, not Discord API
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
DISCORD_USER_ID=                 # Discord user ID — only this user's messages are processed

# Digest scheduler
DIGEST_TIME=09:00                # HH:MM — time of day for the daily digest
DIGEST_TIMEZONE=America/Toronto  # IANA timezone string; validated at startup

# App config
ENVIRONMENT=development
LOG_LEVEL=info
CONTEXT_WINDOW_SIZE=12           # messages passed as context to the agent
PROFILE_EXTRACTION_INTERVAL=10   # extract profile facts every N messages
MAX_REACT_ITERATIONS=8           # hard ceiling on ReAct tool-use loop
TASK_TOKEN_BUDGET=8000           # max tokens per task; honest fallback on exceed
```

Required at startup (missing = halt with clear error): `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`, `DISCORD_USER_ID`, `DATABASE_URL`, at least one LLM key, `TAVILY_API_KEY`. `DIGEST_TIMEZONE` must be a valid IANA string.

> No `MEM0_API_KEY` needed — mem0 runs in self-hosted mode against `DATABASE_URL`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=term-missing

# Run a specific module
pytest tests/test_orchestrator.py
```

Minimum coverage: **80%**. Unit tests for tools (SSRF validation, Tavily wrapper), memory (extraction, store), and the agent (ReAct loop, persona builder). Integration tests for the full ReAct loop end-to-end — use recorded Tavily fixtures to avoid live API calls in CI.

## Build Sequence

| Week | Milestone |
|---|---|
| 1–2 | Discord bot running, unified LangGraph agent, basic LLM conversation end-to-end, content guardrails in system prompt, startup validation |
| 3–4 | Memory layer: `episodic_messages` table, passive extraction, PostgreSQL + pgvector storage, `/profile` + `/forget` commands |
| 5–6 | Daily proactive digest via APScheduler (`DIGEST_TIME` + `DIGEST_TIMEZONE`) |
| 7–8 | ReAct tools expansion: Tavily search, Jina Reader, `search_memory` tool, URL validation (SSRF) |
| 9–10 | Persona tuning, `/start` onboarding |
| 11–12 | Polish, test coverage ≥ 80%, public GitHub launch |

# Arlo — Architecture Reference

**Status:** Canonical  
**Updated:** 2026-05-22 (post eng-review)

This is the contributor-facing technical reference for Arlo's architecture. For product requirements see `prd.md`. For LLM-prompt-style instructions see `CLAUDE.md`.

---

## System Overview

Arlo is a single long-running Python process with two parallel subsystems: **Reactive** (user-triggered) and **Proactive** (scheduler-triggered). Both subsystems share the same database, LLM client, and memory store.

```
┌─────────────────────────────────────────────────────────┐
│                    Arlo Process                         │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │  Subsystem A         │  │  Subsystem B             │ │
│  │  Reactive            │  │  Proactive               │ │
│  │  (discord.py events) │  │  (APScheduler jobs)      │ │
│  └──────────┬───────────┘  └──────────────┬───────────┘ │
│             │                             │             │
│  ┌──────────▼─────────────────────────────▼───────────┐ │
│  │           Shared Infrastructure                    │ │
│  │  core/llm.py  |  core/memory/  |  PostgreSQL       │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Subsystem A — Reactive (user-triggered)

Every Discord message from the authorized user travels through this pipeline:

```
User sends Discord message
    │
    ▼
bot.py: on_message fires
    │
    ▼
handlers.py — Filter gate (drop silently if any condition matches):
    ├── message.author == bot itself
    ├── message.guild.id != DISCORD_GUILD_ID
    ├── message.author.id != DISCORD_USER_ID   ← single-user enforcement
    └── message.content is empty
    │
    ▼
INSERT into episodic_messages (async, non-blocking)
    • Columns: id, user_id, content, role, created_at
    • 30-day retention; pruned weekly
    │
    ▼
Build context window:
    • SELECT last CONTEXT_WINDOW_SIZE rows FROM episodic_messages ORDER BY created_at DESC
    • Default: last 12 messages
    │
    ▼
Unified LangGraph Agent (orchestrator.py)
    │
    ├── System prompt (built by persona.py):
    │       - Anti-bot persona rules (no "Sure! Happy to help!", no bullet points in casual chat)
    │       - Content guardrails (refuses harmful requests, won't impersonate, stays honest)
    │       - Basic profile summary (location, top interests injected from mem0)
    │       - Tone guidance (match user energy)
    │
    ├── Tool registry:
    │       web_search(query)   → Tavily API
    │       read_url(url)       → URL validation → Jina Reader
    │       search_memory(q)    → mem0 semantic search (on-demand)
    │       remember(fact)      → mem0.add() with contradiction handling
    │
    ├── Model routing (implicit — no classifier call):
    │       Casual message       → respond directly, no tool calls
    │       Task request         → web_search + read_url (ReAct loop)
    │       Memory-worthy stmt   → calls remember()
    │       Needs user context   → calls search_memory() mid-reasoning
    │
    └── ReAct loop ceiling: MAX_REACT_ITERATIONS (default: 8)
            → On ceiling hit or TASK_TOKEN_BUDGET exceeded:
               honest fallback ("couldn't find a reliable source"), no hallucination
    │
    ▼
Discord: send reply (plain text only, no rich embeds in MVP)
    │
    ▼
After reply — non-blocking background task:
    message_count % PROFILE_EXTRACTION_INTERVAL == 0?
        → asyncio.create_task(extractor.py):
              SELECT last N rows FROM episodic_messages
              ONE LLM call: extract facts (interests, preferences, opinions, habits)
              mem0.add() for each fact with short-term/long-term tag
```

---

## Subsystem B — Proactive (APScheduler-triggered)

APScheduler runs jobs inside the Arlo process. No external cron, queue, or webhook needed.

```
APScheduler daily job fires at DIGEST_TIME in DIGEST_TIMEZONE
    │
    ▼
digest.py:
    1. mem0.search("interests preferences habits") → user profile snapshot
    2. Build Tavily queries from profile interests
           e.g. "AI news today", "NBA highlights", "r/ProgrammerHumor top posts"
    3. Tavily search → news, Reddit posts, trending topics
    4. LLM composes casual digest:
           - 2–4 items max
           - Conversational framing (not a newsletter)
           - References ≥1 profile fact to feel personalized
    5. discord.get_channel(channel_id).send(message)
    │
    ▼
Job persistence:
    • Schedule, channel_id, and on/off state persisted in PostgreSQL
    • On bot restart: re-read config from DB, re-register job
    • misfire_grace_time=3600: a restart within the hour doesn't re-fire the day's digest
    • /digest off → pause job; /digest on → resume
```

---

## ReAct Loop Detail

The ReAct loop lives entirely inside `orchestrator.py`. There is no separate planner — the LangGraph reason node handles step-by-step decomposition inline.

```
Reason node (LLM):
    Decides which tool to call next, or whether to synthesize
    │
    ├── web_search(query)
    │       → Tavily API
    │       → returns list of {url, title, snippet}
    │
    ├── read_url(url)
    │       → URL validation (see SSRF Prevention)
    │       → Jina Reader: GET r.jina.ai/{url}
    │       → returns page text
    │
    ├── search_memory(q)
    │       → mem0.search(q, user_id=USER_ID)
    │       → returns list of relevant facts
    │
    └── remember(fact)
            → mem0.add(fact, user_id=USER_ID)
            → handles contradiction: new fact overwrites old on same dimension
    │
    ▼
Synthesize node (LLM):
    Compiles all observations into final answer + inline source URLs
    │
    ▼
Exit conditions (first one wins):
    ├── synthesize node reached                     → send answer
    ├── MAX_REACT_ITERATIONS hit (default: 8)       → honest fallback
    └── TASK_TOKEN_BUDGET exceeded (default: 8000)  → honest fallback
```

---

## SSRF Prevention

All URLs pass through validation before reaching Jina Reader. This is enforced in `core/tools/reader.py`.

Rejected URLs:
- Non-http/https schemes (`ftp://`, `file://`, `data:`, etc.)
- Private IP ranges (RFC 1918): `10.x.x.x`, `172.16–31.x.x`, `192.168.x.x`
- Loopback: `127.x.x.x`, `::1`
- Link-local: `169.254.x.x`, `fe80::/10`
- Multicast and reserved ranges
- Malformed / unresolvable hostnames
- Encoded IP variants (hex, octal, URL-encoded)

Libraries used: Python `ipaddress` (stdlib) + `validators` (PyPI).

---

## Memory Architecture

| Layer | What's stored | Storage | How retrieved |
|---|---|---|---|
| Short-term context | Last `CONTEXT_WINDOW_SIZE` messages | PostgreSQL `episodic_messages` | SELECT last N rows, passed directly in prompt |
| Long-term semantic | Facts, preferences, traits | PostgreSQL + pgvector | `search_memory` tool — semantic similarity search |
| Long-term structured | Same facts, structured form | PostgreSQL (via mem0) | mem0 query by dimension |
| Episodic log | Raw interaction history | PostgreSQL `episodic_messages` | Read by extraction job every N messages |

### Episodic Messages Table

```sql
CREATE TABLE episodic_messages (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON episodic_messages (user_id, created_at DESC);
```

Retention: rows older than 30 days are pruned on a weekly cron job. Extracted facts live indefinitely in mem0.

### How Embeddings Work

mem0 converts each stored fact into a vector (dense array of floats) representing its meaning. Semantically similar texts produce nearby vectors. When Arlo receives a message, it embeds the query string and retrieves the most similar stored facts via pgvector's `<=>` operator — so "what should I eat?" surfaces "user is vegetarian" and "user loves spicy food" without scanning every stored memory.

---

## Memory Extraction (Passive Profile Building)

After every `PROFILE_EXTRACTION_INTERVAL` messages (default: 10), a background task runs:

```
asyncio.create_task(extractor.run())
    │
    ▼
extractor.py:
    1. SELECT last N rows FROM episodic_messages WHERE user_id = USER_ID
    2. ONE LLM call with extraction prompt:
           "Extract user facts from this conversation.
            For each fact note: dimension, value, is_short_term (bool)"
    3. For each extracted fact:
           mem0.add(fact, metadata={"short_term": bool})
           Contradiction handling: new value overwrites old on same dimension
```

**Short-term vs long-term tagging:**
- Short-term: time-bound context ("in Tokyo this week", "job interview tomorrow")
- Long-term: stable traits ("hates layovers", "vegetarian", "obsessed with mechanical keyboards")

mem0 uses both when building similarity search results; short-term facts decay in relevance over time.

---

## Slash Commands

| Command | Handler | Description |
|---|---|---|
| `/start` | `commands.py` | Onboarding: 5–7 questions, answers written directly to mem0 |
| `/profile` | `commands.py` | mem0.search("*") → formatted readable summary |
| `/forget [topic]` | `commands.py` | mem0.delete() for facts matching topic, then ack |
| `/digest on\|off` | `commands.py` | Toggle APScheduler digest job, persist state to DB |

---

## Startup Validation

The bot validates all required environment variables at startup and halts with a clear error message if any are missing or invalid. This prevents silent failures from misconfiguration.

Required at startup:
- `DISCORD_BOT_TOKEN` — missing = can't connect to Discord
- `DISCORD_GUILD_ID` — missing = can't filter to correct server
- `DISCORD_USER_ID` — missing = silently drops all messages (treated as unset = no user)
- `DATABASE_URL` — missing = no episodic log, no mem0 store
- At least one of `OPENAI_API_KEY` / `OPENROUTER_API_KEY` — missing = no LLM calls
- `TAVILY_API_KEY` — missing = ReAct task loop has no search tool
- `DIGEST_TIMEZONE` — validate against `pytz.all_timezones` at startup; invalid = scheduler fires at wrong time silently

---

## Module Map

```
core/
├── llm.py                   # LLM client abstraction (OpenAI or OpenRouter)
│                              Switch provider via LLM_PROVIDER env var
│
├── memory/
│   ├── store.py             # mem0 interface: add, search, delete, get_all
│   ├── extractor.py         # Passive profile extraction from episodic_messages
│   └── models.py            # UserProfile, MemoryEntry dataclasses
│
├── agent/
│   ├── orchestrator.py      # LangGraph ReAct loop (unified agent, 4 tools)
│   └── persona.py           # System prompt builder (persona rules + memory injection)
│
├── tools/
│   ├── search.py            # web_search tool: Tavily wrapper
│   └── reader.py            # read_url tool: SSRF validation + Jina Reader
│
├── scheduler/
│   └── digest.py            # Daily proactive digest (APScheduler job)
│
├── interfaces/
│   └── discord/
│       ├── bot.py           # discord.py client setup, on_ready, startup validation
│       ├── handlers.py      # on_message: filter gate, episodic INSERT, agent dispatch
│       └── commands.py      # /start, /profile, /forget, /digest
│
└── api.py                   # FastAPI app — health check endpoint only (MVP)
```

**Removed from original design (post eng-review):**
- `core/agent/classifier.py` — Removed. Separate tone+intent classification call adds latency/cost without benefit. The unified LangGraph agent routes implicitly via tool-calling behavior.
- `core/agent/planner.py` — Removed. The ReAct reason node handles step-by-step decomposition inline. No separate planner LLM call needed.

---

## Environment Variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | One of these | — | |
| `OPENROUTER_API_KEY` | One of these | — | |
| `LLM_PROVIDER` | No | `openai` | `openai` or `openrouter` |
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `TAVILY_API_KEY` | Yes | — | Web search |
| `DISCORD_BOT_TOKEN` | Yes | — | |
| `DISCORD_GUILD_ID` | Yes | — | Server ID for message filtering |
| `DISCORD_USER_ID` | Yes | — | Only messages from this user are processed |
| `DIGEST_TIME` | No | `09:00` | HH:MM format |
| `DIGEST_TIMEZONE` | No | `America/Toronto` | IANA timezone string; validated at startup |
| `ENVIRONMENT` | No | `development` | |
| `LOG_LEVEL` | No | `info` | |
| `CONTEXT_WINDOW_SIZE` | No | `12` | Messages passed as context to the agent |
| `PROFILE_EXTRACTION_INTERVAL` | No | `10` | Run extraction every N messages |
| `MAX_REACT_ITERATIONS` | No | `8` | Hard ceiling on ReAct tool-use loop |
| `TASK_TOKEN_BUDGET` | No | `8000` | Max tokens per task request; honest fallback on exceed |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent orchestration | LangGraph | Tools will expand significantly; structured graph handles complex multi-tool flows cleanly |
| Routing | Implicit (tool-calling behavior) | No classifier call — saves latency + cost; model routes via which tools it calls |
| Planner | None (inline reason node) | ReAct reason node already plans step-by-step; separate planner = redundant LLM call |
| Episodic log | PostgreSQL table | Discord API fetch is fragile (rate limits, deleted messages, permission changes); ~10-50 MB/year at 30-day retention |
| User context in agent | Lightweight summary + `search_memory` tool | Full injection grows with memory; on-demand retrieval scales indefinitely |
| URL validation | `ipaddress` + `validators` | Covers RFC 1918, loopback, link-local, IPv6, encoded variants — stdlib + one well-tested library |
| Digest scheduling | `DIGEST_TIME` + `DIGEST_TIMEZONE` env vars | "9am local" was ambiguous; explicit IANA timezone with startup validation |
| Single-user enforcement | `DISCORD_USER_ID` filter in handlers.py | PRD said "one user" but no mechanism existed; env var + startup validation makes it explicit |
| Guardrails timing | Week 1-2 (with persona) | Must exist before memory extraction starts, so bad data doesn't accumulate |
| Profile commands timing | Week 3-4 (with memory) | `/profile` + `/forget` needed immediately when extraction goes live to catch bad facts early |

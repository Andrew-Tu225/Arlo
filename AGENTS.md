# Arlo — Agent Context (Cursor)

Arlo is a persistent AI companion Discord bot. It learns the user over time, proactively surfaces relevant content, and executes real-world tasks via a ReAct web-search loop — designed to feel like texting a smart friend, not a chatbot.

**Open source constraint:** Self-hostable only. No hard dependency on managed services (mem0 cloud, Supabase, etc.). Users bring API keys (LLM, Tavily) and run PostgreSQL via Docker Compose.

For full product/architecture detail see `docs/prd.md` and `docs/architecture.md`. Claude Code uses `CLAUDE.md` (same baseline; this file adds **implementation status** from `.claude/session`).

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
docker compose up -d          # PostgreSQL + pgvector
python -m core.interfaces.discord.bot
uvicorn core.api:app --reload # optional health server
```

---

## Implementation Status (2026-06)

| Phase | Weeks | Status | Notes |
|-------|-------|--------|-------|
| 1 | 1–2 | **Done** | Discord bot, persona, single-node LangGraph chat, startup validation, `/health` |
| 2 | 3–4 | **Done** | `episodic_messages`, mem0 store, passive extraction, `/profile`, `/forget` |
| 3 | 5–6 | **Done in code** | `schedules` + `arlo_channels` tables, APScheduler digest, Tavily `web_search`, DM handler, `seed_default_schedules` |
| 4 | 7–8 | **Next** | Full ReAct in `orchestrator.py`, `read_url` (Jina + SSRF), schedule write tools, safe-action framework |
| 5+ | — | Planned | Calendar/docs assistants, media/commerce MCPs — see `.claude/session/Arlo-implementation-plan.md` |

### What works today

- Reactive Discord messages (guild + DMs from `DISCORD_USER_ID`)
- Episodic log + mem0 semantic memory + background extraction every `PROFILE_EXTRACTION_INTERVAL` messages
- Proactive morning DM via `schedules` row (`morning-proactive`, `discord_channel_id = NULL`)
- Schedule agent uses Tavily + mem0 profile (single LLM call in `run_schedule_agent`, not full ReAct yet)

### Not built yet (do not assume in code)

- ReAct tool loop in `orchestrator.py` (still Phase 1 `chat` node only)
- `read_url` implementation (`core/tools/reader.py` is docstring-only stub)
- `/digest on|off` slash command — **removed**; schedules are DB rows, conversational CRUD in Phase 4
- Safe-action approval UI for medium-risk tools
- `classifier.py` / `planner.py` — **removed by design**; routing is implicit via tool calls

---

## Architecture (single process)

```
┌─────────────────────────────────────────────────────────┐
│                    Arlo Process                         │
│  Subsystem A: Reactive (discord on_message)             │
│  Subsystem B: Proactive (APScheduler → digest.py)       │
│  Shared: core/llm.py, core/memory/, PostgreSQL          │
└─────────────────────────────────────────────────────────┘
```

### Reactive path

```
Discord message → handlers.py filter gate → INSERT episodic_messages
→ context window (last CONTEXT_WINDOW_SIZE from DB, not Discord API)
→ orchestrator.py (persona + optional mem0 search in prompt)
→ Discord reply → background extractor every N messages
```

**Filter gate** (drop silently): bot self, wrong guild, wrong user (`DISCORD_USER_ID`), empty content.

### Proactive path

```
Cron from schedules table → run_schedule_job → run_schedule_agent
→ Tavily + LLM compose → DM or channel send (NULL channel_id = DM)
```

On startup: `seed_default_schedules` → `scheduler.start()` → `register_digest_jobs`.

### Unified agent (target Phase 4)

One LangGraph graph, four tools, no separate classifier/planner:

| Tool | Purpose |
|------|---------|
| `web_search` | Tavily |
| `read_url` | SSRF check → Jina `r.jina.ai/{url}` |
| `search_memory` | mem0 semantic search |
| `remember` | mem0.add with contradiction handling |

Exit: synthesize, or `MAX_REACT_ITERATIONS` (8), or `TASK_TOKEN_BUDGET` (8000) → honest fallback, no hallucination.

---

## Repo Map

```
core/
├── llm.py                      # OpenAI or OpenRouter via LLM_PROVIDER
├── settings.py                 # pydantic-settings; fail fast at startup
├── db.py                       # asyncpg, episodic_messages, schedules, arlo_channels
├── memory/                     # store (mem0), extractor, models
├── agent/
│   ├── orchestrator.py         # LangGraph — Phase 1 chat node today
│   └── persona.py              # system prompt + guardrails + memory injection
├── tools/
│   ├── search.py               # Tavily (done)
│   └── reader.py               # SSRF + Jina (stub)
├── scheduler/digest.py         # APScheduler jobs
├── interfaces/discord/         # bot, handlers, commands
└── api.py                      # FastAPI /health
```

---

## Coding Conventions (required)

- **No in-place mutation** — return new objects
- **No classifier / planner LLM calls** — routing via which tools the model invokes
- **Memory extraction is async background** — `asyncio.create_task`; never block the reply path
- **Episodic log is source of truth** for context window — not Discord history API
- **Validate URLs before fetch** — `ipaddress` + `validators` in `reader.py`
- **Single-user MVP** — `DISCORD_USER_ID` in `handlers.py`; missing at startup = halt
- **Commit messages: one line** — no file lists or bullets in commit body; use PR for detail
- **Tests:** `pytest`, target **≥80%** on `core`; mock Tavily/mem0/LLM in unit tests

---

## Environment Variables

Required at startup (missing = halt with clear error):

`DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`, `DISCORD_USER_ID`, `DATABASE_URL`, at least one of `OPENAI_API_KEY` / `OPENROUTER_API_KEY`, `TAVILY_API_KEY`. `DIGEST_TIMEZONE` must be valid IANA.

| Variable | Default | Role |
|----------|---------|------|
| `LLM_PROVIDER` | `openai` | `openai` \| `openrouter` |
| `CONTEXT_WINDOW_SIZE` | `12` | Messages in agent prompt |
| `PROFILE_EXTRACTION_INTERVAL` | `10` | Extract facts every N messages |
| `MAX_REACT_ITERATIONS` | `8` | ReAct ceiling (Phase 4) |
| `TASK_TOKEN_BUDGET` | `8000` | Task token cap (Phase 4) |
| `DIGEST_TIME` | `09:00` | Morning schedule cron |
| `DIGEST_TIMEZONE` | `America/Toronto` | IANA timezone |

No `MEM0_API_KEY` — mem0 runs self-hosted against `DATABASE_URL`.

---

## Slash Commands (MVP)

| Command | Status |
|---------|--------|
| `/profile` | Done |
| `/forget [topic]` | Done |
| `/start` | Phase 9–10 onboarding |
| `/digest on\|off` | **Not in roadmap** — use `schedules` + Phase 4 conversational tools |

---

## Session Plans (builder context)

Living plans under `.claude/session/` (gitignored `.claude/` but files exist locally):

| File | Use when |
|------|----------|
| `Arlo-implementation-plan.md` | Master phase roadmap + feature inventory |
| `phase1-plan.md` | Discord foundation checklist |
| `phase2-plan.md` | Memory layer checklist |
| `phase3-plan.md` | Schedules + proactive DM detail |
| `agent-feature-plan.md` | Post-MVP MCP groups (productivity, media, commerce) |
| `project-description-plan.md` | Product pillars and positioning |

When implementing Phase 4, read `Arlo-implementation-plan.md` § Phase 4 first — two ReAct upgrades: user `orchestrator` (6 tools + safe actions) vs proactive `run_schedule_agent` (read-only: `web_search` + `search_memory`).

---

## Agent Workflow Tips

1. Read the phase section in `.claude/session/Arlo-implementation-plan.md` before large features.
2. Match existing patterns in the target module (`handlers.py`, `digest.py`, `store.py`) before adding abstractions.
3. Prefer minimal diffs; do not reintroduce `classifier.py` or `planner.py`.
4. Wire new DB tables in `core/db.py` `init_tables()` with tests in `tests/test_db.py`.
5. Do not commit `.env` or secrets.

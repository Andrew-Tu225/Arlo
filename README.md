# Arlo

> A persistent AI companion that reaches out, remembers you, and gets things done.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What is Arlo?

Arlo is an AI companion that lives in Discord and texts you like a smart friend — not a chatbot you have to prompt.

Most AI tools wait to be asked. Arlo doesn't. It sends you things it knows you'll care about on its own schedule, remembers what matters to you across every conversation, and executes real tasks with web research when you need an answer, not a list of links.

**What it does:**

- **Reaches out proactively.** Scheduled jobs fire throughout the day — morning digest, evening wrap-up, reminders you set — composed fresh each time using your memory and live web search.
- **Remembers you.** Interests, opinions, habits, life context — extracted passively from what you say and recalled naturally, even months later.
- **Researches and synthesizes.** Describe what you need. Arlo searches, reads the relevant pages, and sends back a real answer with sources — not links.
- **Manages your schedules conversationally.** Create, edit, or delete proactive reminders in plain language. Changes require your Discord approval before they go through.

---

## Architecture

Arlo runs two subsystems in parallel inside a single process:

### Reactive path — user-triggered

```
Discord message
  └─ handlers.py: filter gate (guild/user ID check) → INSERT episodic_messages
       └─ orchestrator.py: LangGraph ReAct + MemorySaver
            ├─ search_memory / remember          (inline — mem0)
            ├─ list_schedules                    (inline — DB read)
            ├─ research(task)          ──────►  researcher.py  (ephemeral ReAct)
            │                                       tools: web_search, read_url
            │                                       returns: ResearchBrief JSON
            ├─ plan_schedule_change(request) ──► schedule_planner.py  (ephemeral ReAct)
            │                                       tools: list_schedules, search_memory
            │                                       returns: SchedulePlan JSON
            └─ create/edit/delete_schedule      (HITL — Discord approval required)
                 └─ actions.py: interrupt() → Discord View buttons → resume()
  └─ Discord reply
       └─ background: extractor.maybe_extract() every N messages
```

### Proactive path — scheduler-triggered

```
APScheduler cron fires
  └─ digest.py: run_schedule_job(schedule_row)
       └─ proactive.py: ephemeral ReAct agent
            ├─ search_memory      → user context for personalization
            ├─ run_research        → fresh content via researcher sub-agent
            └─ get_recent_sends   → last 7 messages for anti-repetition
  └─ discord.send(composed message)
       └─ INSERT schedule_run_log
```

### Multi-agent design

The orchestrator never runs web searches directly. It delegates to isolated sub-agents that run their own ReAct loop and return a single compressed result — raw Tavily payloads and intermediate reasoning stay out of the orchestrator's context window.

| Component | Checkpointer | Role |
|-----------|-------------|------|
| Orchestrator | MemorySaver (persistent per user) | Routes messages; manages memory and schedules |
| Research sub-agent | None (ephemeral) | Web search + URL reading; returns `ResearchBrief` JSON |
| Schedule planner sub-agent | None (ephemeral) | NL → `SchedulePlan` JSON or clarifying question |
| Proactive agent | None (ephemeral) | Composes scheduled Discord messages |

### Memory layers

| Layer | What's stored | Backend | How retrieved |
|-------|--------------|---------|---------------|
| Short-term context | Last `CONTEXT_WINDOW_SIZE` messages | PostgreSQL `episodic_messages` | SELECT last N rows, injected into prompt |
| Long-term semantic | Facts, preferences, traits | PostgreSQL + pgvector (via mem0) | `search_memory` tool — semantic similarity |
| Schedule history | Recent sends per schedule | PostgreSQL `schedule_run_log` | `get_recent_sends` tool — anti-repetition |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI (GPT-4o) or OpenRouter — swap via `LLM_PROVIDER` |
| Agent orchestration | LangGraph — shared `build_react_graph` factory; MemorySaver for orchestrator |
| Memory | mem0 (self-hosted) + PostgreSQL + pgvector |
| Web search | Tavily API |
| URL reading | Jina Reader (`r.jina.ai/{url}`) with SSRF validation |
| Discord | discord.py — single-server, single-user |
| Scheduler | APScheduler (in-process, cron rows from DB) |
| Backend | FastAPI (health check + future hooks) |
| Deployment | Docker Compose |

---

## Local Setup

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- API keys: an LLM provider (OpenAI or OpenRouter), Tavily, and a Discord bot token

### 1. Clone and install

```bash
git clone https://github.com/Andrew-Tu225/Arlo.git
cd Arlo

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```bash
OPENAI_API_KEY=sk-...            # or OPENROUTER_API_KEY + LLM_PROVIDER=openrouter
TAVILY_API_KEY=tvly-...
DISCORD_BOT_TOKEN=...
DISCORD_GUILD_ID=...             # right-click your server → Copy Server ID
DISCORD_USER_ID=...              # right-click your profile → Copy User ID
```

### 3. Start the database

```bash
docker compose up -d
```

This starts a PostgreSQL 16 container on port 5432. The bot creates all tables on first startup.

### 4. Run the bot

```bash
python -m core.interfaces.discord.bot
```

For the optional FastAPI health server alongside:

```bash
uvicorn core.api:app --reload
```

---

## Environment Variables

### Required at startup (missing = hard stop)

| Variable | Description |
|----------|-------------|
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_GUILD_ID` | Server ID — messages from other servers are dropped |
| `DISCORD_USER_ID` | Your Discord user ID — only this user's messages are processed |
| `DATABASE_URL` | PostgreSQL connection string |
| `TAVILY_API_KEY` | Tavily web search key |
| `OPENAI_API_KEY` or `OPENROUTER_API_KEY` | At least one LLM key required |

### Optional (defaults shown)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `openrouter` |
| `DIGEST_TIME` | `09:00` | Morning digest time (HH:MM) |
| `EVENING_DIGEST_TIME` | `20:00` | Evening digest time (HH:MM) |
| `DIGEST_TIMEZONE` | `America/Toronto` | IANA timezone string |
| `CONTEXT_WINDOW_SIZE` | `12` | Recent messages passed to the orchestrator |
| `PROFILE_EXTRACTION_INTERVAL` | `10` | Extract memory facts every N messages |
| `MAX_REACT_ITERATIONS` | `8` | Orchestrator ReAct loop ceiling |
| `TASK_TOKEN_BUDGET` | `8000` | Orchestrator token ceiling; honest fallback on exceed |
| `RESEARCH_MAX_REACT_ITERATIONS` | `12` | Research sub-agent loop ceiling |
| `RESEARCH_TASK_TOKEN_BUDGET` | `12000` | Research sub-agent token ceiling |
| `PLANNER_MAX_REACT_ITERATIONS` | `6` | Schedule planner loop ceiling |
| `PLANNER_TASK_TOKEN_BUDGET` | `6000` | Schedule planner token ceiling |
| `PROACTIVE_MAX_REACT_ITERATIONS` | `5` | Proactive agent loop ceiling |
| `PROACTIVE_TASK_TOKEN_BUDGET` | `8000` | Proactive agent token ceiling |
| `TOOL_OBSERVATION_MAX_CHARS` | `2000` | Truncate tool responses to this length |
| `TAVILY_SNIPPET_MAX_CHARS` | `280` | Per-result snippet length in web search |

---

## Slash Commands

| Command | Description |
|---------|-------------|
| `/profile` | Show everything Arlo knows about you (ephemeral) |
| `/forget <topic>` | Remove memory facts matching a topic (ephemeral) |

Schedule management is conversational — just tell Arlo what you want ("set a gym reminder at 7am", "move my standup to 9:30"). Schedule writes require Discord button approval before they execute.

---

## Project Structure

```
arlo/
├── core/
│   ├── llm.py                        # LLM client abstraction (OpenAI / OpenRouter)
│   ├── settings.py                   # pydantic-settings; fail-fast at startup
│   ├── db.py                         # asyncpg; all table DDL and queries
│   ├── memory/
│   │   ├── store.py                  # mem0 read/write/delete
│   │   ├── extractor.py              # background profile extraction
│   │   └── models.py                 # UserProfile, MemoryEntry
│   ├── agent/
│   │   ├── react.py                  # shared build_react_graph factory
│   │   ├── orchestrator.py           # user-facing agent; MemorySaver + HITL
│   │   ├── researcher.py             # research sub-agent; returns ResearchBrief JSON
│   │   ├── schedule_planner.py       # planner sub-agent; returns SchedulePlan JSON
│   │   ├── proactive.py              # scheduled message composer
│   │   ├── prompts.py                # system prompt builders
│   │   ├── tools.py                  # tool builders + OpenAI schemas per surface
│   │   └── actions.py                # HITL: pending_actions table + Discord View
│   ├── tools/
│   │   ├── search.py                 # Tavily wrapper
│   │   ├── reader.py                 # SSRF validation + Jina Reader
│   │   └── schedules.py              # schedule DB read/write tools
│   ├── scheduler/
│   │   └── digest.py                 # APScheduler setup + job runner
│   ├── interfaces/discord/
│   │   ├── bot.py                    # discord.py setup + startup validation
│   │   ├── handlers.py               # on_message filter gate + agent dispatch
│   │   └── commands.py               # /profile, /forget slash commands
│   └── api.py                        # FastAPI health check
├── evals/                            # LLM-judged eval suite (no Discord/DB required)
│   ├── eval_researcher.py            # 10 scenarios — 88.6% pass
│   ├── eval_schedule_planner.py      # 12 scenarios — 92.9% pass
│   ├── eval_proactive.py             # 10 scenarios — 90.0% pass
│   ├── eval_orchestrator_routing.py  # 12 scenarios — 95.8% pass
│   ├── eval_persona.py               # persona + guardrail eval
│   └── eval_extractor.py             # memory extraction eval
├── tests/                            # pytest unit + integration suite
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Testing

```bash
# Unit + integration tests
pytest

# With coverage
pytest --cov=core --cov-report=term-missing

# LLM-judged evals (requires OPENAI_API_KEY or OPENROUTER_API_KEY)
python -m evals.eval_researcher
python -m evals.eval_schedule_planner
python -m evals.eval_orchestrator_routing
PYTHONIOENCODING=utf-8 python -m evals.eval_proactive
```

---

## Roadmap

- [x] Discord bot + LLM conversation
- [x] Persistent memory — passive extraction + PostgreSQL + pgvector
- [x] Multi-agent orchestrator — research, schedule planner, HITL approval
- [x] Proactive scheduled messages with anti-repetition
- [x] LLM-judged eval suite
- [ ] `/start` onboarding flow
- [ ] `edit_schedule` tool implementation
- [ ] Poll-based schedule runner
- [ ] Telegram interface

### Future goals

- **Productive agent** — task management, goal tracking, and accountability check-ins built on top of the proactive scheduler
- **Richer content sources** — X/Twitter, Reddit, YouTube, Spotify, and other feeds as first-class tools the proactive agent can pull from, so Arlo surfaces content wherever you actually spend time
- **More platform interfaces** — Telegram, iMessage, WhatsApp; the core agent is interface-agnostic

---

## License

[MIT](LICENSE)

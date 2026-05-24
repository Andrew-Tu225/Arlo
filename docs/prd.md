# Arlo — Product Requirements Document (MVP)

**Status:** Draft  
**Date:** 2026-05-22 (updated after eng review)  
**Scope:** Discord-only MVP, single user, self-hosted open source

> **Open source constraint:** Every dependency in this repo must be self-hostable. No managed services (Supabase, mem0 cloud, Railway, etc.) as hard requirements. Self-hosters bring their own API keys (LLM, Tavily) and run storage locally via Docker Compose. The managed cloud product built on top of this is a separate repo and out of scope here.

---

## Problem

AI assistants today are reactive — you have to go to them, remember to ask the right thing, and get a generic answer with no context about you. They forget you between sessions. They talk like customer service bots. And they can't do anything on your behalf without a dozen integrations.

The result: people use AI tools for one-off tasks but don't form habits around them. There's no companion.

---

## Target User

A single tech-savvy early adopter (initially: the builder themselves) who:
- Lives in Discord
- Already uses LLMs but finds them impersonal and stateless
- Has things they'd love a smart friend to handle: finding flights, staying on top of niche news, getting recommendations that actually fit them
- Doesn't want a productivity dashboard — wants a frictionless, conversational experience

---

## Goals

### MVP Goals
1. Arlo can hold a conversation in Discord that sounds like a friend, not a bot
2. Arlo remembers what you've told it across sessions and references it naturally
3. Arlo can complete information-retrieval tasks via web search (flights, news, recommendations, scores)
4. Arlo sends you one proactive daily message with content relevant to you
5. You can view and correct your profile at any time

### Non-Goals (MVP)
- Booking / purchasing anything (surfaces links only)
- Telegram, web dashboard, or any non-Discord surface
- Multi-user support
- OAuth integrations (Gmail, Notion, Google Calendar)
- Voice interface
- Browser automation

---

## Success Metrics (MVP)

| Metric | Target |
|---|---|
| Daily active use | User interacts with Arlo at least 3x/day after week 2 |
| Memory accuracy | Profile reflects at least 10 accurate facts after 2 weeks of use |
| Task completion rate | ≥ 80% of task requests return a useful result (not "I couldn't find that") |
| Proactive message open rate | User acknowledges daily digest ≥ 4 out of 7 days |
| Tone rating (qualitative) | Arlo never sounds like a customer service bot in a session review |

---

## Features

### F1 — Conversational Persona (Week 1–2)
**What:** Arlo responds to messages in Discord with a friend tone — no filler, has opinions, short when appropriate, matches user energy.

**Requirements:**
- System prompt with explicit anti-bot rules (no "Sure! Happy to help!", no bullet points in casual chat)
- System prompt includes basic content guardrails: refuses harmful requests, won't impersonate real people, stays honest about its capabilities
- Context window includes last `CONTEXT_WINDOW_SIZE` (default: 12) messages from local episodic log
- All messages — chat, tasks, and memory updates — go through the unified LangGraph agent (no separate classifier or planner; the model routes itself via tool-calling behavior)
- Tone adaptation is a persona-layer concern: the system prompt instructs the agent to match user energy (casual reply for casual messages, structured answer for task requests)

**Out of scope:** Voice, rich embeds (plain text only for MVP)

---

### F2 — Persistent Memory (Week 3–4)
**What:** Arlo learns facts about the user from conversation and recalls them naturally.

**Requirements:**
- Background extraction job runs after every `PROFILE_EXTRACTION_INTERVAL` (default: 10) messages
- Extracts: interests, preferences, opinions, habits, short-term context vs long-term traits
- Handles contradictions (new fact overwrites old on same dimension)
- Distinguishes short-term context ("in Tokyo this week") from long-term traits ("hates layovers")
- User can ask "what do you know about me?" — returns readable summary
- `/profile` command: shows current profile (Week 3–4, alongside memory layer)
- `/forget [topic]` command: removes specific facts (Week 3–4, alongside memory layer)
- Memory injected into persona system prompt: lightweight summary of top facts; detailed retrieval via `search_memory` tool on demand

**Episodic log:** Each incoming user message is written to a local `episodic_messages` PostgreSQL table (async INSERT, non-blocking). The extraction job reads from this table — not the Discord API — to ensure reliable access. Table uses a 30-day retention policy (older rows pruned weekly); extracted facts persist in mem0 long-term.

**Storage:** mem0 (open-source library, self-hosted mode) backed by PostgreSQL + pgvector, run locally via Docker Compose. No mem0 cloud API required.

---

### F3 — Task Execution via ReAct Loop (Week 7–8)
**What:** User describes a goal in plain language; Arlo searches the web and returns a real answer.

**Requirements:**
- The unified LangGraph agent handles task requests using the `web_search` and `read_url` tools; no separate planner LLM call
- The agent has access to `search_memory` and uses it to retrieve user context (location, preferences) before searching, personalizing queries
- ReAct loop: reason → tool call(s) → observe → repeat → synthesize
- Hard limit: `MAX_REACT_ITERATIONS` (default: 8) iterations maximum; configurable
- Token budget: `TASK_TOKEN_BUDGET` (configurable); on budget hit → honest partial answer, no hallucination
- URL validation before fetching: uses `ipaddress` + `validators` Python libraries to reject private IPs (RFC 1918), loopback (127.x, ::1), link-local (169.254.x.x), non-http(s) schemes, and malformed URLs (SSRF prevention)
- Result includes source URLs inline

**Example tasks that must work:**
- "Find me a direct flight YYZ → NRT around Sept 2" → returns top 2–3 options with prices
- "What's trending in AI today" → 3–5 bullet summary of top stories
- "Find a dentist near [city] taking new patients" → top 3 results with contact info
- "Raptors score last night" → direct answer

---

### F4 — Proactive Daily Digest (Week 5–6)
**What:** Arlo sends one unprompted message per day with content tailored to the user's profile.

**Requirements:**
- APScheduler job runs once daily at `DIGEST_TIME` (default: `09:00`) in `DIGEST_TIMEZONE` (default: `America/Toronto`)
- Job config (schedule, channel, on/off state) persisted in PostgreSQL; re-registered on bot restart with `misfire_grace_time=3600` so a recent restart doesn't miss the day's digest
- Content sources: Tavily search queries built from user profile interests
- Content types: news, trending topics, Reddit discussions (via search), relevant events
- Message must reference at least one user profile fact to feel personalized
- Format: casual, not a newsletter — 2–4 items max, conversational framing
- User can pause/resume with `/digest off` / `/digest on`

---

### F5 — Onboarding (Week 9–10)
**What:** New user runs `/start`, Arlo asks seed questions to bootstrap the profile.

**Requirements:**
- `/start` triggers a short conversation (5–7 questions) covering: location, interests, vibe preference, schedule patterns
- Answers saved directly to memory
- After onboarding, Arlo confirms what it now knows
- Designed to take < 3 minutes

---

### F6 — Profile Commands (Week 3–4)
**What:** User can inspect and edit what Arlo knows.

> Moved from Week 9–10: profile inspection must exist from the moment memory extraction goes live so extraction quality can be validated and corrected early.

**Commands:**
- `/profile` — readable summary of current user profile
- `/forget [topic]` — removes facts matching the topic from memory (e.g., `/forget my job`)
- `/start` — re-runs onboarding or opens profile update flow (Week 9–10)

---

## System Flow

Arlo runs as a single long-running process with two parallel subsystems.

### Subsystem A — Reactive (user-triggered)

Every message from the user goes through the same pipeline:

```
User sends Discord message
    ↓
Filter:
  - Drop bot's own messages
  - Drop messages outside DISCORD_GUILD_ID
  - Drop messages not from DISCORD_USER_ID  ← enforces single-user constraint
  - Drop empty messages
    ↓
INSERT into episodic_messages (async, non-blocking)
Build context window: last CONTEXT_WINDOW_SIZE rows from episodic_messages
    ↓
Unified LangGraph Agent (orchestrator.py)
  System prompt:
    - Persona rules (anti-bot tone, has opinions, matches energy)
    - Content guardrails (refuses harmful requests)
    - Basic profile summary (location, key interests injected from mem0)
    - Tone guidance
  Tools available:
    - web_search(query)   → Tavily API
    - read_url(url)       → URL validation → Jina Reader
    - search_memory(q)    → mem0 semantic search (on-demand user context)
    - remember(fact)      → mem0.add() with contradiction handling
    ↓
  The model decides:
    Casual message       → respond directly (no tool calls)
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
After reply (non-blocking background task):
  message_count % PROFILE_EXTRACTION_INTERVAL == 0?
    → asyncio.create_task(extractor):
        read last N rows from episodic_messages
        ONE LLM call: extract facts (interests, preferences, opinions, habits)
        mem0.add() for each fact (with short-term/long-term tag)
```

### Subsystem B — Proactive (APScheduler-triggered)

APScheduler runs jobs inside the same process and sends Discord messages without a user prompt.

```
APScheduler daily job fires at DIGEST_TIME in DIGEST_TIMEZONE
    ↓
digest.py:
  1. mem0.search("interests preferences habits") → user profile snapshot
  2. Build Tavily queries from profile interests
     e.g. "AI news today", "NBA highlights", "r/ProgrammerHumor top posts"
  3. Tavily search → news, Reddit posts, trending topics
  4. LLM composes casual digest — 2–4 items, references ≥1 profile fact
  5. discord.get_channel(channel_id).send(message)

Job config (schedule, channel_id, on/off state) persisted in PostgreSQL.
On bot restart: re-read config from DB and re-register with misfire_grace_time=3600.
/digest off → pause; /digest on → resume.
```

### Slash Commands

| Command | What it does | Available from |
|---|---|---|
| `/profile` | Shows readable summary of what Arlo knows | Week 3–4 |
| `/forget [topic]` | Removes facts matching the topic from memory | Week 3–4 |
| `/digest on\|off` | Pauses or resumes the daily proactive digest | Week 5–6 |
| `/start` | Onboarding: 5–7 questions to bootstrap the memory profile | Week 9–10 |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENROUTER_API_KEY` | OpenRouter API key (alternative to OpenAI) | — |
| `LLM_PROVIDER` | Which LLM provider to use | `openai` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:yourpassword@localhost:5432/arlo` |
| `TAVILY_API_KEY` | Tavily web search API key | — |
| `DISCORD_BOT_TOKEN` | Discord bot token | — |
| `DISCORD_GUILD_ID` | ID of the Discord server Arlo runs in | — |
| `DISCORD_USER_ID` | Discord user ID of the single target user; messages from all other users are silently dropped | — |
| `DIGEST_TIME` | Time of day for the daily digest in HH:MM format | `09:00` |
| `DIGEST_TIMEZONE` | Timezone for the digest time (IANA timezone string) | `America/Toronto` |
| `ENVIRONMENT` | Runtime environment | `development` |
| `LOG_LEVEL` | Logging verbosity | `info` |
| `CONTEXT_WINDOW_SIZE` | Number of recent messages included in the agent's context window | `12` |
| `PROFILE_EXTRACTION_INTERVAL` | Extract profile facts every N messages | `10` |
| `MAX_REACT_ITERATIONS` | Hard ceiling on the ReAct tool-use loop | `8` |
| `TASK_TOKEN_BUDGET` | Max tokens the ReAct loop may consume per task request; on exceed: honest fallback | `8000` |

All required variables (`DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`, `DISCORD_USER_ID`, `DATABASE_URL`, and at least one of `OPENAI_API_KEY` / `OPENROUTER_API_KEY`, `TAVILY_API_KEY`) are validated at startup. Missing or invalid values produce a clear error message and halt the process.

---

## Technical Constraints

- **Self-hostable by design:** the entire stack must run on a single machine or VPS via `docker compose up`. No managed cloud services as hard dependencies.
- **BYO API keys:** self-hosters supply their own LLM provider key (OpenAI / OpenRouter) and Tavily key. These are the only external services required.
- Single-instance deployment (APScheduler is in-process; no distributed jobs)
- One Discord server / one user for MVP; enforced via `DISCORD_USER_ID` filter in `handlers.py`
- No external payment processing
- All user data stored locally in PostgreSQL (self-hosted container) — no third-party data brokers
- LLM provider abstracted behind `core/llm.py`: switch between OpenAI and OpenRouter via `LLM_PROVIDER` env var
- mem0 used in self-hosted mode (open-source library against local Postgres + pgvector); fallback to raw PostgreSQL queries if mem0 introduces blocking issues
- URL fetching validated via `ipaddress` + `validators` libraries before any URL reaches Jina Reader (SSRF prevention)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| mem0 self-hosted limitations / bugs | Medium | High | Keep mem0 behind an interface (`core/memory/store.py`); ready to swap to raw PostgreSQL queries |
| mem0 contradiction handling untested at scale | Medium | Medium | Write explicit contradiction tests in Week 3–4; validate that new facts actually overwrite old ones on the same dimension |
| pgvector setup complexity for self-hosters | Medium | Medium | Provide `docker-compose.yml` with Postgres + pgvector pre-configured; document setup clearly |
| SSRF via LLM-generated URLs | Medium | High | Validate all URLs with `ipaddress` + `validators` before fetching; test against RFC 1918, loopback, link-local, IPv6 variants |
| Profile extraction LLM cost too high | Low | Medium | Tune `PROFILE_EXTRACTION_INTERVAL`; use a cheaper model for the extraction job specifically |
| Discord API rate limits | Low | Low | Use discord.py's built-in rate limit handling |
| APScheduler jobs not surviving restarts | Medium | Low | Store digest schedule config in DB; re-register on bot startup with `misfire_grace_time=3600` |
| Passive extraction writing wrong/stale facts | Medium | Medium | `/profile` and `/forget` available from Week 3–4 so extraction quality can be validated and corrected early |

---

## Open Questions

1. **LLM provider as default in docs:** OpenAI (GPT-4o) or OpenRouter? OpenRouter gives self-hosters more flexibility (can use Claude, Llama, Mistral, etc. with one key). Recommend: default to OpenAI in docs for simplicity; OpenRouter as the "advanced" option.

2. **mem0 self-hosted complexity:** mem0's open-source library against Postgres + pgvector is the right path, but it's newer and less battle-tested than their managed API. Monitor for issues during Week 3–4 and be ready to build a thin custom layer over `asyncpg` + `pgvector` if mem0 creates friction.

3. **Proactive suggestion quality signal:** How do you know if the daily digest is actually good? Consider adding a thumbs-up/thumbs-down reaction handler on digest messages to collect feedback — low effort, high signal.

4. **asyncio.create_task safety for extraction:** `asyncio.create_task` risks unbounded concurrent extraction jobs if message volume spikes. Monitor in Week 3–4 and add a simple async lock or bounded queue if concurrent extractions become a problem.

---

## Timeline

| Week | Milestone |
|---|---|
| 1–2 | Discord bot running, unified LangGraph agent, basic LLM conversation end-to-end, content guardrails in system prompt, startup validation |
| 3–4 | Memory layer: episodic_messages table, passive extraction, PostgreSQL + pgvector storage, `/profile` + `/forget` commands |
| 5–6 | Daily proactive digest via APScheduler (DIGEST_TIME + DIGEST_TIMEZONE) |
| 7–8 | ReAct tools expansion: Tavily search, Jina Reader, search_memory tool, URL validation (SSRF) |
| 9–10 | Persona tuning, `/start` onboarding |
| 11–12 | Polish, test coverage ≥ 80%, public GitHub launch |

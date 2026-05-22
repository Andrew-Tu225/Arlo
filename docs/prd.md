# Arlo — Product Requirements Document (MVP)

**Status:** Draft  
**Date:** 2026-05-21  
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
- Tone + intent classifier runs on every incoming message (single LLM call returning both)
- Classifier output: `{ tone: casual|task|venting|excited, intent: chat|task|memory_update }`
- Context window includes last N messages for continuity

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
- `/profile` command: shows current profile
- `/forget [topic]` command: removes specific facts
- Memory injected into persona system prompt for every response

**Storage:** mem0 (open-source library, self-hosted mode) backed by PostgreSQL + pgvector, run locally via Docker Compose. No mem0 cloud API required.

---

### F3 — Task Execution via ReAct Loop (Week 7–8)
**What:** User describes a goal in plain language; Arlo searches the web and returns a real answer.

**Requirements:**
- Task planner (LLM) breaks goal into steps
- ReAct loop: reason → search (Tavily) → read page (Jina Reader) → repeat → synthesize
- Hard limit: 8 iterations maximum
- Token budget per task (configurable)
- Intent classifier gates entry: rejects harmful or impossible requests before starting loop
- URL validation before fetching (no raw LLM-generated URLs passed to reader)
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
- APScheduler job runs once daily at a user-configurable time (default: 9am local)
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

### F6 — Profile Commands (Week 9–10)
**What:** User can inspect and edit what Arlo knows.

**Commands:**
- `/profile` — readable summary of current user profile
- `/forget [topic]` — removes specific facts (e.g., `/forget my job`)
- `/start` — re-runs onboarding or opens profile update flow

---

## System Flow

Arlo runs as a single long-running process with two parallel subsystems.

### Subsystem A — Reactive (user-triggered)

Every message from the user goes through the same pipeline:

```
User sends Discord message
    ↓
Filter (drop bot messages, empty messages, wrong guild)
Build context window (last 12 messages)
    ↓
Classify — single LLM call
  → tone:   casual | task | venting | excited   (shapes response style)
  → intent: chat | task | memory_update          (selects the route)
    ↓
Route by intent:

  chat          → retrieve relevant memories (semantic search)
                → build system prompt (persona + memories + tone)
                → LLM reply → Discord

  task          → guardrail check (reject harmful requests)
                → task planner (break goal into steps)
                → ReAct loop: search (Tavily) → read page (Jina) → repeat → synthesize
                → Discord reply with source URLs
                  (if loop hits 8-iteration limit: honest fallback, no hallucination)

  memory_update → write fact to memory (mem0, with contradiction handling)
                → short ack to user

    ↓
After reply (non-blocking background task):
  every 10 messages → extraction job: LLM reads recent messages, extracts facts → memory
```

### Subsystem B — Proactive (scheduler-triggered)

APScheduler runs inside the same process and can send Discord messages at any time without a user prompt. This is how all proactive outreach works for MVP — no webhooks or external triggers needed.

```
Daily job fires (default: 9am, user-configurable)
    ↓
Read user profile from memory (interests, humor, topics they follow)
Build Tavily search queries from profile
  e.g. "AI news today", "NBA scores", trending Reddit posts matching user's humor
    ↓
LLM composes casual 2–4 item message referencing ≥1 profile fact
    ↓
Discord sends message to user's channel (unprompted)
```

Job schedule and on/off state are persisted in PostgreSQL and re-registered on bot restart.

### Slash Commands

| Command | What it does |
|---|---|
| `/start` | Onboarding: 5–7 questions to bootstrap the memory profile |
| `/profile` | Shows a readable summary of what Arlo knows about the user |
| `/forget [topic]` | Removes facts matching the topic from memory |
| `/digest on\|off` | Pauses or resumes the daily proactive digest |

---

## Technical Constraints

- **Self-hostable by design:** the entire stack must run on a single machine or VPS via `docker compose up`. No managed cloud services as hard dependencies.
- **BYO API keys:** self-hosters supply their own LLM provider key (OpenAI / OpenRouter) and Tavily key. These are the only external services required.
- Single-instance deployment (APScheduler is in-process; no distributed jobs)
- One Discord server / one user for MVP
- No external payment processing
- All user data stored locally in PostgreSQL (self-hosted container) — no third-party data brokers
- LLM provider abstracted behind `core/llm.py`: switch between OpenAI and OpenRouter via `LLM_PROVIDER` env var
- mem0 used in self-hosted mode (open-source library against local Postgres + pgvector); fallback to raw PostgreSQL queries if mem0 introduces blocking issues

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| mem0 self-hosted limitations / bugs | Medium | High | Keep mem0 behind an interface (`core/memory/store.py`); ready to swap to raw PostgreSQL queries |
| pgvector setup complexity for self-hosters | Medium | Medium | Provide `docker-compose.yml` with Postgres + pgvector pre-configured; document setup clearly |
| ReAct loop producing hallucinated URLs | Medium | Medium | Validate all URLs from LLM output before fetching |
| Profile extraction LLM cost too high | Low | Medium | Tune `PROFILE_EXTRACTION_INTERVAL`; use a cheaper model for the extraction job specifically |
| Discord API rate limits | Low | Low | Use discord.py's built-in rate limit handling |
| Tavily search quality for niche queries | Medium | Medium | Return "I couldn't find a reliable source" honestly rather than hallucinating |
| APScheduler jobs not surviving restarts | Medium | Low | Store digest schedule config in DB; re-register jobs on bot startup |

---

## Open Questions

1. **LLM provider as default in docs:** OpenAI (GPT-4o) or OpenRouter? OpenRouter gives self-hosters more flexibility (can use Claude, Llama, Mistral, etc. with one key). Recommend: default to OpenAI in docs for simplicity; OpenRouter as the "advanced" option.

2. **mem0 self-hosted complexity:** mem0's open-source library against Postgres + pgvector is the right path, but it's newer and less battle-tested than their managed API. Monitor for issues during Week 3–4 and be ready to build a thin custom layer over `asyncpg` + `pgvector` if mem0 creates friction.

3. **Proactive suggestion quality signal:** How do you know if the daily digest is actually good? Consider adding a thumbs-up/thumbs-down reaction handler on digest messages to collect feedback — low effort, high signal.

4. **ReAct loop model cost:** Should the planner and tool-call steps use the same model or a cheaper one for intermediate steps? Recommend: same model for MVP, optimize per-step later once you have usage data.

---

## Timeline

| Week | Milestone |
|---|---|
| 1–2 | Discord bot running, basic LLM conversation end-to-end, tone/intent classifier |
| 3–4 | Memory layer: passive extraction after N messages, PostgreSQL + pgvector storage, profile recall |
| 5–6 | Daily proactive digest via APScheduler |
| 7–8 | ReAct task execution (Tavily + Jina + LangGraph) |
| 9–10 | Persona tuning, `/start` onboarding, `/profile` / `/forget`, guardrails |
| 11–12 | Polish, test coverage ≥ 80%, public GitHub launch + Discord community |

# Arlo

> A persistent AI companion that reaches out, remembers you, and gets things done.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://img.shields.io/badge/build-in_progress-yellow.svg)]()

---

## What is Arlo?

Arlo is an AI companion that reaches out on its own.

Most AI tools wait to be spoken to. Arlo doesn't. Every morning it searches your interest space and sends you things it knows you'll care about — the way a friend texts you something they saw and thought of you. It also just chats. You can vent, catch up, ask it something random. It's a companion, not a query engine.

It remembers. Your interests, your opinions, your habits — picked up quietly across conversations and recalled naturally. Three months later, it still knows what you care about.

When you need something done, describe it. Arlo searches, reads the sources, and comes back with a real answer — not links, synthesis.

Currently runs on Discord. Telegram and other interfaces on the roadmap.

---

## Features

### 🧠 Persistent Memory

Arlo learns from every conversation — your interests, opinions, preferences, and habits. Three months from now, it still knows what you care about. Memory is extracted passively — Arlo picks things up from what you say without you tagging or saving anything. Built on [mem0](https://mem0.ai/) and PostgreSQL with pgvector for semantic recall across sessions.

### 📬 Proactive Outreach

Arlo doesn't wait. Every morning it searches your interest space — news, releases, results you'd care about — and drops the best ones in your chat the way a friend would text you. It also just chats: you can vent, catch up, or ask it something random. You can pause the daily digest with `/digest off` and resume with `/digest on`.

### 🛠️ Task Execution

Describe a goal in plain language. Arlo breaks it down, searches the web, reads the relevant pages, and returns a synthesized answer with citations — not a list of links, a real answer. Powered by a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct loop with a hard ceiling of 8 tool-use iterations.

---

## How It Works

Arlo runs two subsystems in parallel inside a single process:

- **Reactive** — responds to your Discord messages: classifies tone and intent, retrieves relevant memories, and routes to chat, task, or memory-update handling.
- **Proactive** — an APScheduler job that wakes up daily, reads your interest profile, fetches fresh content, and sends a personalized digest to your Discord channel.

---

## Build on Arlo

Arlo is an open source backend for AI companion products. The core is a reusable, opinionated agent stack:

- **Memory layer** — mem0 + pgvector for persistent, semantic user profiles; passive extraction from conversation history
- **Proactive scheduler** — APScheduler-powered daily outreach that runs inside the bot process; no external queue or cron required
- **ReAct agent** — LangGraph loop with Tavily search and Jina Reader; hard ceiling of 8 iterations

Fork it, build the wrapper your users need, and deploy via Docker Compose on any VPS.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI (GPT-4o) or OpenRouter (configurable) |
| Agent orchestration | LangGraph — ReAct loop, max 8 iterations |
| Memory | mem0 (self-hosted) + PostgreSQL + pgvector |
| Web search | Tavily API |
| URL reading | Jina Reader |
| Discord | discord.py |
| Scheduler | APScheduler (in-process) |
| Backend | FastAPI (health check) |
| Deployment | Docker Compose |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + pgvector)
- API keys: an LLM provider (OpenAI or OpenRouter), Tavily, and a Discord bot token

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourname/arlo.git
cd arlo

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and fill in your API keys

# 5. Start the database
docker compose up -d

# 6. Run the bot
python -m core.interfaces.discord.bot
```

> **Note:** The bot is currently in active development. The command above will confirm the process starts; full functionality arrives incrementally as each subsystem is implemented.

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
| `ENVIRONMENT` | Runtime environment | `development` |
| `LOG_LEVEL` | Logging verbosity | `info` |
| `PROFILE_EXTRACTION_INTERVAL` | Extract profile facts every N messages | `10` |
| `MAX_REACT_ITERATIONS` | Hard ceiling on the ReAct tool-use loop | `8` |

---

## Project Structure

```
arlo/
├── core/
│   ├── llm.py                        # LLM client abstraction
│   ├── memory/                       # User profile storage (mem0 + pgvector)
│   ├── agent/                        # ReAct loop, classifier, persona builder
│   ├── tools/                        # Tavily search + Jina Reader
│   ├── scheduler/                    # Daily proactive digest (APScheduler)
│   ├── interfaces/discord/           # Bot, event handlers, slash commands
│   └── api.py                        # FastAPI health check
├── tests/
├── docs/
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Roadmap

- [x] Project scaffolding & architecture
- [ ] Discord bot + basic LLM conversation
- [ ] Memory: passive extraction + PostgreSQL + pgvector storage
- [ ] Daily proactive digest via APScheduler
- [ ] ReAct task loop (Tavily + Jina + LangGraph)
- [ ] Persona tuning, onboarding `/start`, profile commands, guardrails
- [ ] Polish, testing, and public launch

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.

---

## License

[MIT](LICENSE)

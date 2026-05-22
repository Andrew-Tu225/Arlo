# Arlo

> A persistent AI companion for Discord — texts like a smart friend, remembers you, and gets things done.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://img.shields.io/badge/build-in_progress-yellow.svg)]()

---

## What is Arlo?

Most AI tools treat every conversation like it's your first. Arlo doesn't.

Arlo is an AI companion you talk to through Discord — think of it as a close friend who happens to live in your chat. It remembers you, talks with you like a real person, and over time learns your interests, opinions, and habits. Some days it'll send you something it knows you'll love. Other days you're just catching up. And when you need something done, Arlo can handle that too. No commands, no menus — just a conversation.

---

## Features

### 🧠 Persistent Memory

Arlo quietly learns from every conversation. It picks up on your interests, preferences, and habits — and recalls them naturally later, without you having to repeat yourself. Built on [mem0](https://mem0.ai/) and PostgreSQL with pgvector for semantic recall.

### 💬 Friend-like Interaction

Not a chatbot. Arlo matches your tone, has a consistent voice, and actually engages — like a friend you can vent to, joke with, or just catch up with. It also reaches out on its own: when it finds something it thinks you'd love — a news, new movie release, NBA results, etc — it'll drop it in your chat the way a good friend would, not as a scheduled report, but because it thought of you.

### 🛠️ Agent Capabilities

Describe a goal in plain language — Arlo breaks it down, searches the web, reads sources, and comes back with a real answer and citations. Powered by a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct loop with Tavily web search and Jina Reader.

---

## How It Works

Arlo runs two subsystems in parallel inside a single process:

- **Reactive** — responds to your Discord messages: classifies tone and intent, retrieves relevant memories, and routes to chat, task, or memory-update handling.
- **Proactive** — an APScheduler job that wakes up daily, reads your interest profile, fetches fresh content, and sends a personalized digest to your Discord channel.

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

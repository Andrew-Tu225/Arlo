"""Passive profile extraction — background task, never blocks the response path.

Triggered via asyncio.create_task() after every PROFILE_EXTRACTION_INTERVAL
messages (default: 10). Reads from the episodic_messages PostgreSQL table
(not the Discord API) to ensure reliable access regardless of message
deletion or API rate limits.

Extraction flow:
  1. SELECT last N rows FROM episodic_messages WHERE user_id = $1
  2. ONE LLM call with an extraction prompt:
       — extract facts as (dimension, value, is_short_term) triples
       — dimensions: interests, preferences, opinions, habits, short-term context
  3. For each extracted fact: store.add(fact, user_id, short_term=is_short_term)
       Contradiction handling: mem0 overwrites the old value on the same dimension.

Short-term vs long-term examples:
  short_term=True   "in Tokyo this week", "has a job interview tomorrow"
  short_term=False  "hates layovers", "vegetarian", "obsessed with mechanical keyboards"

Concurrency note: asyncio.create_task is non-blocking but unbounded. If message
volume spikes, add an async lock or bounded queue to prevent concurrent extractions.
"""

from __future__ import annotations

import json
import logging

import asyncpg

from core import db, llm
from core.memory import store
from core.memory.models import EpisodicMessage
from core.settings import get_settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the memory extractor for Arlo — a personal AI companion designed to feel like \
texting a smart, perceptive friend, not a generic chatbot. Arlo builds genuine \
understanding of the user over time so it can give better recommendations, proactively \
surface content that actually matters to them, and respond in ways that feel \
personally relevant rather than generic.

The facts you extract become Arlo's long-term memory. Every stored fact directly \
shapes how personal and useful Arlo feels in future conversations.

Read the conversation and extract meaningful facts about the user. Go beyond literal \
statements — a good friend reads between the lines. If someone says "I've been \
grinding this bug for 6 hours", extract that they're likely a developer, stuck on a \
hard problem, and sounding frustrated — even if none of those words appeared. Facts \
inferred from context are often more valuable than facts that were stated plainly, \
because they're the kind of thing only a perceptive friend would notice.

Return JSON in this exact shape:
{"facts": [{"dimension": "...", "value": "...", "is_short_term": true|false}]}

dimension: short category label
  e.g. "work", "diet", "hobby", "personality", "goal", "mood", "relationship", "location", "opinion"

value: a clear, self-contained statement about the user — prefer a complete phrase over a bare noun
  e.g. "works as a software developer", "follows a vegetarian diet", "currently stuck on a frustrating debugging problem"

is_short_term:
  true  — time-bound or situation-specific ("presenting to the board this Friday", "job interview tomorrow")
  false — stable trait or characteristic ("hates layovers", "obsessed with mechanical keyboards")

Prioritise extracting:
- Work, career, current projects and challenges
- Interests, hobbies, passions
- Lifestyle, diet, habits, routines
- Personality, values, communication style
- Opinions on products, events, ideas
- Relationships and social context
- Current mood, energy level, or stress
- Short-term plans, upcoming events, temporary situations

Skip generic conversational filler, facts about Arlo rather than the user, and \
speculation that goes far beyond what the conversation reasonably supports.
If no meaningful facts exist, return {"facts": []}"""


def _format_conversation(messages: list[EpisodicMessage]) -> str:
    return "\n".join(f"[{m.role}] {m.content}" for m in messages)


async def _extract_facts(conversation: str) -> list[dict]:
    client = llm.get_client()
    response = await client.chat.completions.create(
        model=llm.get_default_model(),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": conversation},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content or '{"facts": []}'
    parsed = json.loads(raw)
    return parsed.get("facts", [])


async def maybe_extract(
    pool: asyncpg.Pool | None,
    user_id: str,
) -> None:
    if pool is None:
        return
    settings = get_settings()
    count = await db.count_user_messages(pool, user_id=user_id)
    if count == 0 or count % settings.profile_extraction_interval != 0:
        return
    try:
        messages = await db.get_recent_messages(
            pool,
            user_id=user_id,
            n=settings.profile_extraction_interval,
        )
        if not messages:
            return
        conversation = _format_conversation(messages)
        facts = await _extract_facts(conversation)
        for fact in facts:
            await store.add(
                f"{fact['dimension']}: {fact['value']}",
                user_id,
                short_term=fact["is_short_term"],
            )
    except Exception:
        logger.exception("Extraction failed for user %s", user_id)

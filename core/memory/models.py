"""Data models for the memory subsystem."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EpisodicMessage:
    """A single row from the episodic_messages table.

    Source of truth for the context window (last CONTEXT_WINDOW_SIZE rows
    passed directly to the agent) and the extraction job input.
    """

    id: int
    user_id: str
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime


@dataclass(frozen=True)
class MemoryEntry:
    """A single fact stored in mem0.

    short_term=True  → time-bound context (decays in relevance)
    short_term=False → stable long-term trait
    """

    id: str
    content: str
    short_term: bool
    created_at: datetime


@dataclass(frozen=True)
class UserProfile:
    """Aggregated snapshot of what Arlo knows about the user.

    Built from mem0.get_all() results. Used by /profile to render a
    readable summary and by persona.py to inject a lightweight fact
    summary into the system prompt.
    """

    user_id: str
    facts: tuple[MemoryEntry, ...]

    def summary(self) -> str:
        if not self.facts:
            return "No facts stored yet."
        return "\n".join(f"- {f.content}" for f in self.facts)

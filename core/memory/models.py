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

        grouped: dict[str, list[str]] = {}
        ungrouped: list[str] = []

        for fact in self.facts:
            if ": " in fact.content:
                dimension, value = fact.content.split(": ", 1)
                grouped.setdefault(dimension, []).append(value)
            else:
                ungrouped.append(fact.content)

        sections: list[str] = []
        for dimension, values in grouped.items():
            lines = [f"**{dimension.capitalize()}**"]
            lines.extend(f"- {v}" for v in values)
            sections.append("\n".join(lines))

        if ungrouped:
            sections.append("\n".join(f"- {c}" for c in ungrouped))

        return "\n\n".join(sections)

"""Data models for the memory subsystem."""


class EpisodicMessage:
    """A single row from the episodic_messages table.

    Source of truth for the context window (last CONTEXT_WINDOW_SIZE rows
    passed directly to the agent) and the extraction job input.

    Fields: id, user_id, role ('user' | 'assistant'), content, created_at.
    """

    pass


class MemoryEntry:
    """A single fact stored in mem0.

    Fields: id (mem0 internal), content, short_term (bool), created_at.
    short_term=True  → time-bound context (decays in relevance)
    short_term=False → stable long-term trait
    """

    pass


class UserProfile:
    """Aggregated snapshot of what Arlo knows about the user.

    Built from mem0.get_all() results. Used by /profile to render a
    readable summary and by persona.py to inject a lightweight fact
    summary into the system prompt.
    """

    pass

"""Memory read/write interface.

Wraps mem0 (self-hosted) for:
  - mem0.add()    — store a fact with short-term/long-term tagging
  - mem0.search() — similarity search (pgvector) to retrieve relevant memories
  - mem0.delete() — remove memories by topic (/forget command)

Contradiction resolution and deduplication are delegated to mem0 internals.
"""

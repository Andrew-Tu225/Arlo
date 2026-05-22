"""Passive profile extraction.

Runs asynchronously every PROFILE_EXTRACTION_INTERVAL messages.
Pulls recent conversation history, calls the LLM once to extract
facts (interests, preferences, habits), and writes them to mem0.
"""

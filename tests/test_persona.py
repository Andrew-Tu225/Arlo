"""Tests for core.agent.persona — system prompt builder."""

import pytest

from core.agent.persona import build_system_prompt


def test_build_system_prompt_returns_string():
    result = build_system_prompt()
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_system_prompt_contains_persona_rules():
    result = build_system_prompt()
    assert "Arlo" in result
    assert "filler" in result.lower()
    assert "opinions" in result.lower() or "opinionated" in result.lower()


def test_build_system_prompt_contains_guardrails():
    result = build_system_prompt()
    assert "harmful" in result.lower()
    assert "impersonate" in result.lower()
    assert "honest" in result.lower() or "don't know" in result.lower()


def test_build_system_prompt_no_memories_omits_profile_section():
    result = build_system_prompt()
    assert "What you know about the user" not in result


def test_build_system_prompt_empty_memories_omits_profile_section():
    result = build_system_prompt(memories=[])
    assert "What you know about the user" not in result


def test_build_system_prompt_with_memories_includes_facts():
    memories = ["likes coffee", "lives in Toronto"]
    result = build_system_prompt(memories=memories)
    assert "What you know about the user" in result
    assert "likes coffee" in result
    assert "lives in Toronto" in result


def test_build_system_prompt_is_deterministic():
    memories = ["vegetarian", "loves mechanical keyboards"]
    assert build_system_prompt(memories=memories) == build_system_prompt(memories=memories)
    assert build_system_prompt() == build_system_prompt()

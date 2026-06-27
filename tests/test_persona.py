"""Tests for core.agent.prompts — system prompt builder."""

from core.agent.prompts import build_orchestrator_prompt, get_temporal_context


def test_memories_injected_when_provided():
    result = build_orchestrator_prompt(memories=["likes coffee", "lives in Toronto"])
    assert "likes coffee" in result
    assert "lives in Toronto" in result
    assert "What you know about the user" in result


def test_profile_section_absent_without_memories():
    assert "What you know about the user" not in build_orchestrator_prompt()
    assert "What you know about the user" not in build_orchestrator_prompt(memories=[])


def test_temporal_context_fields_present():
    result = get_temporal_context()
    assert "Current Date" in result
    assert "Current Time" in result
    assert "Timezone" in result


def test_orchestrator_prompt_includes_temporal_context():
    assert "TEMPORAL CONTEXT" in build_orchestrator_prompt()

"""Tests for core/memory/store.py — mem0 cloud wrapper."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import core.memory.store as store
from core.memory.models import MemoryEntry

_NOW_STR = "2026-01-01T12:00:00Z"
_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_client(search_results=None, get_all_results=None):
    client = MagicMock()
    client.add.return_value = {}
    client.search.return_value = search_results or []
    client.delete.return_value = {}
    client.get_all.return_value = get_all_results or []
    return client


def _memory_result(id_="m1", memory="User is vegetarian", short_term=False):
    return {
        "id": id_,
        "memory": memory,
        "metadata": {"short_term": short_term},
        "created_at": _NOW_STR,
    }


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_calls_client_add(self):
        client = _make_client()
        with patch.object(store, "_get_client", return_value=client):
            await store.add("I love spicy food", "u1", short_term=False)
        client.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_passes_fact_as_user_message(self):
        client = _make_client()
        with patch.object(store, "_get_client", return_value=client):
            await store.add("I love spicy food", "u1", short_term=False)
        call_kwargs = client.add.call_args
        messages = call_kwargs.args[0]
        assert messages == [{"role": "user", "content": "I love spicy food"}]

    @pytest.mark.asyncio
    async def test_add_passes_user_id(self):
        client = _make_client()
        with patch.object(store, "_get_client", return_value=client):
            await store.add("fact", "user42", short_term=False)
        call_kwargs = client.add.call_args
        assert call_kwargs.kwargs.get("user_id") == "user42"

    @pytest.mark.asyncio
    async def test_add_passes_short_term_in_metadata(self):
        client = _make_client()
        with patch.object(store, "_get_client", return_value=client):
            await store.add("in Tokyo this week", "u1", short_term=True)
        call_kwargs = client.add.call_args
        metadata = call_kwargs.kwargs.get("metadata", {})
        assert metadata.get("short_term") is True


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_list_of_strings(self):
        client = _make_client(search_results=[_memory_result(memory="User is vegetarian")])
        with patch.object(store, "_get_client", return_value=client):
            results = await store.search("diet", "u1")
        assert results == ["User is vegetarian"]

    @pytest.mark.asyncio
    async def test_search_returns_empty_list_when_no_results(self):
        client = _make_client(search_results=[])
        with patch.object(store, "_get_client", return_value=client):
            results = await store.search("diet", "u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_passes_user_id(self):
        client = _make_client()
        with patch.object(store, "_get_client", return_value=client):
            await store.search("diet", "user99")
        call_kwargs = client.search.call_args
        assert call_kwargs.kwargs.get("user_id") == "user99"

    @pytest.mark.asyncio
    async def test_search_passes_limit_as_top_k(self):
        client = _make_client()
        with patch.object(store, "_get_client", return_value=client):
            await store.search("anything", "u1", limit=3)
        call_kwargs = client.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 3


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_calls_delete_per_match(self):
        results = [_memory_result("id1"), _memory_result("id2")]
        client = _make_client(search_results=results)
        with patch.object(store, "_get_client", return_value=client):
            await store.delete("vegetarian", "u1")
        assert client.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_returns_count(self):
        results = [_memory_result("id1"), _memory_result("id2")]
        client = _make_client(search_results=results)
        with patch.object(store, "_get_client", return_value=client):
            count = await store.delete("vegetarian", "u1")
        assert count == 2

    @pytest.mark.asyncio
    async def test_delete_returns_zero_when_no_matches(self):
        client = _make_client(search_results=[])
        with patch.object(store, "_get_client", return_value=client):
            count = await store.delete("nothing", "u1")
        assert count == 0
        client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_passes_correct_memory_ids(self):
        results = [_memory_result("abc"), _memory_result("xyz")]
        client = _make_client(search_results=results)
        with patch.object(store, "_get_client", return_value=client):
            await store.delete("topic", "u1")
        deleted_ids = [c.args[0] for c in client.delete.call_args_list]
        assert "abc" in deleted_ids
        assert "xyz" in deleted_ids


class TestGetAll:
    @pytest.mark.asyncio
    async def test_get_all_returns_list_of_memory_entries(self):
        client = _make_client(get_all_results=[_memory_result()])
        with patch.object(store, "_get_client", return_value=client):
            results = await store.get_all("u1")
        assert len(results) == 1
        assert isinstance(results[0], MemoryEntry)

    @pytest.mark.asyncio
    async def test_get_all_maps_fields_correctly(self):
        raw = _memory_result(id_="m42", memory="hates layovers", short_term=False)
        client = _make_client(get_all_results=[raw])
        with patch.object(store, "_get_client", return_value=client):
            results = await store.get_all("u1")
        entry = results[0]
        assert entry.id == "m42"
        assert entry.content == "hates layovers"
        assert entry.short_term is False
        assert entry.created_at == _NOW

    @pytest.mark.asyncio
    async def test_get_all_returns_empty_list_when_no_memories(self):
        client = _make_client(get_all_results=[])
        with patch.object(store, "_get_client", return_value=client):
            results = await store.get_all("u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_all_handles_missing_metadata(self):
        raw = {"id": "m1", "memory": "fact", "created_at": _NOW_STR}
        client = _make_client(get_all_results=[raw])
        with patch.object(store, "_get_client", return_value=client):
            results = await store.get_all("u1")
        assert results[0].short_term is False

    @pytest.mark.asyncio
    async def test_get_all_handles_none_created_at(self):
        raw = {"id": "m1", "memory": "fact", "metadata": {}, "created_at": None}
        client = _make_client(get_all_results=[raw])
        with patch.object(store, "_get_client", return_value=client):
            results = await store.get_all("u1")
        assert isinstance(results[0].created_at, datetime)

    @pytest.mark.asyncio
    async def test_get_all_passes_user_id(self):
        client = _make_client()
        with patch.object(store, "_get_client", return_value=client):
            await store.get_all("user77")
        call_kwargs = client.get_all.call_args
        assert call_kwargs.kwargs.get("user_id") == "user77"

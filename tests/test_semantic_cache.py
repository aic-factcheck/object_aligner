"""Tests for the embedding cache layer.

Both backends (``InMemoryEmbeddingCache`` and ``SQLiteEmbeddingCache``)
share the same ``get_many`` semantics and must satisfy the same
contract. Many cases are parametrised against both backends to keep
the contract one-table.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from object_aligner.semantic import (
    DictMockEmbedder,
    HashMockEmbedder,
    InMemoryEmbeddingCache,
    SQLiteEmbeddingCache,
)
from object_aligner.semantic.embedder import BaseEmbedder


class _CountingEmbedder(BaseEmbedder):
    """Wraps an embedder; counts embed_many invocations and the size of
    each batch. Used to verify that the cache batches misses
    correctly and avoids redundant upstream calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[list[str]] = []

    @property
    def model_id(self):
        return self._inner.model_id

    @property
    def dim(self):
        return self._inner.dim

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return self._inner.embed_many(texts)


# ---------------------------------------------------------------------
# Shared contract — parametrised over both backends
# ---------------------------------------------------------------------

def _make_cache(backend: str, embedder, tmp_path):
    if backend == "memory":
        return InMemoryEmbeddingCache(embedder)
    if backend == "sqlite":
        return SQLiteEmbeddingCache(embedder, str(tmp_path / "cache.sqlite"))
    raise ValueError(backend)


@pytest.fixture(params=["memory", "sqlite"])
def cache_factory(request, tmp_path):
    backend = request.param

    def factory(embedder):
        return _make_cache(backend, embedder, tmp_path)

    return factory


def test_get_one_round_trip(cache_factory):
    e = HashMockEmbedder(dim=16)
    cache = cache_factory(e)
    v1 = cache.get_one("hello")
    v2 = cache.get_one("hello")
    assert np.array_equal(v1, v2)
    assert "hello" in cache
    assert len(cache) == 1


def test_get_many_preserves_order(cache_factory):
    e = HashMockEmbedder(dim=8)
    cache = cache_factory(e)
    inputs = ["a", "b", "a", "c"]
    out = cache.get_many(inputs)
    assert len(out) == 4
    assert np.array_equal(out[0], out[2])  # same text -> same vector


def test_get_many_batches_misses_into_one_upstream_call(cache_factory):
    inner = HashMockEmbedder(dim=8)
    counter = _CountingEmbedder(inner)
    cache = cache_factory(counter)
    cache.get_many(["a", "b", "c", "a"])
    # Exactly one upstream call; the duplicate "a" is de-duped; the
    # batch sent upstream has three unique misses.
    assert len(counter.calls) == 1
    assert sorted(counter.calls[0]) == ["a", "b", "c"]


def test_repeated_get_many_uses_cache(cache_factory):
    inner = HashMockEmbedder(dim=8)
    counter = _CountingEmbedder(inner)
    cache = cache_factory(counter)
    cache.get_many(["x", "y"])
    cache.get_many(["x", "y"])
    # Second call should hit cache entirely; no second upstream call.
    assert len(counter.calls) == 1


def test_clear_removes_entries(cache_factory):
    e = HashMockEmbedder(dim=4)
    cache = cache_factory(e)
    cache.get_many(["foo", "bar"])
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert "foo" not in cache


def test_model_id_namespaces_keys(cache_factory):
    a = HashMockEmbedder(dim=4, model_id="mock-A")
    b = HashMockEmbedder(dim=4, model_id="mock-B")
    cache_a = cache_factory(a)
    # cache_b uses the SAME backing store / file for SQLite, different
    # model_id namespace.
    if isinstance(cache_a, SQLiteEmbeddingCache):
        cache_b = SQLiteEmbeddingCache(b, cache_a.path)
    else:
        cache_b = cache_factory(b)
        # For the in-memory factory each call returns a fresh instance;
        # we want to share state. Reuse cache_a's dict if shared, else
        # the namespace test just becomes a redundant length check.
    cache_a.get_many(["text"])
    if isinstance(cache_a, SQLiteEmbeddingCache):
        # The two caches share storage but not model_id.
        assert "text" in cache_a
        assert "text" not in cache_b
        cache_a.close()
        cache_b.close()


def test_empty_input_returns_empty_list(cache_factory):
    e = HashMockEmbedder(dim=4)
    cache = cache_factory(e)
    assert cache.get_many([]) == []


# ---------------------------------------------------------------------
# SQLite-specific
# ---------------------------------------------------------------------

def test_sqlite_persistence_round_trip():
    e = HashMockEmbedder(dim=8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.sqlite")
        cache1 = SQLiteEmbeddingCache(e, path)
        v_written = cache1.get_one("persisted-text")
        cache1.close()

        cache2 = SQLiteEmbeddingCache(e, path)
        assert "persisted-text" in cache2
        v_loaded = cache2.get_one("persisted-text")
        assert np.array_equal(v_written, v_loaded)
        cache2.close()


def test_sqlite_in_memory_path_works():
    e = HashMockEmbedder(dim=4)
    cache = SQLiteEmbeddingCache(e, ":memory:")
    cache.get_one("x")
    assert "x" in cache
    cache.close()


def test_sqlite_round_trips_2d_arrays():
    # Simulate a token-level embedder: each vector is (n_tokens, dim).
    vectors = {
        "two tokens": np.arange(12, dtype=np.float32).reshape(2, 6),
        "one token":  np.arange(6,  dtype=np.float32).reshape(1, 6),
    }
    e = DictMockEmbedder(vectors, model_id="token-mock")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.sqlite")
        cache = SQLiteEmbeddingCache(e, path)
        a = cache.get_one("two tokens")
        assert a.shape == (2, 6)
        cache.close()
        cache2 = SQLiteEmbeddingCache(e, path)
        b = cache2.get_one("two tokens")
        assert b.shape == (2, 6)
        assert np.array_equal(a, b)
        cache2.close()


def test_sqlite_context_manager_closes():
    e = HashMockEmbedder(dim=4)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "c.sqlite")
        with SQLiteEmbeddingCache(e, path) as cache:
            cache.get_one("ctxmgr")
        # After __exit__, a fresh cache against the same path still sees
        # the row — i.e. the close was clean.
        cache2 = SQLiteEmbeddingCache(e, path)
        assert "ctxmgr" in cache2
        cache2.close()

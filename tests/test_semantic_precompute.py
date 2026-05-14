"""Tests for the schema-aware cache pre-warmer."""

from __future__ import annotations

import numpy as np
import pytest

from object_aligner import ObjectAligner
from object_aligner.semantic import (
    HashMockEmbedder,
    InMemoryEmbeddingCache,
    cosine_similarity_metric,
    precompute,
)
from object_aligner.semantic.embedder import BaseEmbedder


class _CountingEmbedder(BaseEmbedder):
    """Wraps an embedder; records every embed_many call to verify
    pre-warming batched the right strings together."""

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


def _build(schema, embedder_dim=32):
    counter = _CountingEmbedder(HashMockEmbedder(dim=embedder_dim))
    cache = InMemoryEmbeddingCache(counter)
    semantic = cosine_similarity_metric(cache)
    aligner = ObjectAligner(schema, custom_metrics={"string": {"semantic": semantic}})
    return aligner, cache, counter


# ---------------------------------------------------------------------
# Discovery: which paths use which caches?
# ---------------------------------------------------------------------

def test_precompute_simple_string_schema():
    aligner, cache, counter = _build({"type": "string", "score": "semantic"})
    report = precompute(aligner, "alpha", "beta")
    assert counter.calls, "expected exactly one upstream embed_many call"
    assert len(counter.calls) == 1
    assert sorted(counter.calls[0]) == ["alpha", "beta"]
    assert report == {counter.model_id: 2}


def test_precompute_inside_dict():
    aligner, cache, counter = _build(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "score": "semantic"},
                "title":   {"type": "string", "score": "jaro"},
            },
        }
    )
    gold = {"summary": "revenue grew", "title": "Q3"}
    pred = {"summary": "earnings up",  "title": "Q3 2025"}
    report = precompute(aligner, gold, pred)
    # Only the "summary" strings should reach the cache; "title" uses
    # built-in Jaro and is not cache-backed.
    assert len(counter.calls) == 1
    assert sorted(counter.calls[0]) == ["earnings up", "revenue grew"]
    assert report[counter.model_id] == 2


def test_precompute_inside_array_with_items():
    aligner, cache, counter = _build(
        {
            "type": "array",
            "order": "align",
            "items": {"type": "string", "score": "semantic"},
        }
    )
    gold = ["a", "b", "c"]
    pred = ["b", "d"]
    precompute(aligner, gold, pred)
    assert len(counter.calls) == 1
    assert sorted(counter.calls[0]) == ["a", "b", "c", "d"]


def test_precompute_mixed_schema_drops_non_embedding_paths():
    # title -> jaro (built-in), summary -> semantic (embedded), tags ->
    # items with semantic embed_many.
    aligner, cache, counter = _build(
        {
            "type": "object",
            "properties": {
                "title":   {"type": "string", "score": "jaro"},
                "summary": {"type": "string", "score": "semantic"},
                "tags": {
                    "type": "array",
                    "order": "align",
                    "items": {"type": "string", "score": "semantic"},
                },
            },
        }
    )
    gold = {"title": "x", "summary": "hello", "tags": ["red", "blue"]}
    pred = {"title": "y", "summary": "hello", "tags": ["green"]}
    precompute(aligner, gold, pred)
    assert len(counter.calls) == 1
    assert sorted(counter.calls[0]) == ["blue", "green", "hello", "red"]


# ---------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------

def test_precompute_is_idempotent():
    aligner, cache, counter = _build({"type": "string", "score": "semantic"})
    precompute(aligner, "alpha")
    precompute(aligner, "alpha")
    # First call: 1 batched embed; second call: cache fully warm,
    # cache.get_many sees no misses, no upstream call.
    assert len(counter.calls) == 1


def test_precompute_then_metric_is_cache_warm():
    aligner, cache, counter = _build({"type": "string", "score": "semantic"})
    precompute(aligner, "alpha", "beta")
    counter.calls.clear()
    aligner.metric("alpha", "beta")
    assert counter.calls == []  # no upstream traffic — fully cached


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_precompute_returns_empty_when_no_cached_metrics():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    assert precompute(aligner, "x", "y") == {}


def test_precompute_handles_missing_keys_gracefully():
    aligner, cache, counter = _build(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "score": "semantic"},
            },
        }
    )
    # pred is missing the "summary" key; precompute should not crash,
    # and the upstream batch should contain only gold's string.
    precompute(aligner, {"summary": "have it"}, {})
    assert len(counter.calls) == 1
    assert counter.calls[0] == ["have it"]


def test_precompute_groups_distinct_caches_separately():
    # Two metrics, two caches, registered under different names.
    counter_a = _CountingEmbedder(HashMockEmbedder(dim=16, model_id="mock-A"))
    cache_a = InMemoryEmbeddingCache(counter_a)
    counter_b = _CountingEmbedder(HashMockEmbedder(dim=16, model_id="mock-B"))
    cache_b = InMemoryEmbeddingCache(counter_b)
    aligner = ObjectAligner(
        {
            "type": "object",
            "properties": {
                "left":  {"type": "string", "score": "semA"},
                "right": {"type": "string", "score": "semB"},
            },
        },
        custom_metrics={
            "string": {
                "semA": cosine_similarity_metric(cache_a),
                "semB": cosine_similarity_metric(cache_b),
            },
        },
    )
    precompute(aligner, {"left": "L1", "right": "R1"}, {"left": "L2", "right": "R2"})
    assert sorted(counter_a.calls[0]) == ["L1", "L2"]
    assert sorted(counter_b.calls[0]) == ["R1", "R2"]

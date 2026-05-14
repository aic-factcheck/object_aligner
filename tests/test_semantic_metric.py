"""Tests for the cosine-similarity metric and its integration with OA."""

from __future__ import annotations

import math

import numpy as np
import pytest

from object_aligner import ObjectAligner
from object_aligner.semantic import (
    CosineSimilarityMetric,
    DictMockEmbedder,
    HashMockEmbedder,
    InMemoryEmbeddingCache,
    cosine_similarity_metric,
)


# ---------------------------------------------------------------------
# Closed-form cosine on hand-crafted vectors
# ---------------------------------------------------------------------

def test_metric_known_cosine_dict_mock():
    # cos("warm","hot") = 0.95 by construction.
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.95, math.sqrt(1 - 0.95 ** 2)], dtype=np.float32)
    cache = InMemoryEmbeddingCache(DictMockEmbedder({"warm": a, "hot": b}))
    m = cosine_similarity_metric(cache)
    assert abs(m("warm", "hot") - 0.95) < 1e-6


def test_metric_identical_strings_score_one():
    cache = InMemoryEmbeddingCache(HashMockEmbedder(dim=32))
    m = cosine_similarity_metric(cache)
    # Cosine of a vector with itself is 1.0 modulo float-precision wobble;
    # the metric clips to [0, 1] but does not snap to exactly 1.0.
    assert m("anything", "anything") == pytest.approx(1.0, abs=1e-6)


def test_metric_clip_negative_cosine():
    # Anti-parallel vectors: cosine = -1.
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    cache = InMemoryEmbeddingCache(DictMockEmbedder({"x": a, "y": b}))
    m = cosine_similarity_metric(cache, sign_convention="clip")
    assert m("x", "y") == 0.0


def test_metric_affine_negative_cosine_maps_to_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    cache = InMemoryEmbeddingCache(DictMockEmbedder({"x": a, "y": b}))
    m = cosine_similarity_metric(cache, sign_convention="affine")
    assert abs(m("x", "y") - 0.0) < 1e-6


def test_metric_affine_orthogonal_maps_to_half():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    cache = InMemoryEmbeddingCache(DictMockEmbedder({"x": a, "y": b}))
    m = cosine_similarity_metric(cache, sign_convention="affine")
    assert abs(m("x", "y") - 0.5) < 1e-6


def test_metric_normalises_unnormalised_input():
    # Two parallel vectors with different magnitudes — cosine must
    # still be 1.0 after the metric's defensive normalisation.
    a = np.array([3.0, 0.0], dtype=np.float32)
    b = np.array([7.0, 0.0], dtype=np.float32)
    cache = InMemoryEmbeddingCache(DictMockEmbedder({"x": a, "y": b}))
    m = cosine_similarity_metric(cache)
    assert abs(m("x", "y") - 1.0) < 1e-6


def test_metric_invalid_sign_convention_raises():
    cache = InMemoryEmbeddingCache(HashMockEmbedder(dim=4))
    with pytest.raises(ValueError):
        cosine_similarity_metric(cache, sign_convention="bogus")


def test_metric_carries_cache_back_reference():
    cache = InMemoryEmbeddingCache(HashMockEmbedder(dim=4))
    m = cosine_similarity_metric(cache)
    assert m.cache is cache
    assert m.kind == "cosine_similarity"


# ---------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------

def test_cosine_similarity_metric_class_wrapper():
    cache = InMemoryEmbeddingCache(HashMockEmbedder(dim=16))
    m = CosineSimilarityMetric(cache)
    assert m("foo", "foo") == pytest.approx(1.0, abs=1e-6)
    assert m.cache is cache
    assert m.kind == "cosine_similarity"


# ---------------------------------------------------------------------
# Integration with ObjectAligner.custom_metrics
# ---------------------------------------------------------------------

def test_integration_with_object_aligner_metric():
    cache = InMemoryEmbeddingCache(HashMockEmbedder(dim=128, model_id="oa-test"))
    semantic = cosine_similarity_metric(cache)
    aligner = ObjectAligner(
        {"type": "string", "score": "semantic"},
        custom_metrics={"string": {"semantic": semantic}},
    )
    assert aligner.metric("identical", "identical")["score"] == pytest.approx(1.0, abs=1e-6)
    # Different strings should score in [0, 1] but not 1.0 for hash mock.
    s = aligner.metric("apple", "orange")["score"]
    assert 0.0 <= s <= 1.0
    assert s != 1.0


def test_integration_respects_threshold():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.95, math.sqrt(1 - 0.95 ** 2)], dtype=np.float32)
    cache = InMemoryEmbeddingCache(DictMockEmbedder({"warm": a, "hot": b}))
    semantic = cosine_similarity_metric(cache)
    aligner = ObjectAligner(
        {"type": "string", "score": "semantic", "threshold": 0.99},
        custom_metrics={"string": {"semantic": semantic}},
    )
    # cos = 0.95, below the 0.99 threshold -> clamped to 0.0 by OA.
    assert aligner.metric("warm", "hot")["score"] == 0.0


def test_integration_inside_dict_schema():
    cache = InMemoryEmbeddingCache(HashMockEmbedder(dim=64, model_id="oa-dict"))
    semantic = cosine_similarity_metric(cache)
    aligner = ObjectAligner(
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "score": "jaro"},
                "body":  {"type": "string", "score": "semantic"},
            },
        },
        custom_metrics={"string": {"semantic": semantic}},
    )
    r = aligner.metric(
        gold={"title": "Q3", "body": "Revenue grew 12%."},
        pred={"title": "Q3", "body": "Revenue grew 12%."},
    )
    assert r["score"] == pytest.approx(1.0, abs=1e-6)

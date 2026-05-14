"""Tests for the semantic embedder protocol + mock embedders.

The full semantic stack lives under ``object_aligner.semantic.*``; this
file exercises only the lowest layer (the embedders themselves). Cache
and metric behaviour have their own files. No test in the suite makes
a network call — ``OpenAIEmbedder`` is only exercised here via a
monkeypatched ``openai.OpenAI`` (see ``test_openai_embedder_with_mock``
below).
"""

from __future__ import annotations

import numpy as np
import pytest

from object_aligner.semantic import (
    BaseEmbedder,
    DictMockEmbedder,
    Embedder,
    HashMockEmbedder,
    OpenAIEmbedder,
)


# ---------------------------------------------------------------------
# HashMockEmbedder
# ---------------------------------------------------------------------

def test_hash_mock_returns_unit_vectors():
    e = HashMockEmbedder(dim=64)
    v = e.embed_one("hello")
    assert v.shape == (64,)
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_hash_mock_is_deterministic_across_calls():
    e = HashMockEmbedder(dim=64)
    assert np.array_equal(e.embed_one("foo"), e.embed_one("foo"))


def test_hash_mock_is_deterministic_across_instances():
    a = HashMockEmbedder(dim=64)
    b = HashMockEmbedder(dim=64)
    assert np.array_equal(a.embed_one("foo"), b.embed_one("foo"))


def test_hash_mock_distinct_strings_diverge():
    e = HashMockEmbedder(dim=512)  # high dim -> orthogonal-ish
    v1 = e.embed_one("apple")
    v2 = e.embed_one("orange")
    cos = float(np.dot(v1, v2))
    # In 512-d, two independent random unit vectors have cosine ~0 with
    # std ~1/sqrt(512) ~ 0.044. |cos| < 0.3 is enormously safe.
    assert abs(cos) < 0.3


def test_hash_mock_batch_matches_single_calls():
    e = HashMockEmbedder(dim=16)
    batch = e.embed_many(["a", "b", "c"])
    single = [e.embed_one(t) for t in ["a", "b", "c"]]
    for x, y in zip(batch, single):
        assert np.array_equal(x, y)


def test_hash_mock_rejects_non_string():
    e = HashMockEmbedder(dim=8)
    with pytest.raises(TypeError):
        e.embed_many(["ok", 42])  # type: ignore[list-item]


def test_hash_mock_validates_construction():
    with pytest.raises(ValueError):
        HashMockEmbedder(dim=0)
    with pytest.raises(ValueError):
        HashMockEmbedder(dim=-1)
    with pytest.raises(ValueError):
        HashMockEmbedder(model_id="")


def test_hash_mock_model_id_is_namespaced():
    a = HashMockEmbedder(dim=8, model_id="hash-mock-A")
    b = HashMockEmbedder(dim=8, model_id="hash-mock-B")
    assert a.model_id != b.model_id


# ---------------------------------------------------------------------
# DictMockEmbedder
# ---------------------------------------------------------------------

def test_dict_mock_returns_registered_vectors():
    vectors = {
        "warm": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "hot":  np.array([0.95, np.sqrt(1 - 0.95 ** 2), 0.0], dtype=np.float32),
    }
    e = DictMockEmbedder(vectors)
    v_w, v_h = e.embed_many(["warm", "hot"])
    cos = float(np.dot(v_w, v_h) / (np.linalg.norm(v_w) * np.linalg.norm(v_h)))
    assert abs(cos - 0.95) < 1e-6


def test_dict_mock_raises_on_unknown_text():
    e = DictMockEmbedder({"only-this": np.zeros(3)})
    with pytest.raises(KeyError):
        e.embed_one("missing")


def test_dict_mock_pins_dim_from_first_1d_entry():
    e = DictMockEmbedder({"a": np.zeros(7)})
    assert e.dim == 7


def test_dict_mock_validates_construction():
    with pytest.raises(ValueError):
        DictMockEmbedder({})
    with pytest.raises(TypeError):
        DictMockEmbedder({"a": [1.0, 2.0]})  # type: ignore[dict-item]


# ---------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------

def test_hash_mock_satisfies_protocol():
    e = HashMockEmbedder(dim=4)
    assert isinstance(e, Embedder)


def test_dict_mock_satisfies_protocol():
    e = DictMockEmbedder({"x": np.zeros(2)})
    assert isinstance(e, Embedder)


def test_base_embedder_default_embed_one():
    class Naive(BaseEmbedder):
        @property
        def model_id(self):
            return "naive"
        @property
        def dim(self):
            return 3
        def embed_many(self, texts):
            return [np.array([i, i, i], dtype=np.float32) for i, _ in enumerate(texts)]

    e = Naive()
    one = e.embed_one("anything")
    assert np.array_equal(one, np.array([0, 0, 0], dtype=np.float32))


# ---------------------------------------------------------------------
# OpenAIEmbedder (no network) — verifies request shape via monkeypatch
# ---------------------------------------------------------------------

class _FakeEmbeddingResponse:
    def __init__(self, vectors):
        self.data = [type("D", (), {"embedding": v})() for v in vectors]


class _FakeEmbeddingsClient:
    """Stand-in for ``openai.OpenAI(...).embeddings``."""

    def __init__(self, captured_requests):
        self.captured_requests = captured_requests

    def create(self, *, model, input, **kwargs):
        self.captured_requests.append({"model": model, "input": list(input), **kwargs})
        # Return identity-ish vectors: a one-hot per index for determinism.
        n = len(input)
        return _FakeEmbeddingResponse([
            [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)
        ])


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.embeddings = _FakeEmbeddingsClient(getattr(_FakeOpenAI, "_captured", []))


def test_openai_embedder_constructs_and_batches(monkeypatch):
    pytest.importorskip("openai")
    import openai
    _FakeOpenAI._captured = []
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    e = OpenAIEmbedder(
        model="fake-model",
        base_url="http://localhost:8333/v1",
        max_batch_size=2,
    )
    # 5 inputs and max_batch_size=2 should produce 3 HTTP calls (2+2+1).
    out = e.embed_many(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    assert all(isinstance(v, np.ndarray) for v in out)
    assert len(_FakeOpenAI._captured) == 3
    assert [r["input"] for r in _FakeOpenAI._captured] == [
        ["a", "b"], ["c", "d"], ["e"],
    ]
    for r in _FakeOpenAI._captured:
        assert r["model"] == "fake-model"


def test_openai_embedder_model_id_includes_base_url(monkeypatch):
    pytest.importorskip("openai")
    import openai
    _FakeOpenAI._captured = []
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    e = OpenAIEmbedder(model="Qwen3", base_url="http://localhost:8333/v1")
    assert "localhost:8333" in e.model_id
    assert "Qwen3" in e.model_id
    assert "d=default" in e.model_id


def test_openai_embedder_dimensions_passes_through(monkeypatch):
    pytest.importorskip("openai")
    import openai
    _FakeOpenAI._captured = []
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    e = OpenAIEmbedder(model="any", dimensions=512)
    e.embed_many(["x"])
    assert _FakeOpenAI._captured[-1].get("dimensions") == 512
    assert "d=512" in e.model_id


def test_openai_embedder_validates_construction():
    pytest.importorskip("openai")
    with pytest.raises(ValueError):
        OpenAIEmbedder(model="")
    with pytest.raises(ValueError):
        OpenAIEmbedder(model="ok", max_batch_size=0)

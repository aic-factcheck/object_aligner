"""Deterministic mock embedders used by the test suite.

Neither mock makes network calls. Both are public so library users can
plug them into their own tests without redefining the same primitives.

* :class:`HashMockEmbedder` — BLAKE2b-seeded RNG produces a stable
  unit-normalised vector per input string. Identical inputs always
  return identical vectors; distinct inputs return near-orthogonal
  vectors in high dimension. Use it to exercise plumbing (caching,
  batching, integration with :class:`ObjectAligner`) — *not* to make
  semantic claims about whether ``"hello"`` and ``"hi"`` are close.

* :class:`DictMockEmbedder` — a hand-curated ``{text: ndarray}`` map.
  Raises :class:`KeyError` on unknown text (explicit by design). Use it
  when you need a specific cosine value, e.g. to drive a known
  Hungarian matrix.
"""

from __future__ import annotations

import hashlib

import numpy as np

from object_aligner.semantic.embedder import BaseEmbedder


class HashMockEmbedder(BaseEmbedder):
    """Deterministic hash-based mock embedder.

    Each input string is hashed with BLAKE2b (8-byte digest), the digest
    is reinterpreted as a 64-bit seed, and that seed drives a fresh
    :class:`numpy.random.Generator` to draw a standard-normal vector of
    length :attr:`dim`. The vector is L2-normalised before return.

    BLAKE2b is used rather than Python's :func:`hash` so that the output
    is stable across Python invocations and platforms (Python's
    ``hash()`` is randomised per-process by default).

    Args:
        dim: Dimensionality of the produced vectors. Default ``32`` —
            high enough that distinct strings produce vectors whose
            cosine concentrates near zero with low variance, low enough
            that test arithmetic stays cheap.
        model_id: Cache-key namespace; defaults to ``"hash-mock"``. Set
            it to something unique if you want to run two
            :class:`HashMockEmbedder` instances against the same cache
            without their entries colliding.
    """

    def __init__(self, dim: int = 32, model_id: str = "hash-mock"):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")
        self._dim = dim
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(f"HashMockEmbedder only embeds str, got {type(t).__name__}")
            digest = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(digest, "little", signed=False)
            v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0.0:
                v = v / norm
            out.append(v)
        return out


class DictMockEmbedder(BaseEmbedder):
    """Mock embedder backed by an explicit ``{text: ndarray}`` map.

    Useful for tests that need controlled cosines — e.g. ``"warm"`` and
    ``"hot"`` mapped to a chosen near-collinear pair. Raises
    :class:`KeyError` on unknown text so that test schemas can never
    silently fall through to a default.

    Args:
        vectors: Mapping from text to a 1-D or 2-D ``np.ndarray``. The
            shape of the first registered vector pins :attr:`dim` (use
            ``None`` to indicate token-level / variable shape — in which
            case shape-sanity checks are skipped here; the metric layer
            is responsible for interpreting shapes).
        model_id: Cache-key namespace; defaults to ``"dict-mock"``.
    """

    def __init__(self, vectors: dict[str, np.ndarray], model_id: str = "dict-mock"):
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")
        if not vectors:
            raise ValueError("DictMockEmbedder requires at least one (text, vector) entry")
        sample = next(iter(vectors.values()))
        if not isinstance(sample, np.ndarray):
            raise TypeError("vectors must map str -> np.ndarray")
        self._vectors = {k: np.asarray(v) for k, v in vectors.items()}
        self._model_id = model_id
        # Pin dim from the first 1-D entry seen; leave None for 2-D.
        self._dim: int | None = None
        for v in self._vectors.values():
            if v.ndim == 1:
                self._dim = int(v.shape[0])
                break

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vectors[t] for t in texts]

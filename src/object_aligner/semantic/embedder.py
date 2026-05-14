"""Embedder protocol and abstract base.

The ``Embedder`` protocol is the lowest layer of the semantic-similarity
stack. Concrete implementations live alongside this module:

* :class:`object_aligner.semantic.HashMockEmbedder` /
  :class:`object_aligner.semantic.DictMockEmbedder` — deterministic mocks
  used by the test suite.
* :class:`object_aligner.semantic.OpenAIEmbedder` — production transport
  against any OpenAI-compatible embeddings endpoint (cloud OpenAI,
  ``llama-cpp`` on localhost, ``vllm``, etc.).

The cache layer (:mod:`object_aligner.semantic.cache`) wraps an
``Embedder`` and adds memoisation + batched miss resolution; the metric
layer (:mod:`object_aligner.semantic.metric`) consumes the cache and
produces an OA-shaped ``(gold, pred) -> float`` callable.

Shape convention
----------------
``embed_many`` returns one ``np.ndarray`` per input text. Sentence-level
embedders return **1-D** arrays of shape ``(dim,)``; token-level
embedders (e.g. for BERTScore) return **2-D** arrays of shape
``(n_tokens, dim)``. The protocol intentionally does not constrain
shape — the metric layer interprets it. The cache layer is
shape-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Minimal protocol for an embedding model.

    Concrete embedders should be safe to share across threads as long as
    their underlying transport (HTTP client, ONNX session, ...) is
    thread-safe — the cache layer in
    :mod:`object_aligner.semantic.cache` may issue batched calls from a
    single thread today but the protocol is forward-compatible.

    Attributes:
        model_id: Non-empty string uniquely identifying the embedding
            model **and any configuration that changes its output**
            (base URL, dimension override, prompt prefix). Included in
            the cache key so that swapping embedders never silently
            returns stale vectors. Example values:
            ``"openai/text-embedding-3-small/d=512"``,
            ``"localhost:8333::Qwen3-Embedding-4B::d=default"``,
            ``"hash-mock"``.
        dim: Per-embedding dimensionality. ``None`` for token-level
            embedders whose leading axis depends on the input.

    Methods:
        embed_many: Embed a list of strings in one batch call.
            Implementations should respect any backend batch limit by
            chunking internally; callers may pass arbitrarily long
            lists. Returns one ``np.ndarray`` per input in the same
            order.
        embed_one: Embed a single string. Default implementation
            delegates to ``embed_many`` so that backends that only
            implement true batching automatically work for single-text
            callers.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int | None: ...

    def embed_many(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_one(self, text: str) -> np.ndarray: ...


class BaseEmbedder:
    """Optional abstract base that supplies the ``embed_one`` default.

    Concrete embedders may either inherit from this class (and only
    implement ``embed_many``) or implement the :class:`Embedder`
    protocol directly. Both forms are accepted everywhere in the
    package.
    """

    @property
    def model_id(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def dim(self) -> int | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

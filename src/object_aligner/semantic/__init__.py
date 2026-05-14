"""Embedding-based semantic similarity for Object Aligner.

A three-layer stack — embedder, cache, metric — composed by dependency
injection and plugged into ``ObjectAligner`` via the standard
``custom_metrics`` mechanism. See :doc:`docs/semantic.md` for the full
chapter; the design discussion lives in
``research/opus47_semantic_search.md``.

Public surface (none of these are re-exported at the top-level
``object_aligner`` namespace — semantic is an opt-in feature):

* :class:`Embedder` — the protocol.
* :class:`BaseEmbedder` — optional abstract base that supplies the
  ``embed_one`` default.
* :class:`HashMockEmbedder`, :class:`DictMockEmbedder` — deterministic
  test backends.
* :class:`OpenAIEmbedder` — OpenAI-compatible HTTP transport (requires
  the ``semantic-openai`` extra at construction time).
* :class:`EmbeddingCache`, :class:`InMemoryEmbeddingCache`,
  :class:`SQLiteEmbeddingCache` — caching layer.
* :func:`cosine_similarity_metric`, :class:`CosineSimilarityMetric` —
  the OA-shaped ``(gold, pred) -> float`` callable.
* :func:`precompute` — schema-aware cache pre-warming helper.
"""

from object_aligner.semantic.embedder import BaseEmbedder, Embedder
from object_aligner.semantic.mock_embedder import DictMockEmbedder, HashMockEmbedder
from object_aligner.semantic.cache import (
    EmbeddingCache,
    InMemoryEmbeddingCache,
    SQLiteEmbeddingCache,
)
from object_aligner.semantic.openai_embedder import OpenAIEmbedder
from object_aligner.semantic.metric import (
    CosineSimilarityMetric,
    cosine_similarity_metric,
)
from object_aligner.semantic.precompute import precompute

__all__ = [
    "Embedder",
    "BaseEmbedder",
    "HashMockEmbedder",
    "DictMockEmbedder",
    "OpenAIEmbedder",
    "EmbeddingCache",
    "InMemoryEmbeddingCache",
    "SQLiteEmbeddingCache",
    "CosineSimilarityMetric",
    "cosine_similarity_metric",
    "precompute",
]

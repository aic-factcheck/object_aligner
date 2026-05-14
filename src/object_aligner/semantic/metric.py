"""Semantic similarity metrics, OA-shaped.

Layer 3 of the semantic stack: turns an
:class:`~object_aligner.semantic.EmbeddingCache` into a
``(gold: str, pred: str) -> float in [0, 1]`` callable that plugs into
:class:`object_aligner.ObjectAligner`'s
``custom_metrics={"string": {...}}`` registry.

v1 ships one metric — cosine similarity between sentence-level
embeddings, with a choice of sign convention for the
``[-1, 1] → [0, 1]`` rescaling. Future metrics (BERTScore, dot product
with no normalisation, asymmetric retrieval, ...) will live in this
module alongside.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from object_aligner.semantic.cache import EmbeddingCache


_VALID_SIGN_CONVENTIONS = ("clip", "affine")


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return ``v / ||v||`` with a tiny epsilon for safety.

    Most OpenAI-compatible servers return embeddings that are already
    L2-normalised; we defensively renormalise so the metric also works
    against backends that don't.
    """
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        return v / norm
    return v


def cosine_similarity_metric(
    cache: EmbeddingCache,
    *,
    sign_convention: str = "clip",
) -> Callable[[str, str], float]:
    """Build a cosine-similarity metric closure over an embedding cache.

    The returned callable matches OA's custom-metric contract: take a
    pair of strings, return a float in :math:`[0, 1]`. It carries two
    extra attributes:

    * ``.cache`` — back-reference to ``cache``. Used by
      :func:`object_aligner.semantic.precompute` to discover which
      strings in the schema flow through which cache and pre-warm them
      in a single batch call.
    * ``.kind`` — the literal string ``"cosine_similarity"``. A simple
      discriminator if you ever need to identify the metric type
      reflectively.

    Args:
        cache: An :class:`EmbeddingCache` wrapping the embedder you want
            to use. The same cache may be shared across multiple
            metrics (e.g. cosine and a future dot-product metric) so
            the embedding work is amortised.
        sign_convention: How to map raw cosine in :math:`[-1, 1]` to
            :math:`[0, 1]`:

            * ``"clip"`` *(default)* — :math:`s = \\max(0, \\cos)`.
              Negative cosines collapse to ``0``. Conservative and
              matches the common practice for modern embeddings, where
              negative cosines indicate genuinely unrelated text.
            * ``"affine"`` — :math:`s = (\\cos + 1) / 2`. Orthogonal
              vectors score ``0.5``; useful when you care about the
              full range and don't want negative cosines to flatline.

    Returns:
        A callable ``(gold: str, pred: str) -> float``.

    Raises:
        ValueError: If ``sign_convention`` is not one of the supported
            values.
    """
    if sign_convention not in _VALID_SIGN_CONVENTIONS:
        raise ValueError(
            f"sign_convention must be one of {_VALID_SIGN_CONVENTIONS!r}, "
            f"got {sign_convention!r}"
        )

    def metric(gold: str, pred: str) -> float:
        v_g, v_p = cache.get_many([gold, pred])
        v_g = _normalise(np.asarray(v_g, dtype=np.float64))
        v_p = _normalise(np.asarray(v_p, dtype=np.float64))
        cos = float(np.dot(v_g, v_p))
        if sign_convention == "clip":
            return max(0.0, min(1.0, cos))
        return 0.5 * (cos + 1.0)

    # Back-references for the pre-warmer to discover this metric's cache.
    metric.cache = cache  # type: ignore[attr-defined]
    metric.kind = "cosine_similarity"  # type: ignore[attr-defined]
    metric.sign_convention = sign_convention  # type: ignore[attr-defined]
    return metric


class CosineSimilarityMetric:
    """Class-shaped wrapper over :func:`cosine_similarity_metric`.

    Functionally identical to the factory; provided for callers who
    prefer instance-based composition. ``__call__`` delegates to the
    factory's closure; the ``.cache``, ``.kind``, and
    ``.sign_convention`` attributes are mirrored on the instance for
    parity with the function-shaped metric.

    Example:
        >>> metric = CosineSimilarityMetric(cache)
        >>> metric("hello", "hi")
        0.87...
    """

    def __init__(
        self,
        cache: EmbeddingCache,
        *,
        sign_convention: str = "clip",
    ):
        self._metric = cosine_similarity_metric(cache, sign_convention=sign_convention)
        self.cache = cache
        self.kind = "cosine_similarity"
        self.sign_convention = sign_convention

    def __call__(self, gold: str, pred: str) -> float:
        return self._metric(gold, pred)

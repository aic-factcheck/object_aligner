"""Schema-aware cache pre-warming.

When OA scores a single ``(gold, pred)`` pair through an embedding-based
metric, the metric layer calls ``cache.get_many([gold, pred])`` and
embeds whatever is missing — fine for one call, but inside a Hungarian
loop OA may invoke the same metric many times in sequence (one per
``(gold_i, pred_j)`` cell). Each individual invocation pays one HTTP
round trip per unique miss.

:func:`precompute` shifts that cost up front: walk the schema once,
identify every string path whose ``score`` resolves to a registered
metric that exposes a ``.cache`` attribute, collect the union of
strings at those paths across the input objects, group them by cache,
and call :meth:`EmbeddingCache.get_many` exactly once per cache. After
``precompute`` returns, every subsequent metric call hits the cache
without making a network request.

This module is a *consumer* of OA internals (it reads
``aligner._primitive_metrics`` and ``aligner.schema``) but never mutates
them — no monkey-patching, no side effects beyond filling caches.

Typical usage::

    precompute(aligner, gold, pred)
    aligner.metric(gold, pred)   # cache-warm; zero embedding round trips
"""

from __future__ import annotations

from typing import Any

from object_aligner.semantic.cache import EmbeddingCache


def _collect_cached_string_paths(
    schema: Any,
    metrics: dict[str, Any],
    path: tuple,
    out: list[tuple[tuple, EmbeddingCache]],
) -> None:
    """Walk the schema and append ``(path, cache)`` entries for every
    string node whose metric has a ``.cache`` attribute.

    ``path`` is a tuple of edges describing how to drill into a data
    object: integers select list items by position, strings select dict
    keys, and the sentinel ``"*"`` means "every item of this list" (used
    for ``items`` schemas where indices aren't statically known).
    """
    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")
    type_str = (
        schema_type
        if isinstance(schema_type, str)
        else (schema_type[0] if isinstance(schema_type, list) and schema_type else None)
    )

    if type_str == "string" or (
        isinstance(schema_type, list) and "string" in schema_type
    ):
        score = schema.get("score")
        if isinstance(score, str):
            metric = metrics.get(score)
            cache = getattr(metric, "cache", None) if metric is not None else None
            if isinstance(cache, EmbeddingCache):
                out.append((path, cache))

    if "properties" in schema and isinstance(schema["properties"], dict):
        for k, sub in schema["properties"].items():
            _collect_cached_string_paths(sub, metrics, path + (("property", k),), out)
    if "items" in schema:
        _collect_cached_string_paths(
            schema["items"], metrics, path + (("items",),), out
        )
    if "prefixItems" in schema and isinstance(schema["prefixItems"], list):
        for i, sub in enumerate(schema["prefixItems"]):
            _collect_cached_string_paths(
                sub, metrics, path + (("prefix", i),), out
            )


def _collect_strings_at_path(
    data: Any,
    path: tuple,
    cursor: int,
    out: list[str],
) -> None:
    """Collect every string in ``data`` reachable by the remaining edges
    of ``path`` starting at ``cursor``. Missing keys / indices skip
    silently — this is a best-effort gather, not validation.
    """
    if cursor == len(path):
        if isinstance(data, str):
            out.append(data)
        return

    edge = path[cursor]
    kind = edge[0]
    if kind == "property":
        if isinstance(data, dict):
            key = edge[1]
            if key in data:
                _collect_strings_at_path(data[key], path, cursor + 1, out)
        return
    if kind == "items":
        if isinstance(data, list):
            for item in data:
                _collect_strings_at_path(item, path, cursor + 1, out)
        return
    if kind == "prefix":
        if isinstance(data, list):
            i = edge[1]
            if i < len(data):
                _collect_strings_at_path(data[i], path, cursor + 1, out)
        return


def precompute(aligner, *objects: Any) -> dict[str, int]:
    """Pre-warm every embedding cache referenced by ``aligner``'s string
    metrics, using strings drawn from the supplied objects.

    Args:
        aligner: An :class:`object_aligner.ObjectAligner` whose
            ``custom_metrics["string"]`` may contain metrics produced by
            :func:`object_aligner.semantic.cosine_similarity_metric` (or
            any other metric whose callable carries a ``.cache``
            attribute pointing at an :class:`EmbeddingCache`).
        *objects: Any number of objects to scan for strings — typically
            the ``gold`` and ``pred`` that you are about to score, but
            any value is acceptable (lists of objects, batched gold,
            etc.).

    Returns:
        A dict mapping each ``cache.embedder.model_id`` to the number of
        unique strings sent to that cache (useful for logging /
        diagnostics — the actual number of *new* HTTP round trips is
        whatever subset of those strings was a cache miss).
    """
    string_metrics: dict[str, Any] = aligner._primitive_metrics.get("string", {})
    pairs: list[tuple[tuple, EmbeddingCache]] = []
    _collect_cached_string_paths(aligner.schema, string_metrics, (), pairs)
    if not pairs:
        return {}

    # Group by cache. Multiple paths can share one cache; collect their
    # strings together so we make one get_many call per cache.
    by_cache: dict[int, tuple[EmbeddingCache, set[str]]] = {}
    for path, cache in pairs:
        bucket = by_cache.setdefault(id(cache), (cache, set()))
        for obj in objects:
            collected: list[str] = []
            _collect_strings_at_path(obj, path, 0, collected)
            bucket[1].update(collected)

    report: dict[str, int] = {}
    for cache, texts in by_cache.values():
        if not texts:
            continue
        cache.get_many(list(texts))
        report[cache.embedder.model_id] = len(texts)
    return report

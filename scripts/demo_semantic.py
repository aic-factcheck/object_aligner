"""Manual end-to-end demo for the semantic-similarity feature.

Requires:
  * the `semantic-openai` extra installed (``pip install -e
    '.[semantic-openai]'``),
  * a running OpenAI-compatible embeddings server, by default
    `llama-cpp` started with ``--embedding --port 8333`` and serving
    the ``Qwen3-Embedding-4B`` model.

Run:
    uv run python scripts/demo_semantic.py

The script aligns two short invoice-shaped objects, prints the OA
score and the prompt-optimizer feedback string, and demonstrates the
SQLite cache by running the same alignment twice and reporting how
many embedding round-trips each pass made.
"""

from __future__ import annotations

import os
from pathlib import Path

from object_aligner import ObjectAligner
from object_aligner.semantic import (
    OpenAIEmbedder,
    SQLiteEmbeddingCache,
    cosine_similarity_metric,
    precompute,
)
from object_aligner.semantic.embedder import BaseEmbedder


class _CountingEmbedder(BaseEmbedder):
    """Wraps an embedder; counts how many strings are sent upstream."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.batches = 0

    @property
    def model_id(self):
        return self._inner.model_id

    @property
    def dim(self):
        return self._inner.dim

    def embed_many(self, texts):
        self.batches += 1
        self.calls += len(texts)
        return self._inner.embed_many(texts)


def main():
    cache_path = Path(os.environ.get("OA_DEMO_CACHE", "./.cache/oa-embeddings.sqlite"))
    base_url = os.environ.get("OA_DEMO_BASE_URL", "http://localhost:8333/v1")
    model = os.environ.get("OA_DEMO_MODEL", "Qwen3-Embedding-4B")

    inner = OpenAIEmbedder(model=model, base_url=base_url)
    embedder = _CountingEmbedder(inner)
    cache = SQLiteEmbeddingCache(embedder, cache_path)
    semantic = cosine_similarity_metric(cache)

    schema = {
        "type": "object",
        "properties": {
            "invoice_no": {"type": "string", "score": "exact"},
            "vendor":     {"type": "string", "score": "jaro"},
            "description": {"type": "string", "score": "semantic"},
            "line_items": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "sku":         {"type": "string", "score": "exact"},
                        "description": {"type": "string", "score": "semantic"},
                        "qty":         {"type": "integer"},
                        "price":       {"type": "number"},
                    },
                },
            },
        },
    }

    aligner = ObjectAligner(
        schema,
        custom_metrics={"string": {"semantic": semantic}},
        generate_feedback=True,
    )

    gold = {
        "invoice_no": "INV-001",
        "vendor": "Acme Inc.",
        "description": "Quarterly software-licensing renewal.",
        "line_items": [
            {"sku": "A1", "description": "Annual seat license",  "qty": 2, "price": 999.00},
            {"sku": "A2", "description": "Premium support add-on", "qty": 1, "price": 199.00},
        ],
    }

    pred = {
        "invoice_no": "INV-001",
        "vendor": "ACME Inc",
        "description": "Renewal of quarterly software licenses.",
        "line_items": [
            {"sku": "A2", "description": "Add-on for premium support tier", "qty": 1, "price": 199.00},
            {"sku": "A1", "description": "Per-seat annual licensing",        "qty": 2, "price": 999.00},
        ],
    }

    print(f"Embedding server: {base_url} (model={model})")
    print(f"Cache: {cache.path}\n")

    print("=" * 70)
    print("Run 1 — pre-warming + alignment.")
    print("=" * 70)
    before = (embedder.calls, embedder.batches)
    report = precompute(aligner, gold, pred)
    after_precompute = (embedder.calls, embedder.batches)
    result = aligner.metric(gold, pred)
    after_metric = (embedder.calls, embedder.batches)

    print(f"precompute report: {report}")
    print(f"  upstream calls before precompute: {before[0]} (batches {before[1]})")
    print(f"  upstream calls after precompute:  {after_precompute[0]} (batches {after_precompute[1]})")
    print(f"  upstream calls after metric:      {after_metric[0]} (batches {after_metric[1]})")
    print(f"\nscore: {result['score']:.4f}\n")
    print("feedback:")
    print(result.get("feedback", "<feedback disabled>"))

    print("\n" + "=" * 70)
    print("Run 2 — same inputs, cache should already be warm.")
    print("=" * 70)
    before2 = (embedder.calls, embedder.batches)
    result2 = aligner.metric(gold, pred)
    after2 = (embedder.calls, embedder.batches)
    print(f"  upstream calls during run 2:      {after2[0] - before2[0]} (batches {after2[1] - before2[1]})")
    print(f"  score: {result2['score']:.4f}")

    cache.close()


if __name__ == "__main__":
    main()

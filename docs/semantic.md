# 16. Semantic Similarity

[Docs](index.md) › Semantic Similarity

Object Aligner's built-in string metrics (`jaro`, `levenshtein`, ...)
operate on characters. They work well when the candidate string is a
*typo* or a *rephrasing of the same surface form*, but they cannot tell
that `"Revenue grew 12%."` and `"Earnings rose by twelve percent."`
mean roughly the same thing.

This chapter documents the **semantic similarity** layer: a
custom-metric implementation backed by an embedding model. Strings are
encoded into dense vectors via a configurable backend (OpenAI cloud,
`llama-cpp` on localhost, future local Transformers); the metric
returns the cosine similarity between the gold and candidate vectors,
mapped into $[0, 1]$. Caching, batching, and an OpenAI-compatible HTTP
transport are provided so users do not have to reinvent them.

The whole stack lives in the optional `object_aligner.semantic`
subpackage. It is **not** re-exported at the package root — semantic is
opt-in.

---

## Quickstart

Run a `llama-cpp` server with embedding support:

```bash
./llama-server --embedding --port 8333 -m /path/to/Qwen3-Embedding-4B.gguf
```

then install the optional dependency that pulls the OpenAI Python
client, configure the embedder, and register the metric:

```bash
pip install 'object-aligner[semantic-openai]'
```

```python
from object_aligner import ObjectAligner
from object_aligner.semantic import (
    OpenAIEmbedder, SQLiteEmbeddingCache, cosine_similarity_metric, precompute,
)

embedder = OpenAIEmbedder(
    model="Qwen3-Embedding-4B",
    base_url="http://localhost:8333/v1",
)
cache = SQLiteEmbeddingCache(embedder, "./.cache/oa-embeddings.sqlite")
semantic = cosine_similarity_metric(cache)

aligner = ObjectAligner(
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "score": "semantic"},
            "title":   {"type": "string", "score": "jaro"},
        },
    },
    custom_metrics={"string": {"semantic": semantic}},
)

gold = {"title": "Q3 2025", "summary": "Revenue grew 12%."}
pred = {"title": "Q3 2025", "summary": "Earnings rose by twelve percent."}

precompute(aligner, gold, pred)              # one batch HTTP call up front
print(aligner.metric(gold, pred))             # uses the cache, no more I/O
```

The first run hits the embedding server twice (once per string),
populates `./.cache/oa-embeddings.sqlite`, and returns a similarity
score that reflects semantic closeness. Every subsequent run with the
same texts is a pure database lookup.

---

## Shared setup for the examples

```python
import numpy as np
from object_aligner import ObjectAligner
from object_aligner.semantic import (
    HashMockEmbedder, DictMockEmbedder,
    InMemoryEmbeddingCache, SQLiteEmbeddingCache,
    cosine_similarity_metric, precompute,
)

schema = {
    "type": "object",
    "properties": {
        "title":  {"type": "string", "score": "jaro"},
        "body":   {"type": "string", "score": "semantic"},
    },
}

gold = {"title": "Q3 2025", "body": "Revenue grew 12%."}
pred = {"title": "Q3 2025", "body": "Earnings rose by twelve percent."}
```

`HashMockEmbedder` is the test-suite default; it is deterministic and
does not require a running server.

---

## The model

### Cosine and sign convention

Let $\mathbf{e}(t)$ denote the embedding of a string $t$. The metric
computes the cosine between the gold and candidate embeddings, both
defensively L2-normalised:

$$
\cos\bigl(\mathbf{e}(g),\, \mathbf{e}(p)\bigr)
\;=\;
\frac{\mathbf{e}(g) \cdot \mathbf{e}(p)}{\|\mathbf{e}(g)\|\,\|\mathbf{e}(p)\|}
\;\in\;[-1,\, 1].
$$

Object Aligner requires a score in $[0, 1]$. Two **sign conventions**
are supported:

| Convention | Formula | Effect |
|---|---|---|
| `"clip"` *(default)* | $s = \max(0,\, \cos)$ | Negative cosines collapse to 0 — conservative; matches the common-practice handling of modern embeddings. |
| `"affine"` | $s = \tfrac{1}{2}\bigl(\cos + 1\bigr)$ | Orthogonal vectors score $0.5$; the full $[-1, 1]$ range is preserved. |

Pick one at metric construction time:

```python
cosine_similarity_metric(cache, sign_convention="clip")    # default
cosine_similarity_metric(cache, sign_convention="affine")
```

### Cache-key namespacing

Cache entries are keyed by `(model_id, text)`. The embedder reports a
`model_id` that bakes in everything that affects the embedding —
typically the base URL host, the model name, and the
`dimensions` parameter. Swapping embedders never silently returns
stale vectors.

```python
OpenAIEmbedder(model="Qwen3", base_url="http://localhost:8333/v1").model_id
# 'localhost:8333::Qwen3::d=default'
```

### Pre-warming

Inside a Hungarian-aligned list, the metric is invoked once per
`(gold_i, pred_j)` cell. Each individual invocation embeds two strings,
which is suboptimal against a batch-aware backend.

`precompute(aligner, *objects)` walks the schema once, finds every
string path whose `score` resolves to a registered cache, collects the
union of strings across the input objects, and calls
`cache.get_many(...)` exactly once per cache. After it returns, every
subsequent metric call is a pure dict lookup.

```python
precompute(aligner, gold, pred)
# ⇒ {'localhost:8333::Qwen3::d=default': 2}
aligner.metric(gold, pred)            # no embedding round trips
```

`precompute` is idempotent — calling it twice with the same inputs
issues zero upstream calls the second time, because the cache is
already warm.

---

## Example 1 — Live `OpenAIEmbedder` against `llama-cpp`

(End-to-end usage; requires the `semantic-openai` extra and a running
embedding server.)

```python
from object_aligner.semantic import OpenAIEmbedder, SQLiteEmbeddingCache, cosine_similarity_metric

embedder = OpenAIEmbedder(
    model="Qwen3-Embedding-4B",
    base_url="http://localhost:8333/v1",
    max_batch_size=128,
)
cache = SQLiteEmbeddingCache(embedder, "./.cache/oa-embeddings.sqlite")
metric = cosine_similarity_metric(cache)

print(metric("Revenue grew 12%.", "Earnings rose by twelve percent."))   # 0.83 (illustrative)
print(metric("Revenue grew 12%.", "The cat sat on the mat."))            # 0.05 (illustrative)
```

The first call to each new pair sends one HTTP request to the server.
Subsequent calls hit the local SQLite cache.

---

## Example 2 — `HashMockEmbedder` for plumbing tests

```python
cache = InMemoryEmbeddingCache(HashMockEmbedder(dim=64))
metric = cosine_similarity_metric(cache)

assert metric("hello", "hello") > 0.99
assert metric("hello", "completely different") < 0.5     # near 0 in 64-d
```

The hash mock is **not** semantic-aware — it cannot tell that `"hello"`
and `"hi"` are related. It exists to exercise plumbing (caching,
batching, integration with `ObjectAligner`) without making network
calls.

---

## Example 3 — `DictMockEmbedder` for controlled cosines

```python
import numpy as np

v_warm = np.array([1.0, 0.0])
v_hot  = np.array([0.95, np.sqrt(1 - 0.95**2)])    # cos(warm, hot) = 0.95
v_cold = np.array([0.0, 1.0])

cache = InMemoryEmbeddingCache(DictMockEmbedder({
    "warm": v_warm, "hot": v_hot, "cold": v_cold,
}))
metric = cosine_similarity_metric(cache)

assert abs(metric("warm", "hot") - 0.95) < 1e-6
assert metric("warm", "cold") == 0.0
```

Use this for tests that need a specific cosine value, e.g. to drive a
hand-crafted Hungarian matrix.

---

## Example 4 — SQLite persistence

```python
embedder = HashMockEmbedder(dim=32)

# First run — populate the cache.
cache1 = SQLiteEmbeddingCache(embedder, "/tmp/demo.sqlite")
metric1 = cosine_similarity_metric(cache1)
metric1("apple", "orange")
cache1.close()

# Second run, fresh Python process style — reuse the same file.
cache2 = SQLiteEmbeddingCache(embedder, "/tmp/demo.sqlite")
assert "apple" in cache2 and "orange" in cache2
cache2.close()
```

SQLite is opened in WAL mode by default; multiple processes can read
and write the same file concurrently.

---

## Example 5 — Pre-warming amortises HTTP calls

This example uses the hash mock so it stays offline, but the batching
behaviour is identical against a real `OpenAIEmbedder`.

```python
from object_aligner.semantic.embedder import BaseEmbedder

class CountingEmbedder(BaseEmbedder):
    def __init__(self, inner):
        self._inner = inner
        self.calls = []
    @property
    def model_id(self): return self._inner.model_id
    @property
    def dim(self): return self._inner.dim
    def embed_many(self, texts):
        self.calls.append(list(texts))
        return self._inner.embed_many(texts)

counter = CountingEmbedder(HashMockEmbedder(dim=16))
cache = InMemoryEmbeddingCache(counter)
metric = cosine_similarity_metric(cache)

aligner = ObjectAligner(
    {"type": "array", "order": "align", "items": {"type": "string", "score": "semantic"}},
    custom_metrics={"string": {"semantic": metric}},
)

gold = ["red", "green", "blue"]
pred = ["blue", "yellow"]

precompute(aligner, gold, pred)
print(counter.calls)
# [['red', 'green', 'blue', 'yellow']]   — one batch call for the union.

aligner.metric(gold, pred)
print(counter.calls)
# Same single entry — no further upstream traffic.
```

---

## Example 6 — Mixing semantic and Jaro in one schema

`custom_metrics` accepts a normal dict; you can register the semantic
metric under whatever name you like and use it on the schema fields
that benefit from it, while leaving other fields to the cheaper Jaro
or `exact` built-ins.

```python
aligner = ObjectAligner(
    {
        "type": "object",
        "properties": {
            "sku":         {"type": "string", "score": "exact"},
            "category":    {"type": "string", "score": "jaro"},
            "description": {"type": "string", "score": "semantic"},
        },
    },
    custom_metrics={"string": {"semantic": metric}},
)
```

Only `description` flows through the embedding cache; `sku` and
`category` are scored by built-ins with no embedding cost.

---

## API reference

The new public surface lives at `object_aligner.semantic`:

| Name | Kind | Purpose |
|---|---|---|
| `Embedder` | Protocol | Minimal contract: `model_id`, `dim`, `embed_many`, `embed_one`. |
| `BaseEmbedder` | Abstract class | Optional base supplying the `embed_one` default. |
| `HashMockEmbedder` | Concrete | Deterministic BLAKE2b-seeded RNG; tests. |
| `DictMockEmbedder` | Concrete | Hand-curated mapping; tests with controlled cosines. |
| `OpenAIEmbedder` | Concrete | OpenAI-compatible HTTP transport. Requires the `semantic-openai` extra at construction. |
| `EmbeddingCache` | Abstract class | Shared `get_many` semantics. |
| `InMemoryEmbeddingCache` | Concrete | Dict-backed; no I/O. |
| `SQLiteEmbeddingCache` | Concrete | Persistent, WAL mode, stdlib-only. |
| `cosine_similarity_metric` | Factory | Returns the OA-shaped callable. |
| `CosineSimilarityMetric` | Class | Class-shaped wrapper over the factory. |
| `precompute` | Function | Schema-aware cache pre-warming. |

Signatures and parameter docs are also reachable from the generated
[API reference](api.md).

---

## Caveats

- **`OpenAIEmbedder` requires the optional `semantic-openai` extra.**
  The class can be imported without it; construction raises
  `ImportError` with the install command.
- **The hash mock is not semantic.** `HashMockEmbedder` is for tests.
  Use a real embedding model in production.
- **No async / parallel scoring.** OA's alignment loop is synchronous;
  for a remote API, every distinct string still pays one HTTP latency
  hop. Pre-warming amortises this into one batch call up front.
- **No GPU management.** If you plug in a local Transformers model,
  the device-placement / batch-sizing logic lives in *your* `Embedder`
  subclass, not in the library.
- **Token-level metrics (BERTScore etc.) are not in v1.** The protocol
  allows 2-D embeddings already; the metric implementation is the
  thing that needs to be written.
- **Concurrent writers on the SQLite cache.** WAL mode handles
  multiple processes safely. Multiple threads in the same process
  share a single connection (we open it with
  `check_same_thread=False`).

---

## See also

- [Primitive Types](primitives.md) — the broader custom-metric
  registration story.
- [Prompt-Optimizer Feedback](feedback.md) — how feedback ranks the
  ops emitted from semantic-metric scores.
- [Alignment Confidence](confidence.md) — Hungarian-derived
  confidence signal that complements semantic similarity.

[← Documentation home](index.md)

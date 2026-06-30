# 🧩 Object Aligner Documentation

**Object Aligner (OA)** computes a fine-grained similarity score in `[0, 1]` between two
structured data objects. It aligns a **gold** (reference) object with a **candidate** object
(the predicted output; the `pred` argument throughout the API) by recursively aligning their
trees, and awards partial credit at the granularity the schema declares — with optional
human-readable feedback and descriptions of the differences.

## ✨ Why Object Aligner?

When evaluating structured outputs — JSON objects, lists of extracted entities, configuration
dicts, knowledge graphs — exact match is too brittle and plain text similarity ignores
structure. A candidate value of `"ages"` vs. a gold value of `"age"` should not score 0%. OA
solves this by:

- 🌳 **Recursively** aligning objects of any nesting depth, with partial credit at every node.
- 🔤 Supporting **fuzzy matching** for strings (Jaro and friends) and numbers (inverse difference).
- 🧬 Offering opt-in **semantic similarity** for strings — embedding-based cosine scoring (with
  caching and batching) so paraphrases like `"car"` vs. `"automobile"` get partial credit
  beyond surface form (see [Semantic Similarity](semantic.md)).
- 🧮 Using the **Hungarian algorithm** for unordered collections and dictionary keys, and
  **sequence alignment** for ordered ones.
- 🕸️ Scoring cross-referenced records (graphs / hypergraphs) **up to identifier renumbering**.
- 🧭 Emitting **deterministic, ranked feedback** — *what* to change, not just *how well* you did —
  with no extra model call.

## 🚀 Quick start

```python
from object_aligner import ObjectAligner

schema = {"type": "string", "score": "jaro"}
aligner = ObjectAligner(schema)

print(aligner.metric("hello", "hallo"))          # {'score': 0.8666666666666667}
```

Score a nested object and ask for human-readable feedback in one call:

```python
result = aligner.metric(gold, pred, generate_feedback=True)
print(result["score"])
print(result["feedback"])   # ranked, prescriptive fix list — deterministic, no LLM
```

## 🗺️ Chapters

### 🌱 Foundations

| Chapter | Description |
|---------|-------------|
| [🌳 Concepts & Architecture](concepts.md) | Core abstractions: schemas, match types, scoring |
| [🔤 Primitive Types](primitives.md) | Strings, numbers, and booleans |
| [📚 Lists & Arrays](lists.md) | Fixed order, reordering, prefix items, and combinations |
| [🗂️ Dictionaries & Objects](dicts.md) | Key matching, value matching, importance weights |
| [🪆 Nesting & Complex Structures](nesting.md) | Deeply nested data, real-world composite examples |

### ⭐ Core capabilities

| Chapter | Description |
|---------|-------------|
| [🕸️ Referential Alignment](referential.md) | Matching graphs and multi-graphs whose ids are arbitrary handles |
| [🧭 Prompt-Optimizer Feedback](feedback.md) | Top-K ranked prescriptive feedback strings for optimizer reflection slots |
| [🎯 Per-Property Score Attribution](attribution.md) | Decompose the deficit into ranked per-path contributions |
| [🩹 Scored JSON-Patch Repair](repair.md) | Ranked structured repair operations with estimated score deltas |

### 🛠️ Using the score

| Chapter | Description |
|---------|-------------|
| [📊 The Metric Function](metric.md) | End-to-end evaluation with validation and optional description/feedback |
| [🗣️ Plain-English Description](describe.md) | Human-readable walk of the alignment tree (`generate_description`) |
| [🧾 Schema Reference](schema_reference.md) | Complete reference of all schema keywords |

### 🧩 Extensions & edge cases

| Chapter | Description |
|---------|-------------|
| [🈳 Null Handling](null_handling.md) | Per-field `nullScore` for asymmetric null/value mismatches |
| [🧬 Semantic Similarity](semantic.md) | Embedding-based string metrics with caching, batching, and OpenAI-compatible transport |

### 🧪 Experimental

| Chapter | Description |
|---------|-------------|
| [📈 Alignment Confidence](confidence.md) | Per-pair stability scores from the Hungarian matrix, with opt-in feedback/describe integration |

### 📖 Reference

| Chapter | Description |
|---------|-------------|
| [🔖 API Reference](api.md) | Generated reference of the public Python surface |

## 📦 Installation

Not on PyPI yet — install straight from GitHub:

```bash
pip install git+https://github.com/aic-factcheck/object_aligner.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/aic-factcheck/object_aligner.git
```

**Optional extras** (embedding-based semantic string similarity via an OpenAI-compatible API):

```bash
pip install "object-aligner[semantic-openai] @ git+https://github.com/aic-factcheck/object_aligner.git"
```

Requires **Python 3.13+**.

## 🧪 Testing and local development

Install project dependencies:

```bash
uv sync
```

Run the automated test suite:

```bash
uv run pytest
```

The repository includes a pytest suite under `tests/` covering utility helpers, primitive
scoring, list and dict alignment, API behavior, nested structures, referential alignment,
feedback, repair, attribution, documented examples, and edge cases.

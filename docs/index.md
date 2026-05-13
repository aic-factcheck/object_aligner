# Object Aligner Documentation

**Object Aligner** is a Python library for computing similarity scores between structured data objects. It aligns a "gold" (reference) object with a "predicted" object and produces a fine-grained similarity score in the range [0, 1], with optional human-readable descriptions of the differences.

## Why Object Aligner?

When evaluating structured outputs — JSON objects, lists of extracted entities, configuration dicts — simple exact-match metrics are too harsh. A predicted value of `"ages"` vs. a gold value of `"age"` should not score 0%. Object Aligner solves this by:

- **Recursively** aligning objects of any nesting depth
- Supporting **fuzzy matching** for strings (Jaro similarity) and numbers (inverse difference)
- Using the **Hungarian algorithm** to find optimal alignments for unordered lists and dictionary keys
- Producing **interpretable descriptions** when you ask for it

## Quick Start

```python
from object_aligner import ObjectAligner

schema = {"type": "string"}
aligner = ObjectAligner(schema)

result = aligner.metric(gold="hello", pred="hallo")
print(result["score"])  # 0.8667

result = aligner.metric(gold="hello", pred="hallo", generate_description=True)
print(result["description"])
```

## Migration note

The constructor now takes only the schema:

```python
aligner = ObjectAligner(schema)
```

The older `ObjectAligner("name", schema)` form and `get_name()` have been removed.

## Tutorial Chapters

| # | Chapter | Description |
|---|---------|-------------|
| 1 | [Concepts & Architecture](concepts.md) | Core abstractions: schemas, match types, scoring |
| 2 | [Primitive Types](primitives.md) | Strings, numbers, and booleans |
| 3 | [Lists & Arrays](lists.md) | Fixed order, reordering, prefix items, and combinations |
| 4 | [Dictionaries & Objects](dicts.md) | Key matching, value matching, importance weights |
| 5 | [Nesting & Complex Structures](nesting.md) | Deeply nested data, real-world composite examples |
| 6 | [The Metric Function](metric.md) | End-to-end evaluation with validation and optional description |
| 7 | [Plain-English Description](describe.md) | Human-readable walk of the alignment tree (`generate_description`) |
| 8 | [Referential Alignment](referential.md) | Matching graphs and multi-graphs whose ids are arbitrary handles |
| 9 | [Schema Reference](schema_reference.md) | Complete reference of all schema keywords |
| 10 | [Per-Property Score Attribution](attribution.md) | Decompose the deficit into ranked per-path contributions |
| 11 | [Scored JSON-Patch Repair](repair.md) | Ranked structured repair operations with estimated score deltas |
| 12 | [Prompt-Optimizer Feedback](feedback.md) | Top-K ranked prescriptive feedback strings for optimizer reflection slots |
| 13 | [API Reference](api.md) | Generated reference of the public Python surface |

## Installation

```bash
uv add object-aligner
```

Or with pip:

```bash
pip install object-aligner
```

## Testing and local development

Install project dependencies:

```bash
uv sync
```

Run the automated test suite:

```bash
uv run pytest
```

The repository now includes a pytest suite under `tests/` covering utility helpers, primitive scoring, list and dict alignment, API behavior, nested structures, documented examples, and edge cases.

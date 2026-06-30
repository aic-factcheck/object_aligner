# 📊 The Metric Function

[Docs](index.md) › The Metric Function

While `align()` gives you the raw match tree, `metric()` is the high-level API
designed for evaluation pipelines. It combines schema validation, alignment,
and optional human-readable description into a single call.

---

## Signatures

```python
aligner = ObjectAligner(schema)

result = aligner.metric(gold, pred, debug=False, generate_description=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gold` | any | *(required)* | Ground truth object |
| `pred` | any | *(required)* | Predicted object |
| `debug` | bool | `False` | If `True`, include a structured `"debug"` alignment tree made only of basic Python container/scalar types |
| `generate_description` | `bool \| "full" \| None` | `None` | Per-call override for the constructor default. See [`describe.md`](describe.md). |

`metric()` also accepts `generate_feedback` for the prompt-optimizer feedback
string — see [`feedback.md`](feedback.md).

### Return value

By default, `metric()` returns only a score:

```python
{"score": 0.87}
```

When `debug=True`, it also includes a structured alignment tree:

```python
{
    "score": 0.87,
    "debug": {
        "kind": "list",
        "score": 0.87,
        "children": [...],
    },
}
```

When `generate_description=True` (or the constructor default is set), the
returned dict additionally contains a `"description"` key with a plain-English
walk of the alignment. The full chapter, including the rendering model,
every template key, and worked examples, is
[`describe.md`](describe.md).

---

## Schema validation

Before alignment, `metric()` validates **both** `gold` and `pred` against the
schema. Validation uses a `jsonschema` validator instance built once at
construction and reused on every call:

1. **Gold must pass validation** — if it doesn't, a `ValidationError` is raised.
2. **If pred fails validation** — the function catches the `ValidationError`
   and returns immediately with score `0.0`.

### Validation failure shape

With description disabled:

```python
{"score": 0.0}
```

With description enabled, the description string carries the validation
error message — see [`describe.md`](describe.md#validation-errors).

### Example: Validation failure

```python
from object_aligner import ObjectAligner

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"],
    "keyScore": "exact"
}

aligner = ObjectAligner(schema)
result = aligner.metric(
    gold={"name": "Alice", "age": 30},
    pred={"name": "Alice"},
)
print(result)  # {"score": 0.0}
```

> **Note:** The schema you pass to `ObjectAligner` can use standard JSON
> Schema keywords (`required`, `additionalProperties`, `minItems`, etc.) for
> validation. Only the custom keywords (`score`, `threshold`, `order`, etc.)
> affect alignment behavior.

---

## End-to-end evaluation example

```python
from object_aligner import ObjectAligner

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "score": "jaro", "valueWeight": 2.0},
        "year": {"type": "integer", "score": "invdiff", "valueWeight": 1.0},
        "director": {"type": "string", "score": "jaro", "valueWeight": 1.5},
        "rating": {"type": "number", "score": "invdiff", "valueWeight": 1.0},
        "genres": {
            "type": "array",
            "items": {"type": "string", "score": "jaro", "threshold": 0.5},
            "order": "align",
            "ignoreExcess": True,
            "valueWeight": 1.0,
        },
    },
    "keyScore": "exact",
    "keyImportance": 0.0,
    "valueImportance": 1.0,
}

aligner = ObjectAligner(schema)

examples = [
    (
        {"title": "Inception", "year": 2010, "director": "Christopher Nolan", "rating": 8.8, "genres": ["Sci-Fi", "Thriller"]},
        {"title": "Inception", "year": 2010, "director": "Christopher Nolan", "rating": 8.8, "genres": ["Sci-Fi", "Thriller"]},
    ),
    (
        {"title": "The Matrix", "year": 1999, "director": "The Wachowskis", "rating": 8.7, "genres": ["Sci-Fi", "Action"]},
        {"title": "The Matrx", "year": 1999, "director": "Wachowski Sisters", "rating": 8.5, "genres": ["Sci-Fi", "Acton"]},
    ),
]

for i, (gold, pred) in enumerate(examples, start=1):
    result = aligner.metric(gold, pred)
    print(f"Example {i}: score={result['score']:.2f}")
    print()
```

---

## `align()` vs. `metric()`

| Feature | `align()` | `metric()` |
|---------|-----------|------------|
| Schema validation | Optional (`skip_validation`) | Always (pred failure → score 0) |
| Return type | Match object (`MatchItem`/`MatchList`/`MatchDict`) | `{"score": float}` (+ optional `description`, `feedback`, `debug` keys) |
| Use case | Programmatic access to alignment tree | Evaluation & reporting |

Use `align()` when you need to inspect or traverse the match tree
programmatically. Use `metric()` when you want a ready-to-log score, and
optionally a built-in description.

---

## See also

- [`describe.md`](describe.md) — the `generate_description` feature in full.
- [`feedback.md`](feedback.md) — `metric(generate_feedback=...)` and the
  prescriptive feedback string for prompt-optimizer reflection slots.
- [`attribution.md`](attribution.md) — per-path decomposition of the deficit
  $1 - \mathrm{s}$.
- [`api.md`](api.md) — generated API reference for `ObjectAligner.metric`.

[← Documentation home](index.md)

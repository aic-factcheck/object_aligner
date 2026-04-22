# 6. The Metric Function

While `align()` gives you the raw match tree, `metric()` is the high-level API designed for evaluation pipelines. It combines schema validation, alignment, and optional human-readable explanation into a single call.

---

## Signatures

```python
aligner = ObjectAligner(
    schema,
    generate_reasoning=False,
    reasoning_templates=None,
)

result = aligner.metric(gold, pred, debug=False, generate_reasoning=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gold` | any | *(required)* | Ground truth object |
| `pred` | any | *(required)* | Predicted object |
| `debug` | bool | `False` | *(currently unused)* |
| `generate_reasoning` | `bool | None` | `None` | Per-call override. `None` uses the constructor default. |

### Return value

By default, `metric()` returns only a score:

```python
{"score": 0.87}
```

When reasoning is enabled, it returns:

```python
{"score": 0.87, "reasoning": "..."}
```

---

## Enabling reasoning

### Constructor default

```python
from object_aligner import ObjectAligner

aligner = ObjectAligner(schema, generate_reasoning=True)
result = aligner.metric(gold, pred)
print(result["reasoning"])
```

### Per-call override

```python
aligner = ObjectAligner(schema)

aligner.metric(gold, pred)  # {"score": ...}
aligner.metric(gold, pred, generate_reasoning=True)  # includes reasoning
```

---

## Schema validation

Before alignment, `metric()` validates **both** `gold` and `pred` against the schema using `jsonschema.validate`:

1. **Gold must pass validation** — if it doesn't, a `ValidationError` is raised.
2. **If pred fails validation** — the function catches the `ValidationError` and returns immediately with score `0.0`.

### Validation failure shape

With reasoning disabled:

```python
{"score": 0.0}
```

With reasoning enabled:

```python
{
    "score": 0.0,
    "reasoning": 'JSON Schema validation failed for path="/some/path". Error message: ...'
}
```

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

result = aligner.metric(
    gold={"name": "Alice", "age": 30},
    pred={"name": "Alice"},
    generate_reasoning=True,
)
print(result["reasoning"])
```

> **Note:** The schema you pass to `ObjectAligner` can use standard JSON Schema keywords (`required`, `additionalProperties`, `minItems`, etc.) for validation. Only the custom keywords (`score`, `threshold`, `order`, etc.) affect alignment behavior.

---

## Reasoning format

The reasoning string explains the alignment in plain English and uses indentation to show nesting depth.

### Perfect match

```
The predicted output perfectly matches the gold.
```

### Imperfect match

```
The predicted output scores overall 72%, let us align the predicted output to the gold and analyze the differences:
  KEY = The predicted key "b" exactly matches the gold.
  VALUE = The predicted list scores 75%:
    The predicted value "3" exactly matches the gold.
    The predicted value "5" does not match the gold "4" (score=50%).
```

### Common built-in messages

- Item mismatch: `The predicted value "X" does not match the gold "Y" (score=67%).`
- List excess: `The predicted list item "X" is excessive, it was not in the gold.`
- List missing: `The predicted output misses the "X" list item from the gold.`
- Dict key mismatch: `KEY = The predicted key "X" does not match the gold "Y" (score=92%).`

---

## Template customization

You can override selected built-in reasoning strings with `reasoning_templates`.

```python
aligner = ObjectAligner(
    schema,
    generate_reasoning=True,
    reasoning_templates={
        "metric.perfect": "Perfect match.",
        "item.mismatch": 'Predicted "{pred}" vs gold "{gold}" ({score_pct}).\n',
    },
)
```

Overrides are **partial**: any template you do not provide keeps its default value.

Unknown template keys raise an error so typos are caught early.

### Stable template keys

- `metric.perfect`
- `metric.imperfect_intro`
- `item.match`
- `item.mismatch`
- `list.match`
- `list.mismatch`
- `list.excess`
- `list.missing`
- `dict.match`
- `dict.mismatch`
- `dict.key.match`
- `dict.key.mismatch`
- `dict.value.prefix`
- `validation.error`

### Common placeholders

Depending on the template, these placeholders are available:

- `indent`
- `gold`
- `pred`
- `score`
- `score_pct`
- `path`
- `message`

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
            "ignoreMissing": True,
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
    if result["score"] < 1.0:
        verbose = aligner.metric(gold, pred, generate_reasoning=True)
        print(verbose["reasoning"])
    print()
```

---

## `align()` vs. `metric()`

| Feature | `align()` | `metric()` |
|---------|-----------|------------|
| Schema validation | Optional (`skip_validation`) | Always (pred failure → score 0) |
| Return type | Match object (`MatchItem`/`MatchList`/`MatchDict`) | `{"score": float}` or `{"score": float, "reasoning": str}` |
| Use case | Programmatic access to alignment tree | Evaluation & reporting |

Use `align()` when you need to inspect or traverse the match tree programmatically. Use `metric()` when you want a ready-to-log score, and optionally a built-in explanation.

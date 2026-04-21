# 6. The Metric Function

While `align()` gives you the raw match tree, `metric()` is the high-level API designed for evaluation pipelines. It combines schema validation, alignment, and human-readable explanation into a single call.

---

## Signature

```python
result = aligner.metric(gold, pred, debug=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gold` | any | *(required)* | Ground truth object |
| `pred` | any | *(required)* | Predicted object |
| `debug` | bool | `False` | *(currently unused)* |

### Return value

A dictionary with two keys:

| Key | Type | Description |
|-----|------|-------------|
| `"score"` | float | Overall similarity score in [0, 1] |
| `"reasoning"` | string | Human-readable explanation of the alignment |

---

## Schema validation

Before alignment, `metric()` validates **both** `gold` and `pred` against the schema using `jsonschema.validate`:

1. **Gold must pass validation** — if it doesn't, an unhandled `ValidationError` is raised. The gold standard should always conform to the schema.
2. **If pred fails validation** — the function catches the `ValidationError` and returns immediately:

```python
{
    "reasoning": 'JSON Schema validation failed for path="/some/path". Error message: ...',
    "score": 0.0
}
```

This means **a structurally invalid prediction always scores 0.0**, regardless of how close its values might be.

### Example: Validation failure

```python
from object_aligner import ObjectAligner

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"}
    },
    "required": ["name", "age"],
    "keyScore": "exact"
}
aligner = ObjectAligner("person", schema)

# Missing required field → validation fails → score = 0.0
result = aligner.metric(
    gold={"name": "Alice", "age": 30},
    pred={"name": "Alice"}  # missing "age"
)
print(result["score"])      # 0.0
print(result["reasoning"])  # JSON Schema validation failed...
```

> **Note:** The schema you pass to `ObjectAligner` can use any standard JSON Schema keywords (`required`, `additionalProperties`, `minItems`, etc.) for validation purposes. Only the custom keywords (like `score`, `threshold`, `order`, etc.) affect alignment behavior.

---

## Reasoning format

The reasoning string explains the alignment in plain English. It uses indentation to show nesting depth.

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

### List-specific messages

For list mismatches, the reasoning distinguishes between:

- **Excessive predicted item**: `The predicted list item "X" is excessive, it was not in the gold.`
- **Missing gold item**: `The predicted output misses the "X" list item from the gold.`
- **Mismatched item**: `The predicted value "X" does not match the gold "Y" (score=67%).`
- **Matched item**: `The predicted value "X" exactly matches the gold.`

### Dict-specific messages

For dict key-value pairs:

- **Key mismatch**: `KEY = The predicted key "X" does not match the gold "Y" (score=92%).`
- **Key match**: `KEY = The predicted key "X" exactly matches the gold.`
- Followed by `VALUE = ` and the recursive reasoning for the value.

---

## End-to-end evaluation pipeline example

Here's how you might use `metric()` in a batch evaluation:

```python
from object_aligner import ObjectAligner

schema = {
    "type": "object",
    "properties": {
        "title":       {"type": "string",  "score": "jaro",   "valueWeight": 2.0},
        "year":        {"type": "integer", "score": "invdiff", "valueWeight": 1.0},
        "director":    {"type": "string",  "score": "jaro",   "valueWeight": 1.5},
        "rating":      {"type": "number",  "score": "invdiff", "valueWeight": 1.0},
        "genres": {
            "type": "array",
            "items": {"type": "string", "score": "jaro", "threshold": 0.5},
            "order": "align",
            "ignoreExcess": True,
            "ignoreMissing": True,
            "valueWeight": 1.0
        }
    },
    "keyScore": "exact",
    "keyImportance": 0.0,
    "valueImportance": 1.0
}

aligner = ObjectAligner("movie-eval", schema)

# Simulated LLM outputs
examples = [
    (
        {"title": "Inception", "year": 2010, "director": "Christopher Nolan", "rating": 8.8, "genres": ["Sci-Fi", "Thriller"]},
        {"title": "Inception", "year": 2010, "director": "Christopher Nolan", "rating": 8.8, "genres": ["Sci-Fi", "Thriller"]}
    ),
    (
        {"title": "The Matrix", "year": 1999, "director": "The Wachowskis", "rating": 8.7, "genres": ["Sci-Fi", "Action"]},
        {"title": "The Matrx", "year": 1999, "director": "Wachowski Sisters", "rating": 8.5, "genres": ["Sci-Fi", "Acton"]}
    ),
    (
        {"title": "Parasite", "year": 2019, "director": "Bong Joon-ho", "rating": 8.5, "genres": ["Thriller", "Comedy", "Drama"]},
        {"title": "Parasite", "year": 2020, "director": "Bong Joon-ho", "rating": 8.5, "genres": ["Thriller", "Drama"]}
    ),
]

scores = []
for i, (gold, pred) in enumerate(examples):
    result = aligner.metric(gold, pred)
    scores.append(result["score"])
    print(f"Example {i+1}: score={result['score']:.2f}")
    if result["score"] < 1.0:
        print(result["reasoning"])
    print()

print(f"Mean score: {sum(scores)/len(scores):.2f}")
```

---

## align() vs. metric()

| Feature | `align()` | `metric()` |
|---------|-----------|------------|
| Schema validation | Optional (`skip_validation`) | Always (pred failure → score 0) |
| Return type | Match object (`MatchItem`/`MatchList`/`MatchDict`) | `{"score": float, "reasoning": str}` |
| Use case | Programmatic access to alignment tree | Evaluation & reporting |

Use `align()` when you need to inspect or traverse the match tree programmatically. Use `metric()` when you want a ready-to-log score and explanation.

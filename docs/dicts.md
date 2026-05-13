# 4. Dictionaries & Objects

[Docs](index.md) › Dictionaries & Objects

Object Aligner matches dictionaries by **aligning keys first**, then recursively aligning the corresponding values. The key alignment uses the Hungarian algorithm to find the best pairing between gold keys and predicted keys — just like reorder alignment for lists.

The result is a `MatchDict` with a `score` and a `children` dict that maps **key-level** `MatchItem`s to **value-level** match objects.
As a simple edge case, an empty dict aligned with an empty dict scores `1.0`.

---

## Key matching

Keys are compared using one of two strategies:

### Exact key matching (`"keyScore": "exact"`)

Use when keys must be identical — e.g., API responses with well-defined field names.

```python
from object_aligner import ObjectAligner

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer", "score": "invdiff"}
    },
    "keyScore": "exact"
}
aligner = ObjectAligner(schema, generate_description=True)

gold = {"name": "Alice", "age": 30}
pred = {"name": "Alicia", "age": 29}
result = aligner.metric(gold, pred)
print(f"Score: {result['score']:.2f}")
```

With exact key matching, `"name"` pairs with `"name"`, `"age"` pairs with `"age"`. No fuzzy key matching occurs.

### Fuzzy key matching (`"keyScore": "jaro"`) — *default*

Use when predicted keys might have typos — e.g., OCR output or informal data entry.

```python
gold = {"weight": 90, "name": "John", "age": 24}
pred = {"name": "Johny", "ages": 23, "title": "Mr."}

schema = {
    "type": "object",
    "properties": {
        "weight": {"type": "integer", "valueWeight": 1.0},
        "name":   {"type": "string", "score": "jaro", "valueWeight": 1.0},
        "age":    {"type": "integer", "valueWeight": 1.0}
    },
    "keyScore": "jaro",
    "keyThreshold": 0.5,
    "keyImportance": 1.0,
    "valueImportance": 1.0
}
aligner = ObjectAligner(schema, generate_description=True)
result = aligner.metric(gold, pred)
print(result["description"])
```

Output:
```
The predicted output scores overall 42%, let us align...
  KEY = The predicted key "ages" does not match the gold "age" (score=92%).
  VALUE = The predicted value "23" does not match the gold "24" (score=50%).
```

Here the Hungarian algorithm matched `"ages"` → `"age"` (Jaro similarity 92%), even though they're not identical. The key `"title"` in the prediction has no good gold match, and `"weight"` in the gold has no good prediction match.

### keyThreshold

Just like for primitives, `keyThreshold` zeros out key similarity scores below the threshold. This prevents spurious key pairings:

```python
schema = {
    "type": "object",
    "keyScore": "jaro",
    "keyThreshold": 0.7,  # only fairly similar keys will be paired
    ...
}
```

---

## Value matching

Once keys are paired, values are aligned **recursively** according to the `properties` schema. Each property can have its own type and scoring configuration.

### valueWeight

Different properties can have different importance via `valueWeight` (default 1.0). The value score is a weighted average:

```python
schema = {
    "type": "object",
    "properties": {
        "id":   {"type": "integer", "score": "exact", "valueWeight": 3.0},  # ID is 3x more important
        "name": {"type": "string",  "score": "jaro",  "valueWeight": 1.0},
        "city": {"type": "string",  "score": "jaro",  "valueWeight": 1.0}
    },
    "keyScore": "exact",
    "keyImportance": 1.0,
    "valueImportance": 1.0
}
```

With this schema, a mismatch on `"id"` hurts the score three times as much as a mismatch on `"name"` or `"city"`.

---

## Importance balancing: key vs. value

The final dictionary score combines the key score and the value score:

```
dictScore = (keyImportance * keysScore + valueImportance * valuesScore) / (keyImportance + valueImportance)
```

| Setting | Effect |
|---------|--------|
| `keyImportance=0, valueImportance=1` | Only care about values; ignore key quality |
| `keyImportance=1, valueImportance=0` | Only care about key matching; ignore values |
| `keyImportance=1, valueImportance=1` (default) | Equal weight to both |

### Example: Only caring about values

When keys are well-structured and you just want to evaluate whether the values are correct:

```python
gold = {"a": [1, 2], "b": [3, 4]}
pred = {"a": [1, 2], "b": [3, 5]}

schema = {
    "type": "object",
    "properties": {
        "a": {"type": "array", "items": {"type": "integer"}, "valueWeight": 1.0},
        "b": {"type": "array", "items": {"type": "integer"}, "valueWeight": 1.0}
    },
    "keyScore": "exact",
    "keyImportance": 0.0,     # ignore key quality
    "valueImportance": 1.0    # only care about values
}
aligner = ObjectAligner(schema, generate_description=True)
result = aligner.metric(gold, pred)
print(result["description"])
```

Output:
```
The predicted output scores overall 88%...
  KEY = The predicted key "b" exactly matches the gold.
  VALUE = The predicted list scores 75%:
    The predicted value "3" exactly matches the gold.
    The predicted value "5" does not match the gold "4" (score=50%).
```

---

## Important caveat: key-value type consistency

Because keys are matched *before* values, and the value schema is looked up from the **gold key**'s property definition, there's a constraint:

> **The types of aligned gold and predicted values must match.**

If a gold key `"age"` maps to an integer 24, but the predicted key `"ages"` maps to a string `"twenty-three"`, the aligner will raise a `TypeError`:

```
TypeError: dict value types differ for key 'age': int vs str
```

This is by design — the schema for a property defines one type, and both the gold and predicted values under that key must conform to it.

### Softer behavior under `skip_validation=True`

When you call `align(gold, pred, skip_validation=True)` (or use `align` directly without validation), the type-mismatch raise is suppressed and the pair is scored as `0` instead. This lets evaluation pipelines tolerate occasional schema-non-conforming predictions without crashing. The `metric()` method validates `pred` against the schema itself and returns `{"score": 0.0}` on validation failure, so the soft-zero branch only surfaces through direct `align(..., skip_validation=True)` use.

---

## Extra and missing keys

When the Hungarian algorithm cannot pair a gold key with any predicted key (similarity below `keyThreshold`), that key appears as unaligned:

- **Missing key**: gold key with no prediction → value scored as 0.0
- **Extra key**: predicted key with no gold match → value scored as 0.0

These contribute negatively to both the key score and the value score.

---

## Schema keywords summary

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | Must be `"object"` |
| `properties` | object | *(required)* | Per-key schemas (JSON Schema style) |
| `keyScore` | string | `"jaro"` | `"jaro"` or `"exact"` — how to compare keys |
| `keyThreshold` | float | `0.0` | Minimum key similarity to form a pairing |
| `keyImportance` | float | `1.0` | Weight of key score in the final dict score |
| `valueImportance` | float | `1.0` | Weight of value score in the final dict score |
| *(in properties)* `valueWeight` | float | `1.0` | Relative weight of this property's value |

---

## See also

- [`lists.md`](lists.md) — the sister array type.
- [`nesting.md`](nesting.md) — real-world composite examples.
- [`referential.md`](referential.md) — using `idScope` / `ref` inside dicts.
- [`schema_reference.md`](schema_reference.md) — every supported keyword.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

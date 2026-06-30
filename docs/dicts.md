# 🗂️ Dictionaries & Objects

[Docs](index.md) › Dictionaries & Objects

Object Aligner matches dictionaries (*maps*, in the paper's data model) by **aligning keys first**, then recursively aligning the corresponding values. The key alignment uses the Hungarian algorithm to find the best pairing between gold keys and candidate keys — just like reorder alignment for lists. As the paper puts it, keys fix the correspondence and the values then grade it: the values are not consulted when choosing which keys pair.

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

Use when candidate keys might have typos — e.g., OCR output or informal data entry.

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
The predicted output scores overall 42%, let us align the predicted output to the gold and analyze the differences:
The predicted dictionary scores 42%:
  KEY = The predicted key "title" does not match the gold "None" (score=0%).
  VALUE = The predicted value "Mr." does not match the gold "None" (score=0%).

  KEY = The predicted key "None" does not match the gold "weight" (score=0%).
  VALUE = The predicted value "None" does not match the gold "90" (score=0%).

  KEY = The predicted key "name" exactly matches the gold.
  VALUE = The predicted value "Johny" does not match the gold "John" (score=93%).

  KEY = The predicted key "ages" does not match the gold "age" (score=92%).
  VALUE = The predicted value "23" does not match the gold "24" (score=50%).
```

Here the Hungarian algorithm matched `"ages"` → `"age"` (Jaro similarity 92%), even though they're not identical. The key `"title"` in the candidate has no good gold match, and `"weight"` in the gold has no good candidate match — those show up as pairings against `None`.

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
    "keyScore": "exact"
}
```

With this schema, a mismatch on `"id"` hurts the score three times as much as a mismatch on `"name"` or `"city"`.

---

## Key importance

The final dictionary score combines the key score and the value score:

$$
\mathrm{s} = \frac{w_K \cdot s_{\mathrm{key}} + w_V \cdot s_{\mathrm{val}}}{w_K + w_V}
$$

where $w_K$ is `keyImportance` and $w_V$ is `valueImportance`, $s_{\mathrm{key}}$
is the aggregate key score and $s_{\mathrm{val}}$ the aggregate value score (the
paper's Eq. for maps).

**Default: `keyImportance=0`, `valueImportance=1`.** Most evaluation
pipelines today have a *fixed schema* — the model fills slots in a
JSON shape the user already chose. In that setting the keys are
*scaffolding*, not data: a free `keysScore = 1.0` term would just pad
the average and compress the dynamic range that should distinguish
good from bad candidates. Set `keyImportance = 0` (the default) to
score only values.

Override the default to `keyImportance ≥ 1` when **the model chose the
keys**:

* **Open-vocabulary extraction** — OCR forms, schema discovery, free-form
  key-value attribute extraction. The choice between `"first_name"` /
  `"firstName"` / `"name"` is part of correctness.
* **Dicts used as maps, not records** — `Map<string, V>` rather than
  `Record<known-fields, V>`. Glossaries `{"term": "definition"}`, entity
  tables keyed by entity name, tag-to-confidence, label-to-count,
  feature-flag-to-config. Key-match precision/recall is half the signal.
* **Polymorphic / discriminated-union objects** — when the *presence* of
  a key carries type information.
* **Meta-evaluation** — aligning two schemas or configs, where the keys
  *are* the data being compared.

| Setting | Effect |
|---------|--------|
| `keyImportance=0, valueImportance=1` (default) | Score only values — fixed-schema evaluation |
| `keyImportance=1, valueImportance=1` | Equal weight on keys and values — open-vocabulary extraction |
| `keyImportance=1, valueImportance=0` | Score only key matching; ignore values |

### Example: open-vocabulary key matching

When the model also chooses the keys (e.g. OCR'd form fields), set
`keyImportance` so that picking the wrong key is penalised:

```python
gold = {"first_name": "Alice", "age": 30}
pred = {"firstname": "Alice", "ages": 30}   # both keys slightly off

schema = {
    "type": "object",
    "properties": {
        "first_name": {"type": "string"},
        "age":        {"type": "integer", "score": "invdiff"}
    },
    "keyScore": "jaro",
    "keyImportance": 1.0,
    "valueImportance": 1.0
}
aligner = ObjectAligner(schema)
print(aligner.metric(gold, pred)["score"])
```

Here the dict score blends key-jaro similarities (`first_name`↔`firstname`,
`age`↔`ages`) with the perfect value matches. With the default
`keyImportance = 0` the same comparison would score $1.0$ — the model
got the *values* right and the keys are treated as scaffolding.

---

## Important caveat: key-value type consistency

Because keys are matched *before* values, and the value schema is looked up from the **gold key**'s property definition, aligned gold and candidate values are expected to share one type. When they don't — typically because fuzzy key matching paired a gold key with an unrelated candidate key, or the schema declares a union type — the pair is scored `0.0` in place:

```python
from object_aligner import ObjectAligner

aligner = ObjectAligner({"type": "object", "properties": {"date": {"type": "string"}}})
# jaro("date", "rate") pairs the keys, but str vs int can't be compared:
aligner.metric({"date": "2024"}, {"rate": 5})   # {'score': 0.0} — no exception
aligner.align({"date": "2024"}, {"rate": 5})    # same: the value pair scores 0.0
```

The soft-zero rule applies uniformly: `align()`, `metric()`, and the `skip_validation=True` variants all agree on the same inputs. The exception is `None`: when exactly one side of a pair is `None`, the pair routes through null-aware scoring and the property's `nullScore` (see [`null_handling.md`](null_handling.md)).

---

## Extra and missing keys

When the Hungarian algorithm cannot pair a gold key with any candidate key (similarity below `keyThreshold`), that key appears as unaligned:

- **Missing key**: gold key with no candidate → value scored as 0.0
- **Extra key**: candidate key with no gold match → value scored as 0.0

These contribute negatively to both the key score and the value score.

### Keys not declared in `properties`

JSON Schema is open-world by default (`additionalProperties` is `true` unless you say otherwise), so a *gold* object can validate while carrying keys the schema never declares. The aligner has no schema node to score such a pair against: it scores `0.0` at weight `1.0` and emits a `UserWarning` naming the key. If your data is closed-world — and for scoring it almost always should be — declare every key in `properties` and set `"additionalProperties": false` so validation catches strays up front.

---

## Schema keywords summary

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | Must be `"object"` |
| `properties` | object | *(required)* | Per-key schemas (JSON Schema style) |
| `keyScore` | string | `"jaro"` | `"jaro"` or `"exact"` — how to compare keys |
| `keyThreshold` | float | `0.0` | Minimum key similarity to form a pairing |
| `keyImportance` | float | `0.0` | Weight of key score in the final dict score. Override to `≥ 1` when the model chooses the keys (see [Key importance](#key-importance)). |
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

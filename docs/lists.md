# 3. Lists & Arrays

[Docs](index.md) › Lists & Arrays

Lists are the most feature-rich type in Object Aligner. They support three alignment strategies — **prefix**, **fixed order**, and **reorder (Hungarian)** — and they can be combined.

All list results are wrapped in a `MatchList`, which contains an overall `score` and a list of `children` (each being a `MatchItem`, `MatchList`, or `MatchDict`).

---

## Fixed-order alignment (`"order": "fixed"`)

Elements are compared positionally, left to right. This uses a **dynamic programming** algorithm (similar to sequence alignment) that finds the optimal pairing while allowing gaps (skipped elements scored as 0).

This is the **default** ordering when `"items"` is present and no `"order"` key is specified.

### Example: Comparing ordered sequences

Imagine evaluating a student's answers against an answer key where order matters:

```python
from object_aligner import ObjectAligner

schema = {
    "type": "array",
    "items": {"type": "integer", "score": "exact"},
    "order": "fixed"
}
aligner = ObjectAligner(schema, generate_description=True)

# Student got 2nd and 3rd right, missed the 1st
gold = [42, 7, 13]
pred = [99, 7, 13]
result = aligner.metric(gold, pred)
print(result["description"])
```

Output:
```
The predicted output scores overall 67%, let us align...
  The predicted value "99" does not match the gold "42" (score=0%).
  The predicted value "7" exactly matches the gold.
  The predicted value "13" exactly matches the gold.
```

### Different-length lists with fixed order

When lists have different lengths, the DP algorithm inserts gaps:

```python
gold = [1, 2, 4]
pred = [2, 3]
result = aligner.metric(gold, pred)
print(result["description"])
```

Output:
```
The predicted output scores overall 25%, let us align...
The predicted list scores 25%:
  The predicted output misses the "1" list item from the gold.
  The predicted value "2" exactly matches the gold.
  The predicted output misses the "4" list item from the gold.
  The predicted list item "3" is excessive, it was not in the gold.
```

---

## Reorder alignment (`"order": "align"`)

When list order doesn't matter (e.g., a set of extracted entities), use the Hungarian algorithm to find the **best pairing** between gold and pred items, maximizing total similarity.

### Example: Extracting a list of skills

```python
schema = {
    "type": "array",
    "items": {"type": "string", "score": "jaro", "threshold": 0.5},
    "order": "align"
}
aligner = ObjectAligner(schema, generate_description=True)

gold = ["Python", "JavaScript", "SQL"]
pred = ["Pythn", "SQL", "JavaScrypt"]
result = aligner.metric(gold, pred)
print(f"Score: {result['score']:.2f}")
```

Even though the order is shuffled and there are typos, the Hungarian algorithm pairs each prediction with its best-matching gold item.

### Extra and missing items

Predicted items with no gold match (and vice versa) are reported:

```python
gold = ["weight", "name", "age"]
pred = ["name", "ages", "title"]

schema = {
    "type": "array",
    "items": {"type": "string", "score": "jaro", "threshold": 0.5},
    "order": "align"
}
aligner = ObjectAligner(schema, generate_description=True)
result = aligner.metric(gold, pred)
print(result["description"])
```

Output:
```
The predicted output scores overall 48%, let us align...
  The predicted list item "title" is excessive, it was not in the gold.
  The predicted output misses the "weight" list item from the gold.
  The predicted value "name" exactly matches the gold.
  The predicted value "ages" does not match the gold "age" (score=92%).
```

### ignoreExcess / ignoreMissing

By default, extra predictions and missing gold items count as mismatches (scored 0). You can change this:

- **`ignoreExcess: true`** — Extra predicted items don't penalize the score (they're simply ignored in normalization).
- **`ignoreMissing: true`** — Missing gold items don't penalize the score.

```python
schema = {
    "type": "array",
    "items": {"type": "string", "score": "jaro"},
    "order": "align",
    "ignoreExcess": True
}
```

Use `ignoreExcess` when you want to be lenient about over-prediction (e.g., the model found extra entities that aren't wrong, just not in the gold standard). Use `ignoreMissing` when partial extraction is acceptable.

The two flags are **mutually exclusive**: setting both on the same array raises `ValueError` at construction. With both flags the score would be the mean over successfully paired items only, which rewards omitting hard items — a strictly closer prediction could score lower than one that simply leaves items out.

The flags also cover the degenerate empty-side cases: with `ignoreExcess: true`, `gold=[]` vs `pred=["junk"]` scores `1.0` (every pred item is excess, all ignored), and symmetrically `ignoreMissing: true` makes `gold=["a"]` vs `pred=[]` score `1.0`.

---

## Prefix items (`prefixItems`)

Some arrays have a fixed positional header followed by variable-length tail. Think of a CSV-like record: the first few fields have known positions, but the rest is a sequence.

`prefixItems` defines schemas for the positional prefix. They are aligned one-to-one by index and can have **individual weights** via `prefixWeights`.

### Example: Transport records

Each record has a transport mode (string), a count (integer), and then zero or more destination names:

```python
schema = {
    "type": "array",
    "prefixItems": [
        {"type": "string"},   # transport mode
        {"type": "integer"},  # count
    ],
    "prefixWeights": [1, 1],  # equal weight for both prefix slots
    "items": {"type": "string"}  # destinations (variable length)
}
aligner = ObjectAligner(schema, generate_description=True)

gold = ["car", 5, "airport", "downtown"]
pred = ["cat", 5, "plane"]  # typo in mode, missing one destination
result = aligner.metric(gold, pred)
print(result["description"])
```

The prefix `[mode, count]` is compared positionally with weighted averaging. The tail `[destinations...]` is compared using whatever `"order"` is specified (default: `"fixed"`).

A prefix position present on only one side scores `0.0` at its full weight. A position absent from **both** sides is *vacuous*: it is excluded from the weight normalization entirely (the match tree still carries a sentinel child with `kind="absent"` for it), so a prediction identical to the gold always scores `1.0` even when both are shorter than `prefixItems`.

---

## Combining prefixItems + items

When both `prefixItems` and `items` are present, you must also specify:

- **`prefixImportance`** (float) — weight of the prefix portion
- **`restImportance`** (float) — weight of the tail portion

The final list score is: `(prefixImportance * prefixScore + restImportance * restScore) / (prefixImportance + restImportance)`

### Example: Mixed record with importance weighting

```python
gold = [["car", 5, "airplane"], ["bus", 3, "ship"]]
pred = [["cat", 5, "plane"], ["bus", 3]]

schema = {
    "type": "array",
    "items": {
        "type": "array",
        "prefixItems": [
            {"type": "string"},   # vehicle name
            {"type": "integer"},  # quantity
        ],
        "prefixWeights": [1, 1],
        "items": {"type": "string"},  # additional tags
        "prefixImportance": 2.0,      # prefix is twice as important
        "restImportance": 1.0,
    },
}
aligner = ObjectAligner(schema, generate_description=True)
result = aligner.metric(gold, pred)
print(f"Score: {result['score']:.2f}")
print(result["description"])
```

Here, the vehicle name and quantity (prefix) contribute twice as much to the score as the destination tags (tail). Even though `"cat"` vs `"car"` is a fuzzy mismatch, the high weight on the prefix helps maintain a reasonable overall score.

---

## List normalization

The number of items used for score normalization (the denominator) depends on `ignoreExcess` and `ignoreMissing`:

| Setting | Normalization |
|---------|--------------|
| Both `false` (default) | Count all aligned pairs (including gaps) |
| `ignoreExcess: true` | Exclude rows where gold is `None` |
| `ignoreMissing: true` | Exclude rows where pred is `None` |

Setting both flags on the same array raises `ValueError` at construction (see above). When the denominator ends up `0` — every row was excluded by the active flag, or both lists are empty — the match is vacuous and scores `1.0`.

---

## Schema keywords summary

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | Must be `"array"` |
| `items` | object | — | Schema for all/remaining items |
| `prefixItems` | list | — | Per-position schemas for the fixed prefix |
| `prefixWeights` | list | all 1s | Weights for each prefix position |
| `order` | string | `"fixed"` | `"fixed"` (DP) or `"align"` (Hungarian) |
| `ignoreExcess` | bool | `false` | Don't penalize extra predicted items (mutually exclusive with `ignoreMissing`) |
| `ignoreMissing` | bool | `false` | Don't penalize missing gold items (mutually exclusive with `ignoreExcess`) |
| `prefixImportance` | float | — | Weight for prefix score (required if both `prefixItems` and `items` present) |
| `restImportance` | float | — | Weight for tail score (required if both `prefixItems` and `items` present) |

---

## Caveats

- **Items beyond `prefixItems` are silently unscored when no `items` schema is present.** A schema with only `prefixItems` slices both lists to the prefix length: gold or pred content past the prefix never enters the score (extra predicted junk there is free). Declare an `items` schema if the tail should be graded.
- **Prefix positions ignore `ignoreExcess` / `ignoreMissing`.** Those flags only affect the `items` part; a one-sided prefix position always costs its full weight.
- **Both-absent prefix positions are vacuous.** They carry zero weight (so identity holds) and appear in the match tree as `kind="absent"` sentinel children.

---

## See also

- [`primitives.md`](primitives.md) — leaf scoring used inside list items.
- [`dicts.md`](dicts.md) — the sister object type.
- [`nesting.md`](nesting.md) — real-world composite examples.
- [`schema_reference.md`](schema_reference.md) — every supported keyword.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

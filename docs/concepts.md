# 1. Concepts & Architecture

## The Big Picture

Object Aligner compares two structured objects — a **gold** (ground truth) and a **pred** (prediction) — and returns a similarity score between 0 and 1, plus a detailed explanation of how they differ.

The comparison is governed by a **schema** that tells the aligner:

1. **What type** each piece of data is (string, number, boolean, array, object)
2. **How to score** each type (exact match, fuzzy match, etc.)
3. **How to align** collections (fixed order vs. best-match reordering)

## Core Abstractions

### Schema

The schema is a JSON Schema–inspired dictionary that describes the structure of your data. Every call to `ObjectAligner` requires a schema. It determines which alignment algorithm is used at each level of the data.

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "score": "jaro"},
        "age":  {"type": "integer", "score": "invdiff"}
    },
    "keyScore": "exact",
    "keyImportance": 1.0,
    "valueImportance": 1.0
}
```

See [Schema Reference](schema_reference.md) for the complete list of supported keywords.

### Match Types

Every alignment result is wrapped in one of three frozen dataclasses:

| Type | Used for | Fields |
|------|----------|--------|
| `MatchItem` | Primitives (strings, numbers, booleans) | `score`, `gold`, `pred` |
| `MatchList` | Arrays / lists | `score`, `children` (list of child matches) |
| `MatchDict` | Objects / dictionaries | `score`, `children` (dict mapping key-matches to value-matches) |

All scores are in **[0, 1]** where 1.0 means perfect match.

```python
from object_aligner.object_aligner import MatchItem, MatchList, MatchDict

# A single primitive match
m = MatchItem(score=0.87, gold="hello", pred="hallo")

# A list-level match wrapping child item matches
lst = MatchList(score=0.75, children=[m, MatchItem(score=0.5, gold="world", pred="werld")])

# A dict-level match mapping key-matches to value-matches
dct = MatchDict(score=0.8, children={
    MatchItem(score=1.0, gold="name", pred="name"): MatchItem(score=0.9, gold="Alice", pred="Alicia")
})
```

### ObjectAligner Class

The main entry point. You create it with an identifier and a schema:

```python
from object_aligner import ObjectAligner

aligner = ObjectAligner("my-evaluator", schema)
```

It provides two primary methods:

- **`align(gold, pred, skip_validation=False)`** — Returns the match object (`MatchItem`, `MatchList`, or `MatchDict`) representing the full alignment tree.
- **`metric(gold, pred, debug=False)`** — Validates both objects against the schema, runs alignment, and returns a dict with `"score"` (float) and `"reasoning"` (human-readable string).

## Alignment Flow

```
ObjectAligner.align(gold, pred)
  │
  ├─ Is it a bool?    → _align_booleans()     → MatchItem (0 or 1)
  ├─ Is it a number?  → _align_numbers()      → MatchItem (0..1)
  ├─ Is it a string?  → _align_strings()      → MatchItem (0..1)
  ├─ Is it a list?    → _align_lists()        → MatchList (recursive)
  │    ├─ prefixItems?          → _align_lists_prefix()
  │    ├─ items + order=fixed   → _align_lists_fixed()    (DP alignment)
  │    └─ items + order=align   → _align_lists_reorder()  (Hungarian algorithm)
  └─ Is it a dict?    → _align_dicts()        → MatchDict (recursive)
       └─ Keys matched via Hungarian algorithm → values aligned recursively
```

Every branch is **recursive**: a list of dicts of lists will be handled naturally by the dispatcher.

## Utility Functions

| Function | Purpose |
|----------|---------|
| `similarity_exact(a, b)` | Returns 1.0 if `a == b`, else 0.0 |
| `similarity_num_inv_diff(a, b)` | Returns `1 / (1 + |a - b|)` — closer numbers score higher |
| `similarity_string_jaro(a, b)` | Jaro normalized similarity between two strings |
| `path2str(p)` | Converts an alignment path (list of keys/indices) to a readable string |
| `to_pct_str(v)` | Formats a fraction as a percentage string, e.g. `"87%"` |

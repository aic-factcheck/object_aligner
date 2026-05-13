# 1. Concepts & Architecture

[Docs](index.md) › Concepts & Architecture

## The Big Picture

Object Aligner compares two structured objects — a **gold** (ground truth) and a **pred** (prediction) — and returns a similarity score between 0 and 1, plus optional explanation of how they differ.

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
from object_aligner import MatchItem, MatchList, MatchDict

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

The main entry point. You create it with a schema and optional reasoning configuration:

```python
from object_aligner import ObjectAligner

aligner = ObjectAligner(schema)
aligner_with_reasoning = ObjectAligner(schema, generate_reasoning=True)
```

`reasoning_templates` (and `feedback_templates`) let you override selected built-in template strings. The packaged defaults ship as TOML data under `src/object_aligner/templates/` (`reasoning.toml`, `feedback.toml`, `feedback.compact.toml`); the public helper `load_templates_from_toml(path)` reads user-authored override files in the same format.

It provides two primary methods:

- **`align(gold, pred, skip_validation=False)`** — Returns the match object (`MatchItem`, `MatchList`, or `MatchDict`) representing the full alignment tree.
- **`metric(gold, pred, debug=False, generate_reasoning=None, generate_feedback=None)`** — Validates both objects against the schema, runs alignment, and returns `{"score": ...}` by default. It adds `"reasoning"` when reasoning is enabled, a top-K prescriptive `"feedback"` string when `generate_feedback=True` (or a structured dict when `generate_feedback="full"`), and a structured `"debug"` alignment tree when `debug=True`.

For post-alignment analysis, `ObjectAligner` exposes three sibling outputs all derived from the same match tree: [`attribute()`](attribution.md) decomposes the deficit into per-path contributions, [`repair()`](repair.md) emits scored RFC 6902-flavored repair ops, and [`feedback()`](feedback.md) renders an optimizer-ready prescriptive feedback string on top of them.

### Migration note

The older `ObjectAligner("name", schema)` constructor form and `get_name()` were removed.

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

---

## See also

- [`primitives.md`](primitives.md) — string, number, and boolean leaves.
- [`schema_reference.md`](schema_reference.md) — every supported keyword.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

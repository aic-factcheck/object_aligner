# 7. Schema Reference

This is a complete reference of all schema keywords recognized by Object Aligner. Keywords marked with ⚡ are custom extensions beyond standard JSON Schema.

---

## Top-level / shared keywords

| Keyword | Type | Default | Applies to | Description |
|---------|------|---------|------------|-------------|
| `type` ⚡ | string | *(required)* | all | One of `"string"`, `"integer"`, `"number"`, `"boolean"`, `"array"`, `"object"` |

All standard JSON Schema validation keywords (e.g., `required`, `additionalProperties`, `minItems`, `maxItems`, `enum`, etc.) are also accepted and used during validation in `metric()`, but they do **not** affect alignment behavior.

---

## String type (`"type": "string"`)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `score` ⚡ | string | `"jaro"` | Scoring function: `"jaro"` or `"exact"` |
| `threshold` ⚡ | float | `0.0` | Scores below this value are set to 0.0 |

### Score functions

| Value | Formula | Use when |
|-------|---------|----------|
| `"jaro"` | Jaro normalized similarity via `rapidfuzz` | Names, labels, short text — tolerant of typos |
| `"exact"` | `1.0 if a == b else 0.0` | Enum values, IDs, categorical strings |

---

## Number / Integer type (`"type": "integer"` or `"type": "number"`)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `score` ⚡ | string | `"invdiff"` | Scoring function: `"invdiff"` or `"exact"` |
| `threshold` ⚡ | float | `0.0` | Scores below this value are set to 0.0 |

### Score functions

| Value | Formula | Use when |
|-------|---------|----------|
| `"invdiff"` | `1 / (1 + |a - b|)` | Continuous values where closeness matters (ages, prices, scores) |
| `"exact"` | `1.0 if a == b else 0.0` | Categorical integers, identifiers |

---

## Boolean type (`"type": "boolean"`)

No additional keywords. Booleans are always compared exactly.

---

## Array type (`"type": "array"`)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `items` ⚡ | object | — | Schema for variable-length items. Used with `order` and/or after `prefixItems`. |
| `prefixItems` ⚡ | list[object] | — | Per-position schemas for the fixed-length prefix. Each element is a schema object. |
| `prefixWeights` ⚡ | list[float] | all `1.0` | Weight for each prefix position. Length must match `prefixItems`. Normalized internally. |
| `order` ⚡ | string | `"fixed"` | Alignment strategy for `items`: `"fixed"` (DP sequence alignment) or `"align"` (Hungarian reordering) |
| `ignoreExcess` ⚡ | bool | `false` | If true, extra predicted items don't count toward normalization denominator |
| `ignoreMissing` ⚡ | bool | `false` | If true, missing gold items don't count toward normalization denominator |
| `prefixImportance` ⚡ | float | — | Weight of prefix score in combined prefix+items score. **Required** when both `prefixItems` and `items` are present. |
| `restImportance` ⚡ | float | — | Weight of tail (`items`) score. **Required** when both `prefixItems` and `items` are present. |

### Combining prefixItems and items

When both are present:
- The first `len(prefixItems)` elements are aligned positionally (with weights)
- The remaining elements are aligned according to `order`
- The final score is: `(prefixImportance * prefixScore + restImportance * restScore) / (prefixImportance + restImportance)`

### Fixed-order alignment details

Uses dynamic programming (similar to DNA sequence alignment):
- Find the optimal pairing that maximizes total similarity
- Allows gaps (unmatched items scored as 0)
- Preserves left-to-right ordering

### Reorder alignment details

Uses the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`):
- Build an `n × m` similarity matrix between all gold–pred pairs
- Find the assignment that maximizes total similarity
- Unmatched items are scored as 0

---

## Object type (`"type": "object"`)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `properties` ⚡ | object | *(required)* | Mapping from key name → property schema. Each property schema defines the type and scoring for that key's value. |
| `keyScore` ⚡ | string | `"jaro"` | Key comparison function: `"jaro"` or `"exact"` |
| `keyThreshold` ⚡ | float | `0.0` | Minimum key similarity to form a pairing; pairs below this are treated as unaligned |
| `keyImportance` ⚡ | float | `1.0` | Weight of key score in the final dict score |
| `valueImportance` ⚡ | float | `1.0` | Weight of value score in the final dict score |

### Property-level keywords (inside `properties`)

Each property schema supports all the type-specific keywords above, plus:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `valueWeight` ⚡ | float | `1.0` | Relative weight of this property's value score. Higher = more important. Weights are normalized across all properties before averaging. |

### Key alignment process

1. Compute a similarity matrix between all gold keys and pred keys using `keyScore`
2. Apply `keyThreshold` to zero out low-similarity pairings
3. Run Hungarian algorithm to find the optimal key assignment
4. For each paired (gold_key, pred_key), look up the property schema from `properties[gold_key]`
5. Recursively align the values using that schema
6. Compute weighted average of value scores
7. Combine key score and value score using `keyImportance` / `valueImportance`

---

## Quick reference: scoring defaults

| Type | Default score | Fuzzy? |
|------|--------------|--------|
| `boolean` | exact | No |
| `integer` / `number` | `invdiff` | Yes — close values score higher |
| `string` | `jaro` | Yes — similar strings score higher |
| `array` (keys/items) | depends on inner type | Recursive |
| `object` (keys) | `jaro` | Yes — similar keys pair together |

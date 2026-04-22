# 7. Schema Reference

This is a complete reference of all schema keywords recognized by Object Aligner. Keywords marked with ⚡ are custom extensions beyond standard JSON Schema.

---

## Top-level / shared keywords

| Keyword | Type | Default | Applies to | Description |
|---------|------|---------|------------|-------------|
| `type` ⚡ | string | *(required)* | all | One of `"string"`, `"integer"`, `"number"`, `"boolean"`, `"array"`, `"object"` |

All standard JSON Schema validation keywords (e.g., `required`, `additionalProperties`, `minItems`, `maxItems`, `enum`, etc.) are also accepted and used during validation in `metric()`, but they do **not** affect alignment behavior.

Custom primitive metrics are supplied through `ObjectAligner(..., custom_metrics=...)`, not as inline callables in the schema.

---

## String type (`"type": "string"`)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `score` ⚡ | string | `"jaro"` | Any built-in string metric name or a registered custom metric name |
| `threshold` ⚡ | float | `0.0` | Scores below this value are set to `0.0` |

### Score functions

| Value | Formula / implementation | Use when |
|-------|---------------------------|----------|
| `"jaro"` | Jaro normalized similarity via `rapidfuzz` | Names, labels, short text with typos |
| `"jaro_winkler"` | Jaro-Winkler normalized similarity | Like Jaro, but shared prefixes should count more |
| `"levenshtein"` | Levenshtein normalized similarity | General-purpose edit-distance matching |
| `"damerau_levenshtein"` | Damerau-Levenshtein normalized similarity | Adjacent transpositions should be treated naturally |
| `"osa"` | Optimal string alignment normalized similarity | Another transposition-aware edit metric |
| `"indel"` | Indel normalized similarity | Insert/delete-oriented matching |
| `"lcsseq"` | Longest-common-subsequence normalized similarity | Sequence overlap matters more than exact edits |
| `"exact"` | `1.0 if a == b else 0.0` | Enum values, IDs, categorical strings |
| custom name | User-provided `(gold, pred) -> float` | Semantic similarity or domain-specific scoring |

---

## Number / Integer type (`"type": "integer"` or `"type": "number"`)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `score` ⚡ | string | `"invdiff"` | `"invdiff"`, `"exact"`, or a registered custom metric name |
| `threshold` ⚡ | float | `0.0` | Scores below this value are set to `0.0` |

### Score functions

| Value | Formula / implementation | Use when |
|-------|---------------------------|----------|
| `"invdiff"` | `1 / (1 + |a - b|)` | Continuous values where closeness matters |
| `"exact"` | `1.0 if a == b else 0.0` | Categorical integers, identifiers |
| custom name | User-provided `(gold, pred) -> float` | Domain-specific numeric scoring |

For integer schemas, custom metric lookup checks the `integer` registry first and then falls back to the `number` registry.

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
- Allows gaps (unmatched items scored as `0`)
- Preserves left-to-right ordering

### Reorder alignment details

Uses the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`):
- Build an `n × m` similarity matrix between all gold–pred pairs
- Find the assignment that maximizes total similarity
- Unmatched items are scored as `0`

---

## Object type (`"type": "object"`)

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `properties` ⚡ | object | *(required)* | Mapping from key name → property schema. Each property schema defines the type and scoring for that key's value. |
| `keyScore` ⚡ | string | `"jaro"` | Key comparison function: `"jaro"` or `"exact"` |
| `keyThreshold` ⚡ | float | `0.0` | Minimum key similarity to form a pairing; pairs below this are treated as unaligned |
| `keyImportance` ⚡ | float | `1.0` | Weight of key score in the final dict score |
| `valueImportance` ⚡ | float | `1.0` | Weight of value score in the final dict score |

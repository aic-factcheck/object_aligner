# 2. Primitive Types

Primitive types are the leaves of the alignment tree. They produce a `MatchItem` with a `score`, `gold`, and `pred` field.

## Booleans

Boolean comparison is always **exact** — there's no notion of "fuzzy" for booleans.

```python
from object_aligner import ObjectAligner

schema = {"type": "boolean"}
aligner = ObjectAligner("bool-test", schema)

# Perfect match
result = aligner.align(True, True)
print(result)  # MatchItem(score=1.0, gold=True, pred=True)

# Mismatch
result = aligner.align(True, False)
print(result)  # MatchItem(score=0.0, gold=True, pred=False)
```

### Schema keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | Must be `"boolean"` |

No scoring options — booleans are always compared exactly.

---

## Numbers (integers and floats)

Numbers support two scoring modes:

### Exact match (`"exact"`)

Returns 1.0 if the values are equal, 0.0 otherwise. Best for categorical or identifier numbers where any difference is fatal.

```python
schema = {"type": "integer", "score": "exact"}
aligner = ObjectAligner("num-exact", schema)

print(aligner.align(42, 42))   # MatchItem(score=1.0, gold=42, pred=42)
print(aligner.align(42, 43))   # MatchItem(score=0.0, gold=42, pred=43)
```

### Inverse difference (`"invdiff"`) — *default*

Returns `1 / (1 + |a - b|)`. This gives a smooth penalty for numeric differences: close values score high, distant values score low.

```python
schema = {"type": "integer", "score": "invdiff"}
aligner = ObjectAligner("num-invdiff", schema)

print(aligner.align(50, 51))   # score ≈ 0.5  (1 / (1 + 1))
print(aligner.align(50, 52))   # score ≈ 0.33 (1 / (1 + 2))
print(aligner.align(50, 100))  # score ≈ 0.02 (1 / (1 + 50))
print(aligner.align(50, 50))   # score = 1.0
```

This is particularly useful for evaluating numerical predictions like ages, prices, or measurements where being "close" is meaningful.

### Threshold

You can set a `threshold` (default 0.0) to zero out scores below it. This acts as a hard cutoff: any similarity below the threshold is treated as a complete mismatch.

```python
schema = {"type": "integer", "score": "invdiff", "threshold": 0.5}
aligner = ObjectAligner("num-thresh", schema)

# |50 - 51| = 1 → 1/(1+1) = 0.5, which is not < 0.5, so it passes
print(aligner.align(50, 51))  # MatchItem(score=0.5, gold=50, pred=51)

# |50 - 52| = 2 → 1/(1+2) ≈ 0.33, which is < 0.5, so it becomes 0.0
print(aligner.align(50, 52))  # MatchItem(score=0.0, gold=50, pred=52)
```

### Schema keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | `"integer"` or `"number"` |
| `score` | string | `"invdiff"` | `"exact"` or `"invdiff"` |
| `threshold` | float | `0.0` | Minimum score to be considered a match; scores below are set to 0.0 |

---

## Strings

Strings support two scoring modes:

### Jaro similarity (`"jaro"`) — *default*

Uses the [Jaro similarity](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) from `rapidfuzz`. It accounts for character transpositions and partial matches, making it ideal for names, labels, and other short strings where typos are common.

```python
schema = {"type": "string", "score": "jaro"}
aligner = ObjectAligner("str-jaro", schema)

print(aligner.align("hello", "hallo"))    # score ≈ 0.87
print(aligner.align("Katherine", "Catherine"))  # score ≈ 0.92
print(aligner.align("hello", "world"))    # score = 0.0
```

### Exact match (`"exact"`)

Returns 1.0 if the strings are identical, 0.0 otherwise. Use when you need strict matching (e.g., enum values, IDs).

```python
schema = {"type": "string", "score": "exact"}
aligner = ObjectAligner("str-exact", schema)

print(aligner.align("hello", "hello"))  # MatchItem(score=1.0, ...)
print(aligner.align("hello", "Hello"))  # MatchItem(score=0.0, ...)
```

> **Note:** Jaro similarity is case-sensitive. `"hello"` vs `"Hello"` will not score 1.0.

### Threshold

Same as for numbers — scores below the threshold are set to 0.0:

```python
schema = {"type": "string", "score": "jaro", "threshold": 0.7}
aligner = ObjectAligner("str-thresh", schema)

# "cat" vs "car" has Jaro similarity ≈ 0.78 → above threshold
print(aligner.align("cat", "car"))   # score ≈ 0.78

# "cat" vs "dog" has Jaro similarity ≈ 0.0 → below threshold → 0.0
print(aligner.align("cat", "dog"))   # score = 0.0
```

### Real-world example: Matching person names

```python
schema = {"type": "string", "score": "jaro", "threshold": 0.5}
aligner = ObjectAligner("name-match", schema)

# Common name variations
print(aligner.align("Elizabeth", "Elisabeth"))  # score ≈ 0.97
print(aligner.align("Jonathan", "Jonathon"))    # score ≈ 0.97
print(aligner.align("Margaret", "Margarret"))   # score ≈ 0.96
print(aligner.align("Bob", "Robert"))           # score ≈ 0.0 (too different → thresholded)
```

### Schema keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | Must be `"string"` |
| `score` | string | `"jaro"` | `"jaro"` or `"exact"` |
| `threshold` | float | `0.0` | Minimum score; scores below are set to 0.0 |

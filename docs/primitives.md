# 2. Primitive Types

[Docs](index.md) › Primitive Types

Primitive types are the leaves of the alignment tree. They produce a `MatchItem` with a `score`, `gold`, and `pred` field.

## Booleans

Boolean comparison is always **exact** — there's no notion of "fuzzy" for booleans.

```python
from object_aligner import ObjectAligner

schema = {"type": "boolean"}
aligner = ObjectAligner(schema)

print(aligner.align(True, True))   # MatchItem(score=1.0, gold=True, pred=True)
print(aligner.align(True, False))  # MatchItem(score=0.0, gold=True, pred=False)
```

### Schema keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | Must be `"boolean"` |

No scoring options — booleans are always compared exactly.

---

## Numbers (integers and floats)

Numbers support three built-in scoring modes:

### Exact match (`"exact"`)

Returns `1.0` if the values are equal, `0.0` otherwise. Best for categorical or identifier numbers where any difference is fatal.

```python
schema = {"type": "integer", "score": "exact"}
aligner = ObjectAligner(schema)

print(aligner.align(42, 42))
print(aligner.align(42, 43))
```

### Inverse difference (`"invdiff"`) — *default*

Returns `1 / (1 + |a - b|)`. This gives a smooth penalty for numeric differences: close values score high, distant values score low.

```python
schema = {"type": "integer", "score": "invdiff"}
aligner = ObjectAligner(schema)

print(aligner.align(50, 51))
print(aligner.align(50, 52))
print(aligner.align(50, 100))
print(aligner.align(50, 50))
```

> **Calibration caveat — `invdiff` depends on the absolute scale of the field.**
> It scores the raw difference, so `invdiff(2020, 2021) = 0.5` (adjacent years)
> while `invdiff(1000000, 1000010) ≈ 0.09` even though the second pair is
> relatively almost identical. Rescaling a field's unit (cents vs. euros,
> ms vs. s) changes its scores. `invdiff` works well for small-integer domains
> where a difference of 1 is genuinely "half wrong" (counts, days, hours);
> for quantities with arbitrary magnitude prefer `"relative"` below.

### Relative difference (`"relative"`)

Returns $1 - \min\!\bigl(1, |a - b| / \max(|a|, |b|)\bigr)$, with equal values
(including `0` vs `0`) scoring `1.0`. The score is **scale-invariant**:
`relative(k·a, k·b) == relative(a, b)` for any `k ≠ 0`, so it is the right
choice for measurements, prices, and other quantities whose unit is arbitrary.

```python
schema = {"type": "number", "score": "relative"}
aligner = ObjectAligner(schema)

print(aligner.align(2020, 2021))            # ≈ 0.9995
print(aligner.align(100.0, 110.0))          # ≈ 0.909
print(aligner.align(1_000_000, 1_000_010))  # ≈ 0.99999
print(aligner.align(1, -1))                 # 0.0 (difference ≥ larger magnitude)
```

### Threshold

You can set a `threshold` (default `0.0`) to zero out scores below it.

```python
schema = {"type": "integer", "score": "invdiff", "threshold": 0.5}
aligner = ObjectAligner(schema)

print(aligner.align(50, 51))  # score = 0.5
print(aligner.align(50, 52))  # score = 0.0
```

### Custom numeric metrics

You can register your own named metric through `ObjectAligner(..., custom_metrics=...)`. The schema still references it by name through `score`.

Metric callables must have signature `(gold, pred) -> float` and return a value in `[0, 1]`.

```python
from object_aligner import ObjectAligner


def closish(gold: float, pred: float) -> float:
    return 0.8 if abs(gold - pred) <= 2 else 0.2


aligner = ObjectAligner(
    {"type": "number", "score": "closish", "threshold": 0.5},
    custom_metrics={"number": {"closish": closish}},
)

print(aligner.align(10, 12))  # score = 0.8
print(aligner.align(10, 20))  # score = 0.0 after thresholding
```

Integer schemas use the `integer` registry first and then fall back to the `number` registry.

### Schema keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | `"integer"` or `"number"` |
| `score` | string | `"invdiff"` | `"exact"`, `"invdiff"`, `"relative"`, or a registered custom metric name |
| `threshold` | float | `0.0` | Minimum score to be considered a match; scores below are set to `0.0` |

---

## Strings

Strings support these built-in scoring modes:

- `jaro` *(default)*
- `jaro_winkler`
- `levenshtein`
- `damerau_levenshtein`
- `osa`
- `indel`
- `lcsseq`
- `exact`

All built-in scores return values in `[0, 1]`.

### Jaro similarity (`"jaro"`) — *default*

Good for names, labels, and short text with typos or transpositions.

```python
schema = {"type": "string", "score": "jaro"}
aligner = ObjectAligner(schema)

print(aligner.align("hello", "hallo"))
print(aligner.align("Katherine", "Catherine"))
print(aligner.align("hello", "world"))
```

> **Calibration caveat — Jaro has a high floor.** Jaro-family metrics assign
> substantial similarity to strings that are semantically unrelated:
> `jaro("invoice_total", "customer_name") ≈ 0.53`,
> `jaro("2024-01-15", "9999-12-31") = 0.60`. With the default
> `threshold: 0.0`, a field whose every prediction is *wrong* can still
> average above 0.5, compressing the useful range of the score. When a
> "completely wrong" string should score near 0, set a `threshold`
> (e.g. `0.5`–`0.7`) or use `"exact"` for closed vocabularies.

### Other built-in string metrics

| Score | Good for |
|------|----------|
| `exact` | Strict equality, IDs, enums, categorical values |
| `jaro_winkler` | Like Jaro, but gives extra weight to shared prefixes |
| `levenshtein` | Classic edit distance similarity |
| `damerau_levenshtein` | Edit distance that treats adjacent transpositions naturally |
| `osa` | Optimal string alignment, another transposition-aware edit metric |
| `indel` | Insert/delete-oriented similarity |
| `lcsseq` | Similarity based on longest common subsequence |

```python
schema = {"type": "string", "score": "damerau_levenshtein"}
aligner = ObjectAligner(schema)

print(aligner.align("abcd", "abdc"))
print(aligner.align("kitten", "sitting"))
```

### Exact match (`"exact"`)

Returns `1.0` if the strings are identical, `0.0` otherwise.

```python
schema = {"type": "string", "score": "exact"}
aligner = ObjectAligner(schema)

print(aligner.align("hello", "hello"))
print(aligner.align("hello", "Hello"))
```

> **Note:** String metrics are case-sensitive unless your custom metric chooses otherwise.

### Threshold

Scores below `threshold` are set to `0.0`.

```python
schema = {"type": "string", "score": "levenshtein", "threshold": 0.7}
aligner = ObjectAligner(schema)

print(aligner.align("abcd", "abce"))  # score = 0.75
print(aligner.align("abcd", "abdc"))  # score = 0.0 after thresholding
```

### Custom string metrics

Custom metrics use the same registration pattern as numeric metrics.

```python
from object_aligner import ObjectAligner


def semantic_toy(gold: str, pred: str) -> float:
    return 0.95 if gold.lower()[0] == pred.lower()[0] else 0.2


aligner = ObjectAligner(
    {"type": "string", "score": "semantic_toy", "threshold": 0.5},
    custom_metrics={"string": {"semantic_toy": semantic_toy}},
)

print(aligner.align("cat", "car"))
print(aligner.align("cat", "dog"))
```

This same mechanism is what the embedding-backed semantic-similarity metric uses; see [Semantic Similarity](semantic.md) for the production-ready stack (OpenAI-compatible transport, in-memory and SQLite caches, pre-warming).

### Schema keywords

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | *(required)* | Must be `"string"` |
| `score` | string | `"jaro"` | Any built-in string score or a registered custom metric name |
| `threshold` | float | `0.0` | Minimum score; scores below are set to `0.0` |

---

## See also

- [`lists.md`](lists.md) — composing primitives into arrays.
- [`dicts.md`](dicts.md) — composing primitives into objects.
- [`schema_reference.md`](schema_reference.md) — every supported keyword.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

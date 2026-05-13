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

Numbers support two built-in scoring modes:

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
| `score` | string | `"invdiff"` | `"exact"`, `"invdiff"`, or a registered custom metric name |
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

This same mechanism can later be used for embedding-based or other semantic similarity functions.

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

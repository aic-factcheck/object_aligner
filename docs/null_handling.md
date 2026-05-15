# 14. Null Handling

[Docs](index.md) › Null Handling

Real-world LLM extractors emit `null` whenever the answer is missing or
unknown. Earlier versions of Object Aligner crashed on `None` values
because the alignment dispatcher had no branch for them. This chapter
describes the **null-aware leaf**: a small dispatch path activated when
exactly one of `gold` / `pred` is `None`, plus the per-field
`nullScore` schema keyword that calibrates how harshly the asymmetric
case should be penalized.

The rule is simple:

| Case | Score |
|------|-------|
| `gold` and `pred` both `None` | `1.0` (a perfect null/null match) |
| Exactly one of them `None`    | the field's `nullScore` (default `0.0`) |
| Neither is `None`             | the existing primitive comparator (e.g. `jaro`, `invdiff`) |

`nullScore` is **symmetric**: the same number governs both
gold-null-pred-value and gold-value-pred-null. (The 3-state policy that
distinguishes the two directions is intentionally out of scope for v1.)

---

## Quickstart

```python
from object_aligner import ObjectAligner

schema = {"type": ["string", "null"], "nullScore": 0.8}
aligner = ObjectAligner(schema)

aligner.metric(None, None)         # {'score': 1.0}
aligner.metric(None, "Smith")      # {'score': 0.8}
aligner.metric("Smith", None)      # {'score': 0.8}
aligner.metric("Smith", "Smyth")   # primitive comparator (jaro)
```

The JSON Schema `type` must include `"null"` (use a union like
`["string", "null"]` or a bare `"null"`). Otherwise default schema
validation rejects the null prediction and `metric()` returns
`{"score": 0.0}` before alignment ever runs — which is the correct
behavior when the schema says "this field cannot be null."

---

## Shared setup for the examples

Every example below uses the same imports:

```python
from object_aligner import ObjectAligner
```

Each example builds its own `schema`, `gold`, and `pred` (the worked
cases need different shapes), so nothing else is shared.

---

## Example 1 — calibrated penalty per property

The point of a *per-field* `nullScore` is that not every missing field
hurts equally. A missing `middle_name` is fine; a missing `diagnosis`
in a medical extraction is catastrophic.

```python
schema = {
    "type": "object",
    "properties": {
        "diagnosis":   {"type": ["string", "null"], "nullScore": 0.0},
        "middle_name": {"type": ["string", "null"], "nullScore": 0.8},
    },
}
aligner = ObjectAligner(schema)

gold = {"diagnosis": "flu", "middle_name": None}
pred = {"diagnosis": None,  "middle_name": None}

aligner.metric(gold, pred)
# {'score': 0.5}
```

The score breaks down cleanly: the values block averages `0.0`
(diagnosis: asymmetric null with `nullScore=0.0`) and `1.0`
(middle_name: both null), giving `0.5`. With the default
`keyImportance = 0` the keys block does not contribute, so the dict
score is `0.5`.

---

## Example 2 — both sides null is always a perfect match

`nullScore` does not enter when both sides are null:

```python
schema = {"type": ["string", "null"], "nullScore": 0.0}
aligner = ObjectAligner(schema)
aligner.metric(None, None)   # {'score': 1.0} — not 0.0
```

That's deliberate: "both gold and pred agree there is nothing here" is
a real, positive observation, not a deficit. If you want to *exclude*
null-on-both positions from the score entirely, use
`ignoreExcess` / `ignoreMissing` on the enclosing array; for object
properties it's expected behavior that paired nulls contribute a
positive score to the average.

---

## Example 3 — nullable nested object or array

`nullScore` applies at any level, not just primitive leaves. A whole
sub-object or array can be `null`:

```python
schema = {
    "type": "object",
    "properties": {
        "address": {
            "type": ["object", "null"],
            "nullScore": 0.5,
            "properties": {
                "city": {"type": "string"},
                "zip":  {"type": "string"},
            },
        },
    },
}
aligner = ObjectAligner(schema)

aligner.metric(
    {"address": {"city": "Prague", "zip": "11000"}},
    {"address": None},
)
# {'score': 0.5}   (value scored 0.5 via nullScore; default keyImportance=0 → keys do not contribute)
```

The dispatcher fires the null branch **before** descending into the
object's properties, so the missing subtree never has to be matched
position by position.

---

## Example 4 — nullable list items

`items` is just another schema slot; nullable items work the same way.
Use `order: "align"` if you want the Hungarian assignment to decide
which gold position pairs with which pred position when nulls mingle
with real values:

```python
schema = {
    "type": "array",
    "order": "align",
    "items": {"type": ["integer", "null"], "nullScore": 0.5},
}
aligner = ObjectAligner(schema)

aligner.metric([1, None], [1, 2])
# {'score': 0.75}   (1↔1 → 1.0, None↔2 → 0.5)
```

For fixed-position lists the same `nullScore` applies element-wise.

---

## Example 5 — feedback and repair surface the null case

The downstream consumers — `attribute()`, `repair()`, `feedback()`,
`describe()` — all distinguish null/value mismatches from generic
primitive mismatches via the dedicated `MatchItem.kind = "null"` and
`RepairOp.kind = "null_value_replace"` markers:

```python
schema = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": ["string", "null"], "nullScore": 0.0},
    },
}
aligner = ObjectAligner(schema, generate_feedback=True)
r = aligner.metric({"diagnosis": "flu"}, {"diagnosis": None})

print(r["feedback"])
# The prediction scored 0.00 (deficit 1.00). Top 1 of 1 fix locations:
# 1. /diagnosis: null/value mismatch (expected 'flu', got None). Fixing this recovers +1.000.
# Focus on null-value errors — they account for 100% of the deficit shown.
```

`aligner.repair(...).ops[0]` is a `RepairOp(op="replace",
path="/diagnosis", kind="null_value_replace", value="flu", ...)`.
`RepairResult.apply_to(pred)` round-trips back to the gold value.

---

## API reference

- `nullScore` schema keyword — see
  [Schema Reference](schema_reference.md#null-aware-scoring-nullscore).
- `MatchItem.kind` — see
  [`api.md#matchitem`](api.md#matchitem); equals `"null"` for null-aware
  leaves.
- `RepairOp.kind == "null_value_replace"` — see
  [`api.md#repairop`](api.md#repairop) and
  [Scored JSON-Patch Repair](repair.md).
- Feedback template key `feedback.op.null_value_replace` — see
  [Prompt-Optimizer Feedback](feedback.md) for placeholder details.
- Description template keys `describe.null.match` /
  `describe.null.mismatch` — see
  [Plain-English Description](describe.md).

---

## Caveats

- **Declare nullability in the schema.** The JSON Schema validator runs
  before alignment. If the schema says `type: "string"` and the
  prediction is `null`, validation fails and `metric()` returns
  `{"score": 0.0}` without ever consulting `nullScore`. Use union
  types such as `type: ["string", "null"]` to opt in.
- **Range.** `nullScore` must be a real number in `[0, 1]`. Booleans
  and non-numeric types are rejected at construction time, as are
  out-of-range floats.
- **Symmetric for v1.** A single `nullScore` covers both directions
  (gold-null-pred-value and gold-value-pred-null). The asymmetric
  variant (separate "hallucination" vs "omission" penalties) is a
  candidate for a future release.
- **List-position semantics.** A `None` *item* inside a list — present
  in the data — is **not** the same thing as a *missing* item slot
  produced by an unequal-length pairing. Nullability of items is a
  per-item property; absence is a list-length property handled by
  `ignoreExcess` / `ignoreMissing`.
- **Skip-validation mode.** Under `skip_validation=True` the schema's
  `type` is no longer consulted, so the null branch fires for any
  `None` value regardless of the declared types. The same `nullScore`
  rules apply.

---

## See also

- [Schema Reference — Null-aware scoring](schema_reference.md#null-aware-scoring-nullscore)
- [Primitive Types](primitives.md) — non-null comparators used when both sides have values.
- [Per-Property Score Attribution](attribution.md) — null mismatches surface with `leaf_kind="null"`.
- [Scored JSON-Patch Repair](repair.md) — the `null_value_replace` op.
- [Prompt-Optimizer Feedback](feedback.md) — null/value mismatch in feedback templates.

[← Documentation home](index.md)

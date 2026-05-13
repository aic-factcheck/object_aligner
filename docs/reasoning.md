# 7. Plain-English Reasoning

[Docs](index.md) › Plain-English Reasoning

`metric()` answers *how well* a prediction matches the gold. `attribute()`
answers *where the deficit lives*. `repair()` answers *what to change*.
`feedback()` produces a prompt-optimizer-shaped feedback string. The
**`generate_reasoning`** feature complements them with a different output:
a **plain-English, human-readable walk of the alignment tree** — useful for
spot-checking evaluation runs, surfacing failures in dashboards, or attaching
a one-shot explanation alongside the score.

The reasoning is **deterministic and template-based — there is no LLM
involved**. The alignment is already performed by `align()`; the reasoning
layer is a deterministic projection of the resulting match tree onto a text
surface.

This page documents the API and shows worked examples for every match kind
and every render branch.

---

## Quickstart

```python
from object_aligner import ObjectAligner

schema = {"type": "string", "score": "jaro"}
aligner = ObjectAligner(schema, generate_reasoning=True)

result = aligner.metric("hello", "helo")
print(result["score"])
# 0.9333333333333333
print(result["reasoning"])
# The predicted output scores overall 93%, let us align the predicted output to the gold and analyze the differences:
# The predicted value "helo" does not match the gold "hello" (score=93%).
```

You can also enable reasoning on a per-call basis without flipping the
constructor flag:

```python
aligner = ObjectAligner(schema)
aligner.metric("hello", "helo")                              # {"score": 0.93}
aligner.metric("hello", "helo", generate_reasoning=True)     # ... + reasoning
```

---

## Shared setup for the examples

Several examples below reuse a `movie_schema`, an `aligner`, and a
`gold` / `pred` pair. They are defined once here:

```python
from object_aligner import ObjectAligner

movie_schema = {
    "type": "object",
    "keyScore": "exact",
    "keyImportance": 0,
    "valueImportance": 1,
    "properties": {
        "title":  {"type": "string",  "score": "jaro",  "valueWeight": 2.0},
        "year":   {"type": "integer", "score": "exact", "valueWeight": 1.0},
        "genres": {
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
            "order": "align",
            "valueWeight": 1.0,
        },
    },
}
aligner = ObjectAligner(movie_schema, generate_reasoning=True)
gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
pred = {"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]}
```

Each example states explicitly when it builds its own schema/aligner instead
of reusing these.

---

## The model

`generate_reasoning=True` causes `metric()` to walk the match tree returned
by `align()` and render it into an indented English string.

1. If the overall score is exactly `1.0`, render the `metric.perfect`
   template and stop.
2. Otherwise, emit the `metric.imperfect_intro` line (carrying the overall
   percentage), then recurse into the root match node.
3. Each node selects one template by `(match_kind, outcome)` —
   `item.match` / `item.mismatch`, `list.match` / `list.mismatch`,
   `list.excess` / `list.missing`, `dict.match` / `dict.mismatch`,
   `dict.key.match` / `dict.key.mismatch`, `ref.match` / `ref.mismatch`.
4. `id` rows are intentionally rendered as the empty string — id fields are
   referential bookkeeping, not user-facing content.
5. Indentation grows with depth (two spaces per level) via the `{indent}`
   placeholder.
6. Schema-validation failure of `pred` short-circuits the whole walk: the
   `validation.error` template is rendered with `{path}` and `{message}`,
   and the score is `0.0`.

The renderer lives in `_ReasoningRenderer.render`
(`src/object_aligner/object_aligner.py`). Default template strings live in
`src/object_aligner/templates/reasoning.toml`.

---

## Examples

### Example 1: Perfect match

A perfect score short-circuits the walk and emits a single line — no tree
walk, no per-leaf detail.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

aligner_perfect = ObjectAligner(
    {"type": "string", "score": "jaro"},
    generate_reasoning=True,
)
print(aligner_perfect.metric("hello", "hello")["reasoning"])
# The predicted output perfectly matches the gold.
```

### Example 2: Primitive mismatch

For a primitive root the intro line is emitted, then a single
`item.mismatch` line with no indent.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

aligner_str = ObjectAligner(
    {"type": "string", "score": "jaro"},
    generate_reasoning=True,
)
print(aligner_str.metric("hello", "helo")["reasoning"])
# The predicted output scores overall 93%, let us align the predicted output to the gold and analyze the differences:
# The predicted value "helo" does not match the gold "hello" (score=93%).
```

### Example 3: List with excess and missing items

`order: "align"` lists produce one row per child plus rows for
unmatched-on-either-side items.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

list_schema = {
    "type": "array",
    "items": {"type": "integer", "score": "exact"},
    "order": "align",
}
aligner_list = ObjectAligner(list_schema, generate_reasoning=True)
print(aligner_list.metric([1, 2, 4], [2, 3])["reasoning"])
# The predicted output scores overall 25%, let us align the predicted output to the gold and analyze the differences:
# The predicted list scores 25%:
#   The predicted list item "3" is excessive, it was not in the gold.
#   The predicted output misses the "1" list item from the gold.
#   The predicted value "2" exactly matches the gold.
#   The predicted output misses the "4" list item from the gold.
```

### Example 4: Dictionary with key+value rows

Each `dict` child renders as a `KEY = ...` line followed by a
`VALUE = ...` line. Keys with `keyImportance=0` still appear in the
reasoning — the importance affects scoring, not rendering.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

dict_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "score": "jaro"},
        "age":  {"type": "integer", "score": "exact"},
    },
    "keyScore": "exact",
    "keyImportance": 0,
}
aligner_dict = ObjectAligner(dict_schema, generate_reasoning=True)
print(aligner_dict.metric(
    {"name": "Alice", "age": 30},
    {"name": "Alic",  "age": 31},
)["reasoning"])
# The predicted output scores overall 47%, let us align the predicted output to the gold and analyze the differences:
# The predicted dictionary scores 47%:
#   KEY = The predicted key "name" exactly matches the gold.
#   VALUE = The predicted value "Alic" does not match the gold "Alice" (score=93%).
#
#   KEY = The predicted key "age" exactly matches the gold.
#   VALUE = The predicted value "31" does not match the gold "30" (score=0%).
```

### Example 5: Nested dict + list (uses the shared setup)

Reuses `movie_schema`, `aligner`, `gold`, `pred` from the shared setup.

```python
print(aligner.metric(gold, pred)["reasoning"])
# The predicted output scores overall 67%, let us align the predicted output to the gold and analyze the differences:
# The predicted dictionary scores 67%:
#   KEY = The predicted key "title" exactly matches the gold.
#   VALUE = The predicted value "The Matrx" does not match the gold "The Matrix" (score=97%).
#
#   KEY = The predicted key "year" exactly matches the gold.
#   VALUE = The predicted value "2000" does not match the gold "1999" (score=0%).
#
#   KEY = The predicted key "genres" exactly matches the gold.
#   VALUE = The predicted list scores 75%:
#     The predicted value "Sci-Fi" exactly matches the gold.
#     The predicted value "Adventure" does not match the gold "Action" (score=50%).
```

The exact score depends on the chosen metric — what matters for this page is
that the reasoning shape parallels the match tree.

### Example 6: Referential alignment (`idScope` / `ref`)

`id` rows are intentionally rendered as the empty string — that's why the
`VALUE = ` lines below appear empty for `id` properties. References render
through `ref.match` / `ref.mismatch`.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

ref_schema = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string", "idScope": "person"},
                    "name": {"type": "string", "score": "jaro"},
                },
                "keyScore": "exact",
                "keyImportance": 0,
            },
            "order": "align",
        },
        "best_friend": {"type": "string", "ref": "person"},
    },
    "keyScore": "exact",
    "keyImportance": 0,
}
aligner_ref = ObjectAligner(ref_schema, generate_reasoning=True)
gold = {"people": [{"id": "p1", "name": "Alice"}, {"id": "p2", "name": "Bob"}], "best_friend": "p1"}
pred = {"people": [{"id": "x",  "name": "Aliec"}, {"id": "y",  "name": "Bob"}], "best_friend": "y"}
print(aligner_ref.metric(gold, pred)["reasoning"])
# (excerpt)
# ...
#   KEY = The predicted key "best_friend" exactly matches the gold.
#   VALUE = The predicted reference "y" does not match the gold reference "p1" under the inferred id mapping (score=0%).
```

### Example 7: Validation errors {#validation-errors}

If `pred` fails schema validation, `metric()` short-circuits to a score of
`0.0` and the reasoning carries the `validation.error` template instead of a
tree walk.

```python
strict_schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "keyScore": "exact",
}
strict = ObjectAligner(strict_schema, generate_reasoning=True)
print(strict.metric(
    {"name": "Alice", "age": 30},
    {"name": "Alice"},  # missing age → validation fail
))
# {'score': 0.0, 'reasoning': "JSON Schema validation failed for path=\"/\". Error message: 'age' is a required property."}
```

### Example 8: Customizing a single template

Pass a partial dict via `reasoning_templates`. Any key you do not provide
keeps its packaged default. Unknown keys raise a `ValueError` at
construction time (typos caught early), and unknown placeholders inside the
override are rejected by the placeholder allowlist.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

aligner_custom = ObjectAligner(
    {"type": "string", "score": "jaro"},
    generate_reasoning=True,
    reasoning_templates={
        "metric.perfect": "Perfect.",
        "item.mismatch":  '{indent}pred="{pred}" gold="{gold}" ({score_pct})\n',
    },
)
print(aligner_custom.metric("hello", "hello")["reasoning"])
# Perfect.
print(aligner_custom.metric("hello", "helo")["reasoning"])
# The predicted output scores overall 93%, let us align the predicted output to the gold and analyze the differences:
# pred="helo" gold="hello" (93%)
```

### Example 9: Loading templates from a TOML file

For larger overrides, keep your strings in a TOML file and load them with
`load_templates_from_toml`. The function returns a flat
`dict[str, str]` you pass straight into the constructor. See
[`api.md`](api.md#load_templates_from_toml) for the full
signature.

This example uses a different schema:

```python
from pathlib import Path
from object_aligner import ObjectAligner, load_templates_from_toml

# overrides.toml contents (flat or nested-table style both accepted):
#   "metric.perfect" = "Perfect."
#   "item.mismatch"  = '{indent}pred="{pred}" gold="{gold}" ({score_pct})\n'
overrides = load_templates_from_toml(Path("overrides.toml"))
aligner_toml = ObjectAligner(
    {"type": "string", "score": "jaro"},
    generate_reasoning=True,
    reasoning_templates=overrides,
)
```

---

## API reference

Canonical signatures, parameter descriptions, and field tables live in
[`api.md`](api.md). This section only links into them and documents the
chapter-specific template-key table that has no natural home there.

- [`ObjectAligner`](api.md#objectaligner) — constructor accepts
  `generate_reasoning` and `reasoning_templates`.
- [`ObjectAligner.metric()`](api.md#objectalignermetric) — `generate_reasoning`
  per-call override (`None` defers to the constructor flag).
- [`load_templates_from_toml()`](api.md#load_templates_from_toml) — for
  larger or translated reasoning-template sets.

### Template keys

| Key | Placeholders | When emitted |
|-----|--------------|--------------|
| `metric.perfect` | *(none)* | Overall score is exactly `1.0` |
| `metric.imperfect_intro` | `score`, `score_pct` | Overall score is `< 1.0` |
| `item.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | Primitive leaf, matched |
| `item.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | Primitive leaf, mismatched |
| `ref.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | `ref` leaf, matched |
| `ref.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | `ref` leaf, mismatched |
| `id.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | `idScope` leaf, matched (defaults to empty) |
| `id.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | `idScope` leaf, mismatched (defaults to empty) |
| `list.match` | `indent`, `score`, `score_pct` | List, all children matched |
| `list.mismatch` | `indent`, `score`, `score_pct` | List, some children mismatched |
| `list.excess` | `indent`, `gold`, `pred`, `score`, `score_pct` | Predicted list item with no gold counterpart |
| `list.missing` | `indent`, `gold`, `pred`, `score`, `score_pct` | Gold list item with no predicted counterpart |
| `dict.match` | `indent`, `score`, `score_pct` | Dict, all children matched |
| `dict.mismatch` | `indent`, `score`, `score_pct` | Dict, some children mismatched |
| `dict.key.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | Dict child, key part matched |
| `dict.key.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | Dict child, key part mismatched |
| `dict.value.prefix` | `indent` | Emitted before each dict child's value row |
| `validation.error` | `path`, `message` | `pred` failed schema validation |

The template-key allowlist and per-key placeholder set live in
`_TEMPLATE_PLACEHOLDERS` in `src/object_aligner/object_aligner.py` and are
self-validated at import time against `reasoning.toml`.

---

## Caveats

- **Output is a string, not structured data.** Tooling that needs
  per-path information should use [`attribute()`](attribution.md) or
  [`feedback()`](feedback.md) instead.
- **No LLM involved.** Reasoning is a fixed projection of the match tree —
  reproducible bit-for-bit across runs and machines.
- **`id.match` / `id.mismatch` are intentionally empty** in the defaults
  so that `id` fields don't pollute prose with bookkeeping noise. If your
  use case wants id rows visible, override those two keys.
- **Template stability.** Keys may be **added** when the underlying
  match-kind taxonomy grows; renames and removals are not permitted within
  a major version.
- **Recursion cost.** The walk is $O(N)$ in the number of match-tree
  nodes — cheap, but the string itself can grow large on deeply nested
  inputs.

---

## See also

- [`metric.md`](metric.md) — the `metric()` function this output rides on.
- [`feedback.md`](feedback.md) — prescriptive, top-K feedback string for
  prompt-optimizer reflection slots.
- [`attribution.md`](attribution.md) — structured per-path decomposition.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

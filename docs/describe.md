# 7. Plain-English Description

[Docs](index.md) › Plain-English Description

`metric()` answers *how well* a prediction matches the gold. `attribute()`
answers *where the deficit lives*. `repair()` answers *what to change*.
`feedback()` produces a prompt-optimizer-shaped feedback string.
**`describe()`** complements them with a different output: a
**plain-English, human-readable walk of the alignment tree** — useful
for spot-checking evaluation runs, surfacing failures in dashboards, or
attaching a one-shot explanation alongside the score.

The description is **deterministic and template-based — there is no LLM
involved**. The alignment is already performed by `align()`; the
description layer is a deterministic projection of the resulting match
tree onto a text surface (default) or onto a flat list of structured
per-node entries (`"json"` style).

This page documents the API and shows worked examples for every match
kind and every render branch.

---

## Quickstart

```python
from object_aligner import ObjectAligner

schema = {"type": "string", "score": "jaro"}
aligner = ObjectAligner(schema)

dr = aligner.describe("hello", "helo")
print(dr.score)
# 0.9333333333333333
print(dr.text)
# The predicted output scores overall 93%, let us align the predicted output to the gold and analyze the differences:
# The predicted value "helo" does not match the gold "hello" (score=93%).
```

`aligner.describe(gold, pred)` returns a `DescriptionResult` whose
`.text` is the rendered prose and whose `.entries` is a
traversal-ordered tuple of `DescriptionEntry` records (one per visited
match-tree node).

The same output is reachable through `metric()`:

```python
aligner = ObjectAligner(schema, generate_description=True)
result = aligner.metric("hello", "helo")
# result["score"]       = 0.9333
# result["description"] = "The predicted output scores overall 93% …"
```

You can also enable it on a per-call basis without flipping the
constructor flag:

```python
aligner = ObjectAligner(schema)
aligner.metric("hello", "helo")                              # {"score": 0.93}
aligner.metric("hello", "helo", generate_description=True)   # ... + description
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
aligner = ObjectAligner(movie_schema)
gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
pred = {"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]}
```

Each example states explicitly when it builds its own schema/aligner
instead of reusing these.

---

## The model

A `DescriptionResult` is computed by:

1. **Align** `gold` and `pred` once.
2. **Walk** the resulting match tree in schema-declared order, emitting
   one `DescriptionEntry` per visited node and rendering one prose line
   per node via the matching `describe.<kind>.<outcome>` template.
3. **Combine** the lines into an indented string under
   `describe.intro.imperfect`; for a perfect overall score, short-circuit
   to the single `describe.intro.perfect` line.
4. In `"json"` style, skip step 3: `text` is `""` and the consumer reads
   `entries` directly.

Each rendered line for a leaf carries the local pair `(gold, pred)` and
the local `score` (as a percentage); container nodes (dicts, lists) emit
a header line carrying the aggregate score.

The walk is $O(N)$ in the number of match-tree nodes — cheap, but the
text and entries lists themselves can grow large on deeply nested
inputs.

---

## Examples

### Example 1: Perfect match

A perfect score short-circuits the walk and emits a single line — no
tree walk, no per-leaf detail, no entries.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

aligner_perfect = ObjectAligner({"type": "string", "score": "jaro"})
dr = aligner_perfect.describe("hello", "hello")
print(dr.text)
# The predicted output perfectly matches the gold.
print(dr.entries)
# ()
```

### Example 2: Primitive mismatch

For a primitive root the intro line is emitted, then a single
`describe.item.mismatch` line with no indent.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

aligner_str = ObjectAligner({"type": "string", "score": "jaro"})
print(aligner_str.describe("hello", "helo").text)
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
aligner_list = ObjectAligner(list_schema)
print(aligner_list.describe([1, 2, 4], [2, 3]).text)
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
description — the importance affects scoring, not rendering.

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
aligner_dict = ObjectAligner(dict_schema)
print(aligner_dict.describe(
    {"name": "Alice", "age": 30},
    {"name": "Alic",  "age": 31},
).text)
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
print(aligner.describe(gold, pred).text)
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

The exact score depends on the chosen metric — what matters for this
page is that the description shape parallels the match tree.

### Example 6: Referential alignment (`idScope` / `ref`)

`id` rows are intentionally rendered as the empty string — that's why
the `VALUE = ` lines below appear empty for `id` properties. References
render through `describe.ref.match` / `describe.ref.mismatch`.

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
aligner_ref = ObjectAligner(ref_schema)
gold = {"people": [{"id": "p1", "name": "Alice"}, {"id": "p2", "name": "Bob"}], "best_friend": "p1"}
pred = {"people": [{"id": "x",  "name": "Aliec"}, {"id": "y",  "name": "Bob"}], "best_friend": "y"}
print(aligner_ref.describe(gold, pred).text)
# (excerpt)
# ...
#   KEY = The predicted key "best_friend" exactly matches the gold.
#   VALUE = The predicted reference "y" does not match the gold reference "p1" under the inferred id mapping (score=0%).
```

### Example 7: `style="json"` — structured entries, empty text

For programmatic consumers (e.g. dashboards, JSON exporters) the
`"json"` style emits an empty `.text` and populates `.entries`. Each
`DescriptionEntry` exposes `path`, `depth`, `match_kind`, `outcome`,
`score`, and `text` (the per-node prose).

Reuses `movie_schema`, `aligner`, `gold`, `pred` from the shared setup.

```python
dr = aligner.describe(gold, pred, style="json")
print("text:", repr(dr.text))
print("style:", dr.style)
for e in dr.entries[:6]:
    print(f"  depth={e.depth} kind={e.match_kind:<5} outcome={e.outcome:<8} path={e.path!r} score={e.score:.2f}")
```

```
text: ''
style: json
  depth=0 kind=dict  outcome=mismatch path='' score=0.67
  depth=1 kind=key   outcome=match    path='/title' score=1.00
  depth=1 kind=item  outcome=mismatch path='/title' score=0.97
  depth=1 kind=key   outcome=match    path='/year' score=1.00
  depth=1 kind=item  outcome=mismatch path='/year' score=0.00
  depth=1 kind=key   outcome=match    path='/genres' score=1.00
```

For Hungarian-aligned (`order: "align"`) list children the path stops
at the list itself — see the
[caveat below](#reorder-list-paths-point-at-the-list-not-an-index).

### Example 8: Validation errors {#validation-errors}

If `pred` fails schema validation, both `describe()` and
`metric(..., generate_description=True)` short-circuit to a score of
`0.0` and emit the `describe.validation_error` template instead of a
tree walk.

```python
strict_schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "keyScore": "exact",
}
strict = ObjectAligner(strict_schema, generate_description=True)
print(strict.metric(
    {"name": "Alice", "age": 30},
    {"name": "Alice"},  # missing age → validation fail
))
# {'score': 0.0, 'description': "JSON Schema validation failed for path=\"/\". Error message: 'age' is a required property."}
```

### Example 9: Customizing a single template

Pass a partial dict via `description_templates`. Any key you do not
provide keeps its packaged default. Unknown keys raise a `ValueError`
at construction time (typos caught early), and unknown placeholders
inside the override are rejected by the placeholder allowlist.

This example uses a different schema:

```python
from object_aligner import ObjectAligner

aligner_custom = ObjectAligner(
    {"type": "string", "score": "jaro"},
    generate_description=True,
    description_templates={
        "describe.intro.perfect": "Perfect.",
        "describe.item.mismatch": '{indent}pred="{pred}" gold="{gold}" ({score_pct})\n',
    },
)
print(aligner_custom.metric("hello", "hello")["description"])
# Perfect.
print(aligner_custom.metric("hello", "helo")["description"])
# The predicted output scores overall 93%, let us align the predicted output to the gold and analyze the differences:
# pred="helo" gold="hello" (93%)
```

### Example 10: Loading templates from a TOML file

For larger overrides, keep your strings in a TOML file and load them
with `load_templates_from_toml`. The function returns a flat
`dict[str, str]` you pass straight into the constructor. See
[`api.md`](api.md#load_templates_from_toml) for the full signature.

This example uses a different schema:

```python
from pathlib import Path
from object_aligner import ObjectAligner, load_templates_from_toml

# overrides.toml contents (flat or nested-table style both accepted):
#   "describe.intro.perfect" = "Perfect."
#   "describe.item.mismatch" = '{indent}pred="{pred}" gold="{gold}" ({score_pct})\n'
overrides = load_templates_from_toml(Path("overrides.toml"))
aligner_toml = ObjectAligner(
    {"type": "string", "score": "jaro"},
    generate_description=True,
    description_templates=overrides,
)
```

### Example 11: `metric(generate_description=True | "full")`

Reuses `aligner`, `gold`, `pred` from the [shared
setup](#shared-setup-for-the-examples). Symmetric with
`generate_feedback`:

```python
result = aligner.metric(gold, pred, generate_description=True)
print(type(result["description"]).__name__, "→", result["description"][:60], "...")
```

```
str → The predicted output scores overall 67%, let us align the p ...
```

For programmatic consumers, pass `generate_description="full"`:

```python
result = aligner.metric(gold, pred, generate_description="full")
print(type(result["description"]).__name__, "→ keys:", sorted(result["description"]))
```

```
dict → keys: ['entries', 'score', 'style', 'text']
```

The dict is the same shape as `DescriptionResult.to_dict()` — basic
types only, JSON-serializable as long as the underlying `gold` / `pred`
values are.

### Example 12: `describe_from_match` (cached-alignment fast path)

If you've already computed a match tree (e.g. for attribution, repair,
or feedback), reuse it:

```python
match_tree = aligner.align(gold, pred)
dr = aligner.describe_from_match(match_tree)
```

No `mappings` argument is required: describe does not need referential
state — the match tree already encodes the resolved ref scores.

---

## API reference

Canonical signatures, parameter descriptions, and field tables live in
[`api.md`](api.md). This section only links into them and documents the
chapter-specific template-key table that has no natural home there.

- [`ObjectAligner.describe()`](api.md#objectalignerdescribe) — renders
  a description directly from `(gold, pred)`. Aligns once internally;
  no LLM.
- [`ObjectAligner.describe_from_match()`](api.md#objectalignerdescribe_from_match)
  — same emission logic against an already-computed match tree.
- [`render_description()`](api.md#render_description) — low-level
  functional entry; takes an already-computed match tree.
- [`load_templates_from_toml()`](api.md#load_templates_from_toml) — for
  larger or translated description-template sets.
- [`DescriptionResult`](api.md#descriptionresult) and
  [`DescriptionEntry`](api.md#descriptionentry) — result types.
  Iterable and indexable over `entries`; `str(result) == result.text`;
  call `.to_dict()` for a basic-types dict.
- Constructor kwargs `generate_description`, `description_templates`,
  `description_style`, and the per-call
  `metric(generate_description=...)` override (`None` / `False` /
  `True` / `"full"`) — see [`ObjectAligner`](api.md#objectaligner) and
  [`ObjectAligner.metric()`](api.md#objectalignermetric).

### Template keys

| Key | Placeholders | When emitted |
|-----|--------------|--------------|
| `describe.intro.perfect` | *(none)* | Overall score is exactly `1.0` |
| `describe.intro.imperfect` | `score`, `score_pct` | Overall score is `< 1.0` |
| `describe.item.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | Primitive leaf, matched |
| `describe.item.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | Primitive leaf, mismatched |
| `describe.ref.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | `ref` leaf, matched |
| `describe.ref.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | `ref` leaf, mismatched |
| `describe.id.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | `idScope` leaf, matched (defaults to empty) |
| `describe.id.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | `idScope` leaf, mismatched (defaults to empty) |
| `describe.list.match` | `indent`, `score`, `score_pct` | List, all children matched |
| `describe.list.mismatch` | `indent`, `score`, `score_pct` | List, some children mismatched |
| `describe.list.excess` | `indent`, `gold`, `pred`, `score`, `score_pct` | Predicted list item with no gold counterpart |
| `describe.list.missing` | `indent`, `gold`, `pred`, `score`, `score_pct` | Gold list item with no predicted counterpart |
| `describe.dict.match` | `indent`, `score`, `score_pct` | Dict, all children matched |
| `describe.dict.mismatch` | `indent`, `score`, `score_pct` | Dict, some children mismatched |
| `describe.dict.key.match` | `indent`, `gold`, `pred`, `score`, `score_pct` | Dict child, key part matched |
| `describe.dict.key.mismatch` | `indent`, `gold`, `pred`, `score`, `score_pct` | Dict child, key part mismatched |
| `describe.dict.value.prefix` | `indent` | Emitted before each dict child's value row |
| `describe.validation_error` | `path`, `message` | `pred` failed schema validation |

The template-key allowlist and per-key placeholder set live in
`_DESCRIPTION_PLACEHOLDERS` in `src/object_aligner/describe.py` and are
self-validated at import time against `describe.toml`.

The default templates are exported as `DEFAULT_DESCRIPTION_TEMPLATES`
(`from object_aligner import DEFAULT_DESCRIPTION_TEMPLATES`).

---

## Caveats

### Reorder-list paths point at the list, not an index

For `order: "align"` lists, every `DescriptionEntry` for a list child
shares the list path (e.g. `/items`) rather than carrying a specific
index, because Hungarian-matched indices aren't stable into the
original `pred` array. The same convention is used by `feedback()` and
`repair()` — see
[the corresponding `repair.md` caveat](repair.md#reorder-list-paths-are-not-rfc-6902-strict).
Fixed-position lists (`order: "fixed"`, and any prefix items) do carry
the strict `/index` path.

### No LLM

The describe module never calls an LLM — every rendered line is a
deterministic template substitution. If you want LLM-rewritten prose,
pass `dr.text` to your LLM as input. The library itself stays
deterministic.

### `id.match` / `id.mismatch` are intentionally empty

In the defaults so that `id` fields don't pollute prose with
bookkeeping noise. If your use case wants id rows visible, override
those two keys.

### Stability across versions

These fields are part of the **stable API** and won't change shape
without a major version bump:

- `DescriptionEntry`: `path`, `depth`, `match_kind`, `outcome`,
  `score`, `text`.
- `DescriptionResult`: `score`, `text`, `entries`, `style`.
- Template key set: keys may be **added** (with sensible English
  defaults) but not renamed or removed.

The wording inside `DescriptionResult.text` (and inside individual
`DescriptionEntry.text` values) may evolve between minor versions as
the English defaults are tuned. If you depend on exact strings,
override the relevant templates.

---

## See also

- [`metric.md`](metric.md) — the `metric()` function this output rides on.
- [`feedback.md`](feedback.md) — prescriptive, top-K feedback string for
  prompt-optimizer reflection slots.
- [`attribution.md`](attribution.md) — structured per-path decomposition.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

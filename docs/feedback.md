# 11. Prompt-Optimizer Feedback

[Docs](index.md) › Prompt-Optimizer Feedback

`metric()` answers *how well* a prediction matches the gold. `attribute()`
answers *where the deficit lives*. `repair()` answers *what to change*.
**`feedback()`** takes the final step: it turns those signals into a
**top-K ranked, prescriptive, optimizer-shaped feedback string** suitable
for pasting into the reflection slot of a DSPy / GEPA / TextGrad-style
prompt optimizer.

The feedback is **deterministic and template-based — there is no LLM
involved**. That is the design constraint: the optimization-relevant
analysis is already performed by `align()` + `repair()`, and the
feedback layer is a deterministic projection of that work onto a text
surface.

This page documents the API and shows worked examples for every op kind.

---

## Quickstart

```python
from object_aligner import ObjectAligner

schema = {
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

aligner = ObjectAligner(schema)
fb = aligner.feedback(
    gold={"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]},
    pred={"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]},
)
print(fb.text)
```

Output:

```
The prediction scored 0.67 (deficit 0.33). Top 3 of 3 fix locations:
1. /year: expected 1999, got 2000. Fixing this recovers +0.250.
2. inside list /genres: replace item 'Adventure' with 'Action'. Fixing this recovers +0.062.
3. /title: expected 'The Matrix', got 'The Matrx'. Fixing this recovers +0.017.
Focus on primitive-value errors — they account for 81% of the deficit shown.
```

The same string is available through `metric()`:

```python
result = aligner.metric(gold, pred, generate_feedback=True)
# result["score"]    = 0.6708
# result["feedback"] = "The prediction scored 0.67 …"
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

A `FeedbackResult` is computed by:

1. **Align** `gold` and `pred` once.
2. **Generate** the ranked `RepairResult` (sorted descending by
   `score_delta`).
3. **Filter** ops with `score_delta < min_score_delta`. Key-rename pairs
   are kept atomically — both halves pass or neither does.
4. **Slice** the first `top_k` ops (default `5`).
5. **Render** each op via its `feedback.op.<kind>` template.
6. **Synthesize** a trailing sentence: if one op-kind bucket accounts for
   ≥ `dominant_fraction_threshold` of the displayed deficit
   (default `0.60`), emit "Focus on X errors …"; otherwise emit "The
   deficit is spread across multiple issue types …".

Each rendered line carries:

$$
\mathrm{score\_delta}(\mathrm{op}) \;=\; c_w \cdot (1 - s_w)
$$

— the same number `attribute()` reports for that location (see
[Attribution](attribution.md) and [Repair](repair.md) for the derivation).

Under the alignment's fixed Hungarian/DP assignment, before filtering,
$\sum \mathrm{score\_delta} = 1 - S$. After `top_k` and `min_score_delta`
filtering, that exact identity no longer holds; the dropped contribution
is the difference between `FeedbackResult.n_total_ops`-worth of deltas
and the displayed `sum(e.score_delta for e in entries)`. The flag
`FeedbackResult.truncated` is `True` whenever filtering dropped any op
with positive delta.

---

## Examples

Every code block has been executed; the outputs are real.

### Example 1 — Primitive value mismatch

```python
from object_aligner import ObjectAligner

aligner = ObjectAligner({"type": "string", "score": "jaro"})
print(aligner.feedback("hello", "hallo").text)
```

```
The prediction scored 0.87 (deficit 0.13). Top 1 of 1 fix locations:
1. : expected 'hello', got 'hallo'. Fixing this recovers +0.133.
Focus on primitive-value errors — they account for 100% of the deficit shown.
```

The empty `path` (`""`) at the start of the line is the RFC 6901
whole-document pointer — used when the root of the schema is the
primitive itself. For nested schemas the path is meaningful (e.g.
`/user/name`).

### Example 2 — Missing dict key (`key_add`)

```python
schema = {
    "type": "object", "keyScore": "exact",
    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
}
aligner = ObjectAligner(schema)
print(aligner.feedback({"a": "x", "b": 1}, {"a": "x"}).text)
```

```
The prediction scored 0.50 (deficit 0.50). Top 1 of 1 fix locations:
1. /b: missing key "b" with value 1. Adding it recovers +0.500.
Focus on missing-key errors — they account for 100% of the deficit shown.
```

### Example 3 — Extraneous dict key (`key_remove`)

Reuses the `aligner` (and schema) from Example 2.

```python
print(aligner.feedback({"a": "x"}, {"a": "x", "b": 1}).text)
```

```
The prediction scored 0.50 (deficit 0.50). Top 1 of 1 fix locations:
1. /b: extraneous key "b" (value 1). Removing it recovers +0.500.
Focus on extraneous-key errors — they account for 100% of the deficit shown.
```

### Example 4 — Fuzzy key rename (single-line pair)

```python
schema = {
    "type": "object", "keyScore": "jaro",
    "properties": {"phoneNumber": {"type": "string"}},
}
aligner = ObjectAligner(schema)
print(aligner.feedback({"phoneNumber": "555-1234"}, {"phone": "555-1234"}).text)
```

```
The prediction scored 0.91 (deficit 0.09). Top 1 of 2 fix locations:
1. rename key "phone" -> "phoneNumber" at /phoneNumber (value '555-1234'). Fixing this recovers +0.091.
Focus on key-rename errors — they account for 100% of the deficit shown.
```

`Top 1 of 2` because the rename produces two `RepairOp`s (`key_rename_add`
and `key_rename_remove`), but the `_remove` template is empty by default
so the pair renders as one visible line. Both ops are present in
`FeedbackResult.entries`; only one carries text.

### Example 5 — Reorder list, missing/excess items

```python
schema = {
    "type": "object", "keyScore": "exact",
    "properties": {
        "items": {
            "type": "array", "order": "align",
            "items": {"type": "string", "score": "jaro"},
        },
    },
}
aligner = ObjectAligner(schema)
print(aligner.feedback(
    {"items": ["alpha", "beta", "gamma"]},
    {"items": ["alpha", "beta"]},
).text)
```

```
The prediction scored 0.83 (deficit 0.17). Top 1 of 1 fix locations:
1. /items: list is missing item 'gamma'. Adding it recovers +0.167.
Focus on missing-list-item errors — they account for 100% of the deficit shown.
```

`/items` is the *list path*, not a specific index — see
[the caveat about reorder-list paths](#reorder-list-paths-point-at-the-list-not-an-index)
and the corresponding section in [`repair.md`](repair.md#reorder-list-paths-are-not-rfc-6902-strict).

### Example 6 — Fixed list, missing/excess items

```python
aligner = ObjectAligner({"type": "array", "items": {"type": "string", "score": "jaro"}})
print(aligner.feedback(["a", "b", "c"], ["a", "b"]).text)
```

```
The prediction scored 0.67 (deficit 0.33). Top 1 of 1 fix locations:
1. /2: missing list item 'c'. Adding it recovers +0.333.
Focus on missing-list-item errors — they account for 100% of the deficit shown.
```

In a fixed (non-reorder) list, the path is the exact index (`/2`), and
the op kinds are `list_item_add` / `list_item_remove`.

### Example 7 — Reference fix (`ref_fix`)

```python
schema = {
    "type": "object", "keyScore": "exact",
    "properties": {
        "entities": {
            "type": "array", "order": "align",
            "items": {
                "type": "object", "keyScore": "exact",
                "properties": {
                    "id":   {"type": "string", "idScope": "entity"},
                    "name": {"type": "string", "score": "exact"},
                },
            },
        },
        "primary": {"type": "string", "ref": "entity"},
    },
}
aligner = ObjectAligner(schema)
print(aligner.feedback(
    {"entities": [{"id": "g1", "name": "Alice"}, {"id": "g2", "name": "Bob"}],
     "primary": "g1"},
    {"entities": [{"id": "p1", "name": "Alice"}, {"id": "p2", "name": "Bob"}],
     "primary": "p2"},
).text)
```

```
The prediction scored 0.75 (deficit 0.25). Top 1 of 1 fix locations:
1. /primary: wrong reference (expected 'g1', got 'p2'). Fixing this recovers +0.250.
Focus on reference errors — they account for 100% of the deficit shown.
```

See [Referential Alignment](referential.md) for the underlying
`idScope` / `ref` semantics.

### Example 8 — Style presets: `gepa` vs `compact` vs `json`

Reuses `aligner`, `gold`, `pred` from the [shared setup](#shared-setup-for-the-examples).
Same call, three styles:

```python
fb = aligner.feedback(gold, pred, style="gepa")
print(fb.text)
```

```
The prediction scored 0.67 (deficit 0.33). Top 3 of 3 fix locations:
1. /year: expected 1999, got 2000. Fixing this recovers +0.250.
2. inside list /genres: replace item 'Adventure' with 'Action'. Fixing this recovers +0.062.
3. /title: expected 'The Matrix', got 'The Matrx'. Fixing this recovers +0.017.
Focus on primitive-value errors — they account for 81% of the deficit shown.
```

```python
fb = aligner.feedback(gold, pred, style="compact", include_synthesis_line=False)
print(fb.text)
```

```
Score 0.67. Top 3/3 fixes:
1. /year: 1999 (got 2000) [+0.250]
2. /genres: replace 'Adventure'->'Action' [+0.062]
3. /title: 'The Matrix' (got 'The Matrx') [+0.017]
```

```python
fb = aligner.feedback(gold, pred, style="json", include_metadata=True)
print("text:", repr(fb.text))
print("error_breakdown:", fb.error_breakdown)
for e in fb.entries:
    print(f"  rank={e.rank} kind={e.op_kind} path={e.path} delta={e.score_delta:.4f}")
```

```
text: ''
error_breakdown: {'primitive_replace': 0.2667, 'primitive_replace_reorder': 0.0625}
  rank=1 kind=primitive_replace path=/year delta=0.2500
  rank=2 kind=primitive_replace_reorder path=/genres delta=0.0625
  rank=3 kind=primitive_replace path=/title delta=0.0167
```

In `"json"` style, both `FeedbackResult.text` and every
`FeedbackEntry.text` are empty; the structured fields are populated.
Consumers serialize them as they wish — e.g.
`FeedbackResult.to_dict()` produces a basic-types dict.

### Example 9 — Customizing templates (Czech translation recipe)

Reuses `movie_schema`, `gold`, `pred` from the [shared setup](#shared-setup-for-the-examples).

Templates are validated at construction time: unknown keys and unknown
placeholders raise `ValueError` before any render call. The same
mechanism is the localization story — override per-key with strings in
your target language.

```python
cs_templates = {
    "feedback.intro.imperfect":
        "Predikce dosáhla skóre {score:.2f} (chyba {deficit:.2f}). "
        "Nejdůležitějších {n_shown} z {n_total} oprav:\n",
    "feedback.op.primitive_replace":
        "{rank}. {path}: očekáváno {gold}, dostáno {pred}. "
        "Oprava získá +{score_delta:.3f}.",
    "feedback.synthesis.single_dominant":
        "\nZaměřte se na chyby typu {dominant_kind_human} — tvoří "
        "{dominant_fraction_pct:.0f}% zobrazené chyby.",
    "feedback.synthesis.mixed":
        "\nChyby jsou rozloženy mezi typy ({top_kinds}).",
}
cs_aligner = ObjectAligner(movie_schema, feedback_templates=cs_templates)
print(cs_aligner.feedback(gold, pred).text)
```

```
Predikce dosáhla skóre 0.67 (chyba 0.33). Nejdůležitějších 3 z 3 oprav:
1. /year: očekáváno 1999, dostáno 2000. Oprava získá +0.250.
2. inside list /genres: replace item 'Adventure' with 'Action'. Fixing this recovers +0.062.
3. /title: očekáváno 'The Matrix', dostáno 'The Matrx'. Oprava získá +0.017.
Zaměřte se na chyby typu primitive-value — tvoří 81% zobrazené chyby.
```

Templates you don't override fall through to the English defaults (see
the second `/genres` line above — `feedback.op.primitive_replace_reorder`
wasn't translated in this example).

### Example 9.5 — Loading templates from a TOML file

For large or translated template sets, keep the strings in a TOML file
rather than inlining a dict. The shipped helper
`load_templates_from_toml(path)` returns a dict suitable for passing
directly as `feedback_templates=` or `description_templates=`:

```toml
# my_feedback_cs.toml
"feedback.intro.imperfect" = "Predikce dosáhla skóre {score:.2f} (chyba {deficit:.2f}). Nejdůležitějších {n_shown} z {n_total} oprav:\n"
"feedback.op.primitive_replace" = "{rank}. {path}: očekáváno {gold}, dostáno {pred}. Oprava získá +{score_delta:.3f}."
"feedback.synthesis.single_dominant" = "\nZaměřte se na chyby typu {dominant_kind_human} — tvoří {dominant_fraction_pct:.0f}% zobrazené chyby."
"feedback.synthesis.mixed" = "\nChyby jsou rozloženy mezi typy ({top_kinds})."
```

```python
from object_aligner import ObjectAligner, load_templates_from_toml

templates = load_templates_from_toml("my_feedback_cs.toml")
cs_aligner = ObjectAligner(movie_schema, feedback_templates=templates)
print(cs_aligner.feedback(gold, pred).text)
```

(Reuses `movie_schema`, `gold`, `pred` from the
[shared setup](#shared-setup-for-the-examples).)

The helper accepts both flat (recommended, matches the package files)
and nested-table TOML styles:

```toml
# Flat (recommended)
"feedback.op.key_add" = "..."

# Nested tables also work — flattened back to the dotted key
[feedback.op]
key_add = "..."
```

Both forms produce the same `{"feedback.op.key_add": "..."}` entry.
Construction-time validation (unknown keys, bad placeholders) still
runs when `ObjectAligner(...)` consumes the loaded dict — bad files
fail at construction, not at first render.

The packaged defaults themselves live as TOML data under
`src/object_aligner/templates/` (`describe.toml`, `feedback.toml`,
`feedback.compact.toml`); use those as the canonical reference when
authoring overrides.

### Example 10 — Custom `value_formatter` for long blobs

The default `value_formatter` truncates `repr(value)` to 80 characters
with a trailing `…`. For tighter or wider control, pass your own:

```python
from object_aligner import ObjectAligner, render_feedback

schema = {
    "type": "object", "keyScore": "exact",
    "properties": {
        "a": {"type": "string"},
        "b": {"type": "integer", "score": "exact"},
    },
}
aligner = ObjectAligner(schema)

gold = {"a": "x" * 200, "b": 1}
pred = {"a": "y" * 200, "b": 2}
rep = aligner.repair(gold, pred)
fb = render_feedback(
    rep,
    value_formatter=lambda v: repr(v)[:20] + "..." if len(repr(v)) > 20 else repr(v),
)
print(fb.text)
```

```
The prediction scored 0.62 (deficit 0.38). Top 2 of 2 fix locations:
1. /a: expected 'xxxxxxxxxxxxxxxxxxx..., got 'yyyyyyyyyyyyyyyyyyy.... Fixing this recovers +0.250.
2. /b: expected 1, got 2. Fixing this recovers +0.125.
Focus on primitive-value errors — they account for 100% of the deficit shown.
```

### Example 11 — `metric(generate_feedback=True | "full")`

Reuses `aligner`, `gold`, `pred` from the [shared setup](#shared-setup-for-the-examples).

```python
result = aligner.metric(gold, pred, generate_feedback=True)
print(type(result["feedback"]).__name__, "→", result["feedback"][:60], "...")
```

```
str → The prediction scored 0.67 (deficit 0.33). Top 3 of 3 fi ...
```

For programmatic consumers (e.g. DSPy adapters that want the structured
entries), pass `generate_feedback="full"`:

```python
result = aligner.metric(gold, pred, generate_feedback="full")
print(type(result["feedback"]).__name__, "→ keys:", sorted(result["feedback"]))
```

```
dict → keys: ['entries', 'error_breakdown', 'n_total_ops', 'score', 'style', 'text', 'truncated']
```

The dict is the same shape as `FeedbackResult.to_dict()` — basic types
only, JSON-serializable as long as the underlying `gold` / `pred`
values are.

### Example 12 — `feedback_from_match` (cached-alignment fast path)

Reuses `aligner`, `gold`, `pred` from the [shared setup](#shared-setup-for-the-examples).
If you've already computed a match tree (e.g. for attribution or
describing), reuse it:

```python
match_tree = aligner.align(gold, pred)
fb = aligner.feedback_from_match(match_tree, gold, pred, mappings={})
```

`mappings` is the `ctx.current_mappings` dict from the align-time
context — required only if your schema uses `ref` fields. An empty dict
is fine otherwise.

### Example 13 — `granularity="subtree"` for whole-subtree replacement

```python
schema = {
    "type": "object", "keyScore": "exact",
    "properties": {
        "user": {
            "type": "object", "keyScore": "exact",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"},
            },
        },
        "status": {"type": "string", "score": "exact"},
    },
}
aligner = ObjectAligner(schema)
print(aligner.feedback(
    {"user": {"name": "Alice", "age": 30}, "status": "active"},
    {"user": {"name": "Alic",  "age": 31}, "status": "active"},
    granularity="subtree",
).text)
```

```
The prediction scored 0.96 (deficit 0.04). Top 1 of 1 fix locations:
1. : subtree differs. Replacing it recovers +0.035.
Focus on subtree errors — they account for 100% of the deficit shown.
```

The empty path (`""`) here is the whole-document root: in `"subtree"`
mode, the outermost imperfect node emits the op. Use
`granularity="all"` to also surface inner-subtree and leaf-level ops.

---

## API reference

Canonical signatures, parameter descriptions, and field tables live in
[`api.md`](api.md). This section only points into them and documents the
feedback-specific surface (template keys, style presets) that has no
natural home there.

- [`ObjectAligner.feedback()`](api.md#objectalignerfeedback) — renders
  the top-K prescriptive feedback string. Aligns once internally; no
  LLM.
- [`ObjectAligner.feedback_from_match()`](api.md#objectalignerfeedback_from_match)
  — same emission logic against an already-computed match tree.
- [`render_feedback()`](api.md#render_feedback) — low-level functional
  entry; takes an already-computed `RepairResult`.
- [`load_templates_from_toml()`](api.md#load_templates_from_toml) — for
  large or translated template sets.
- [`FeedbackResult`](api.md#feedbackresult) and
  [`FeedbackEntry`](api.md#feedbackentry) — result types. Iterable and
  indexable over `entries`; `str(result) == result.text`; call
  `.to_dict()` for a basic-types dict.
- Constructor kwargs `generate_feedback`, `feedback_templates`,
  `feedback_style`, `dominant_fraction_threshold`, and the per-call
  `metric(generate_feedback=...)` override (`None` / `False` / `True` /
  `"full"`) — see [`ObjectAligner`](api.md#objectaligner) and
  [`ObjectAligner.metric()`](api.md#objectalignermetric).

### Template keys

The full table of 18 template keys and their placeholders.

| Key | Fires when | Placeholders |
|---|---|---|
| `feedback.intro.perfect` | `score == 1.0` | `score`, `score_pct` |
| `feedback.intro.imperfect` | otherwise, entries non-empty | `score`, `score_pct`, `deficit`, `deficit_pct`, `n_shown`, `n_total` |
| `feedback.op.primitive_replace` | primitive leaf, both sides present | `rank`, `path`, `gold`, `pred`, `score_delta`, `score_delta_pct` |
| `feedback.op.primitive_replace_reorder` | primitive replace inside reorder list | `rank`, `list_path`, `gold`, `pred`, `score_delta`, `score_delta_pct` |
| `feedback.op.key_add` | missing dict key | `rank`, `path`, `key`, `gold`, `score_delta`, `score_delta_pct` |
| `feedback.op.key_remove` | extraneous dict key | `rank`, `path`, `key`, `pred`, `score_delta`, `score_delta_pct` |
| `feedback.op.key_rename_add` | fuzzy key rename, add half | `rank`, `gold_path`, `pred_path`, `gold_key`, `pred_key`, `gold`, `score_delta`, `score_delta_pct` |
| `feedback.op.key_rename_remove` | fuzzy key rename, remove half (default empty — silenced) | `rank`, `pred_path`, `pred_key`, `pred` |
| `feedback.op.list_item_add` | fixed-list missing item | `rank`, `path`, `gold`, `score_delta`, `score_delta_pct` |
| `feedback.op.list_item_remove` | fixed-list excess item | `rank`, `path`, `pred`, `score_delta`, `score_delta_pct` |
| `feedback.op.list_item_missing` | reorder-list missing item | `rank`, `list_path`, `gold`, `score_delta`, `score_delta_pct` |
| `feedback.op.list_item_excess` | reorder-list excess item | `rank`, `list_path`, `pred`, `score_delta`, `score_delta_pct` |
| `feedback.op.ref_fix` | wrong reference | `rank`, `path`, `gold`, `pred`, `score_delta`, `score_delta_pct` |
| `feedback.op.subtree_replace` | `granularity != "leaf"` and subtree imperfect | `rank`, `path`, `score_delta`, `score_delta_pct` |
| `feedback.synthesis.single_dominant` | one op-kind ≥ `dominant_fraction_threshold` of displayed deficit | `dominant_kind`, `dominant_kind_human`, `dominant_fraction`, `dominant_fraction_pct` |
| `feedback.synthesis.mixed` | otherwise (and synthesis enabled) | `top_kinds` |
| `feedback.empty` | `top_k` filter empties everything | `score`, `score_pct` |
| `feedback.validation_error` | pred fails schema validation | `path`, `message` |

The default templates are exported as `DEFAULT_FEEDBACK_TEMPLATES`
(`from object_aligner import DEFAULT_FEEDBACK_TEMPLATES`).

---

## Caveats

### No LLM

The feedback module never calls an LLM — every rendered line is a
deterministic template substitution. If you want LLM-rewritten feedback,
pass `fb.text` to your LLM as input. The library itself stays
deterministic.

### Reorder-list paths point at the list, not an index

For `order: "align"` lists, `list_item_missing` / `list_item_excess` /
`primitive_replace_reorder` paths point at the *list* (e.g. `/items`)
rather than a specific index. This carries through from
[the corresponding `repair()` caveat](repair.md#reorder-list-paths-are-not-rfc-6902-strict)
because Hungarian-aligned positions aren't stable into the original
`pred` array.

### Deltas are approximate under re-pairing

Same as for `attribute()` and `repair()`: the deltas are exact under the
fixed Hungarian / DP assignment used by `align()`. In schemas where the
optimal pairing would change when one input is perturbed, the deltas are
a first-order linearization. See
[the `repair.md` caveat](repair.md#non-additivity-under-hungarian-dp-re-pairing)
for the underlying math.

### Stability across versions

These fields are part of the **stable API** and won't change shape
without a major version bump:

- `FeedbackEntry`: `op_kind`, `path`, `score_delta`, `pair_id`.
- `FeedbackResult`: `score`, `entries`, `style`, `truncated`,
  `n_total_ops`, `error_breakdown`.
- Template key set: keys may be **added** (with sensible English
  defaults) but not renamed or removed.

The wording inside `FeedbackResult.text` (and inside individual
`FeedbackEntry.text` values) may evolve between minor versions as we
tune the English defaults. If you depend on exact strings, override the
relevant templates.

### Soft size budget on `text`

A typical optimizer reflection slot has a few hundred tokens of context
budget. With the default `top_k=5` and 80-char value cap, the rendered
`text` is comfortably under 2 KB. If you bump `top_k` or render full
nested values, the text can grow — there's no hard cap, but at some
point an optimizer will truncate it.

---

## See also

- [`repair.md`](repair.md) — the structured op list this feedback rides on.
- [`attribution.md`](attribution.md) — per-path deficit decomposition.
- [`describe.md`](describe.md) — narrative walk of the match tree.
- [`metric.md`](metric.md) — the surrounding evaluation call.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

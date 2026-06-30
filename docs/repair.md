# 10. Scored JSON-Patch Repair

[Docs](index.md) › Scored JSON-Patch Repair

`attribute()` tells you *where* a candidate is wrong and *how much* each
location costs. **`repair()`** takes the next step: it emits a **ranked list
of structured operations** that, if applied to `pred`, would close some
fraction of the deficit $1 - S$.

The shape borrows from [RFC 6902 JSON Patch](https://datatracker.ietf.org/doc/html/rfc6902):
each op is one of `add` / `remove` / `replace`, has a JSON Pointer `path`, and
carries an estimated `score_delta`. There's also a finer `kind` discriminator
so consumers can dispatch on op semantics (primitive value vs. key add vs.
list reorder, etc.).

The v1 flavor is *approximate*: deltas come from the same tree-walk math as
[`attribute()`](attribution.md). The numbers are exact under the alignment's
fixed Hungarian/DP assignment; in schemas with re-pairing the deltas are a
first-order linearization (see [Caveats](#caveats)). An *exact* flavor —
apply each op, re-run `metric()`, measure the true delta — is planned future
work.

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
result = aligner.repair(
    gold={"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]},
    pred={"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]},
)
for op in result.ops:
    print(f"{op.op:7s} {op.path:14s} delta={op.score_delta:.4f}  value={op.value!r}")

patched = result.apply_to({"title": "The Matrx", "year": 2000,
                            "genres": ["Sci-Fi", "Adventure"]})
print("patched score =", aligner.metric(
    {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]},
    patched,
)["score"])
```

Output:

```
replace /year          delta=0.2500  value=1999
replace /genres        delta=0.0625  value='Action'
replace /title         delta=0.0167  value='The Matrix'
patched score = 1.0
```

Top op accounts for 76 % of the deficit (replace year). Applying all three
restores pred to the gold exactly.

---

## The model

`repair()` walks the post-`align()` match tree and, at every location where
gold and pred disagree, emits the operation that would close the gap. Each
op is annotated with:

$$
\mathrm{score\_delta}(\mathrm{op}) \;=\; c_w \cdot (1 - s_w)
$$

where $c_w$ is the effective weight of the node the op targets and $s_w$ is
its score — the same numbers `attribute()` reports. Under the alignment's
fixed Hungarian/DP assignment:

$$
\sum_{\mathrm{op}\in \mathrm{ops}}\mathrm{score\_delta}(\mathrm{op}) \;=\; 1 - S.
$$

`RepairResult` exposes this as the `.total_delta` and `.residual` fields
(residual = total - deficit).

---

## Shared setup for the examples

Every code block was executed; numbers below are real. The examples share
an import and a small printing helper that formats each op with its kind
and (when present) the `pair_id`, defined once here and reused throughout:

```python
from object_aligner import ObjectAligner

def show(result):
    header = f"score = {result.score:.4f}   deficit = {1 - result.score:.4f}"
    if abs(result.residual) > 1e-9:
        header += f"   residual = {result.residual:+.4f}"
    print(header)
    for op in result.ops:
        marker = f"  [{op.kind}]"
        if op.pair_id:
            marker += f" pair={op.pair_id}"
        path = op.path or "(root)"
        value_str = "" if op.op == "remove" else f"  value={op.value!r}"
        print(f"  {op.op:7s} {path:24s} delta={op.score_delta:.4f}{value_str}{marker}")
    for note in result.notes:
        print(f"  note: {note}")
```

Each example below builds its own schema inline but reuses `ObjectAligner`
and `show` from this block.

## Examples

### Example 1 — Primitive replace

```python
schema = {
    "type": "object", "keyScore": "exact",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer", "score": "exact"},
    },
}
r = ObjectAligner(schema).repair({"name": "Alice", "age": 30},
                                  {"name": "Alicia", "age": 29})
show(r)
```

```
score = 0.7056   deficit = 0.2944
  replace /age                     delta=0.2500  value=30  [primitive_replace]
  replace /name                    delta=0.0444  value='Alice'  [primitive_replace]
```

Two leaves are imperfect. Each gets a `replace` op carrying the gold value.

### Example 2 — Missing gold key (`key_add`)

```python
schema = {
    "type": "object", "keyScore": "exact",
    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
}
r = ObjectAligner(schema).repair({"a": "x", "b": "y"}, {"a": "x"})
show(r)
```

```
score = 0.5000   deficit = 0.5000
  add     /b                       delta=0.5000  value='y'  [key_add]
```

`b` is missing from pred. The single `add` op carries the gold key *and* its
value; its `score_delta` includes both the key-mismatch and value-mismatch
contributions for this slot.

### Example 3 — Excess pred key (`key_remove`)

```python
schema = {
    "type": "object", "keyScore": "exact",
    "additionalProperties": True,
    "properties": {"a": {"type": "string"}},
}
r = ObjectAligner(schema).repair({"a": "x"}, {"a": "x", "extra": "junk"})
show(r)
```

```
score = 0.5000   deficit = 0.5000
  remove  /extra                   delta=0.5000  [key_remove]
```

The pred has a key gold doesn't. A single `remove` does the job.

### Example 4 — Fuzzy key rename (`key_rename_*` pair)

```python
schema = {
    "type": "object",
    "keyScore": "jaro",
    "keyImportance": 1, "valueImportance": 1,
    "additionalProperties": True,
    "properties": {"userName": {"type": "string"}},
}
r = ObjectAligner(schema).repair({"userName": "alice"}, {"usrname": "alice"})
show(r)
```

```
score = 0.9345   deficit = 0.0655
  add     /userName                delta=0.0655  value='alice'  [key_rename_add] pair=repair_pair_1
  remove  /usrname                 delta=0.0000  [key_rename_remove] pair=repair_pair_1
```

Two ops share a `pair_id`. The `add` carries the full key-mismatch
contribution; the `remove` carries 0 (it does nothing alone — the pair must
be applied together). The `add`'s value is `'alice'` because the pred value
under the noisy key was already correct.

> **Why a pair instead of `move`?** v1 sticks to `add` / `remove` /
> `replace`. RFC 6902's `move` is the idiomatic op here and may ship as a
> future enhancement.

### Example 5 — Fixed list, missing item

```python
schema = {"type": "array", "items": {"type": "string", "score": "jaro"}}
r = ObjectAligner(schema).repair(["a", "b", "c"], ["a", "c"])
show(r)
```

```
score = 0.6667   deficit = 0.3333
  add     /1                       delta=0.3333  value='b'  [list_item_add]
```

For positional lists (no `order: "align"`), the path includes the index.

### Example 6 — Reorder list, semantic ops

```python
schema = {"type": "array", "items": {"type": "string", "score": "jaro"},
          "order": "align"}
r = ObjectAligner(schema).repair(["alpha", "beta", "gamma"],
                                  ["beta", "gamma", "delta"])
show(r)
```

```
score = 0.8667   deficit = 0.1333
  replace (root)                   delta=0.1333  value='alpha'  [primitive_replace_reorder]
  note: schema contains at least one order='align' list;
        list_item_missing / list_item_excess / primitive_replace_reorder
        ops use semantic paths (path points at the list, not a specific index).
```

The path is `(root)` — the root *is* the list. For reorder lists positional
indices are meaningless after the Hungarian permutation, so the op kind is
`primitive_replace_reorder` and the path is *list-level*. `RepairResult.apply_to`
applies it by scanning the list for an item equal to `op.pred` and replacing
in place.

> **Note on reorder paths:** every op kind ending in `_reorder` / `_missing` /
> `_excess` uses a *list-level* path. That's the schema declaring "position
> doesn't matter" — repair ops match by content, not by index.

### Example 7 — The canonical nested movie schema

```python
schema = {
    "type": "object", "keyScore": "exact",
    "keyImportance": 0, "valueImportance": 1,
    "properties": {
        "title":  {"type": "string", "score": "jaro", "valueWeight": 2.0},
        "year":   {"type": "integer", "score": "exact", "valueWeight": 1.0},
        "genres": {"type": "array", "items": {"type": "string", "score": "jaro"},
                   "order": "align", "valueWeight": 1.0},
    },
}
gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
pred = {"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]}
r = ObjectAligner(schema).repair(gold, pred)
show(r)
```

```
score = 0.6708   deficit = 0.3292
  replace /year                    delta=0.2500  value=1999  [primitive_replace]
  replace /genres                  delta=0.0625  value='Action'  [primitive_replace_reorder]
  replace /title                   delta=0.0167  value='The Matrix'  [primitive_replace]
  note: schema contains at least one order='align' list; ...
```

Three ops, ranked by impact. Apply them all:

```python
patched = r.apply_to(pred)
# patched = {'title': 'The Matrix', 'year': 1999, 'genres': ['Sci-Fi', 'Action']}
# aligner.metric(gold, patched)["score"] == 1.0
# pred itself is unchanged (apply_to deep-copies).
```

### Example 8 — `granularity="subtree"` — fix the whole thing

```python
r = ObjectAligner(schema).repair(gold, pred, granularity="subtree")
show(r)
```

```
score = 0.6708   deficit = 0.3292
  replace (root)                   delta=0.3292  value={'title': 'The Matrix',
                                                       'year': 1999,
                                                       'genres': ['Sci-Fi', 'Action']}  [subtree_replace]
```

Subtree mode emits one whole-subtree replace per internal node. For
non-trivial structures you'd see one per nested dict / list. The root
op alone closes the entire deficit.

### Example 9 — `granularity="all"` — leaf-level + subtree-level

```python
r = ObjectAligner(schema).repair(gold, pred, granularity="all")
show(r)
```

```
score = 0.6708   deficit = 0.3292   residual = +0.3917
  replace (root)                   delta=0.3292  value={...}  [subtree_replace]
  replace /year                    delta=0.2500  value=1999  [primitive_replace]
  replace /genres                  delta=0.0625  value='Action'  [primitive_replace_reorder]
  replace /genres                  delta=0.0625  value=[...]  [subtree_replace]
  replace /title                   delta=0.0167  value='The Matrix'  [primitive_replace]
```

`granularity="all"` returns both views. **Do not naively sum** —
subtree ops *contain* their descendant leaf ops. The `residual` field
flags the overlap (+0.39 here = double-counting).

### Example 10 — Filtering small ops with `min_contribution`

```python
schema = {
    "type": "object", "keyScore": "exact",
    "keyImportance": 0, "valueImportance": 1,
    "properties": {
        "big":   {"type": "string", "score": "exact", "valueWeight": 9.0},
        "small": {"type": "string", "score": "exact", "valueWeight": 1.0},
    },
}
r = ObjectAligner(schema).repair({"big": "x", "small": "y"},
                                  {"big": "X", "small": "Y"},
                                  min_contribution=0.5)
show(r)
```

```
score = 0.0000   deficit = 1.0000   residual = -0.1000
  replace /big                     delta=0.9000  value='x'  [primitive_replace]
```

Only the `big` op (delta 0.9) passes the threshold; the `small` op (delta
0.1) is filtered. The residual reflects the filtered contribution.

Key-rename pairs are treated **atomically** — both kept or both dropped,
gated by the `add`'s delta.

### Example 11 — Referential repair (`ref_fix`)

```python
schema = {
    "type": "object", "keyScore": "exact",
    "properties": {
        "people": {"type": "array", "order": "align",
            "items": {"type": "object", "keyScore": "exact",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                }}},
        "relations": {"type": "array", "order": "align",
            "items": {"type": "object", "keyScore": "exact",
                "properties": {
                    "source": {"type": "integer", "ref": "person"},
                    "target": {"type": "integer", "ref": "person"},
                }}},
    },
}
gold = {"people":    [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "relations": [{"source": 1, "target": 2}]}
pred = {"people":    [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}],
        "relations": [{"source": 124, "target": 53}]}      # ← swapped
r = ObjectAligner(schema).repair(gold, pred)
show(r)
```

```
score = 0.8750   deficit = 0.1250
  replace /relations/0/source      delta=0.0625  value=53  [ref_fix]
  replace /relations/0/target      delta=0.0625  value=124  [ref_fix]
```

The `value` on a `ref_fix` op is the **pred-side** id that pred *should*
have, looked up via the derived bijection. So although `gold.relations[0].source`
is `1`, the suggested replacement is `53` — pred's id for Alice.

> `idScope` leaves never produce repair ops — they're matched symbolically
> and always score 1.

### Example 12 — `RepairResult.apply_to` round-trip

```python
patched = result.apply_to(pred)
print("pred:    ", pred)        # original, unchanged
print("patched: ", patched)     # deep copy with all ops applied
print("score:   ", aligner.metric(gold, patched)["score"])
```

```
pred:     {'title': 'The Matrx', 'year': 2000, 'genres': ['Sci-Fi', 'Adventure']}
patched:  {'title': 'The Matrix', 'year': 1999, 'genres': ['Sci-Fi', 'Action']}
score:    1.0
```

`apply_to` always operates on a deep copy — the original `pred` is never
mutated. Useful both for testing and for producing the patched object you'd
hand off to a re-extraction loop.

---

## API reference

Canonical signatures, parameter descriptions, and field tables live in
[`api.md`](api.md). This section only links into them and documents the
chapter-specific tables (granularity modes, the `kind` discriminator) that
have no natural home there.

- [`ObjectAligner.repair()`](api.md#objectalignerrepair) — runs `align()`
  then emits ranked repair ops.
- [`ObjectAligner.repair_from_match()`](api.md#objectalignerrepair_from_match)
  — same emission against a pre-computed match tree.
- [`generate_repairs()`](api.md#generate_repairs) — low-level functional
  entry; takes a match tree directly.
- [`RepairOp`](api.md#repairop) and
  [`RepairResult`](api.md#repairresult) — result types.
  `RepairResult.apply_to(target)` returns a deep-copied `target` with every
  op applied. Both types are iterable and indexable over `ops`.

### Granularity modes

| Mode | Emits | Sum invariant |
|---|---|---|
| `"leaf"` *(default)* | every `add`/`remove`/`replace` for an individual mismatch | $\sum \mathrm{score\_delta} = 1 - S$ |
| `"subtree"` | one `subtree_replace` per imperfect internal node | Root op equals the deficit; nested ops *contain* their descendants — not additive. |
| `"all"` | union of the two | **Not additive.** `residual` reflects the overlap. |

### `min_contribution` filter

Ops with `score_delta < min_contribution` are dropped. `RepairResult.residual`
reflects the filtered contributions. Key-rename pairs (`key_rename_remove` +
`key_rename_add` sharing a `pair_id`) are treated **atomically** — both kept
or both dropped, gated by the `add`'s delta.

### `kind` discriminator

| `kind` | When emitted |
|---|---|
| `primitive_replace` | Leaf mismatch in a positional context (dict value, `fixed`/`prefix`/`combined` list). |
| `primitive_replace_reorder` | Leaf mismatch inside an `order: "align"` list. Path is list-level. |
| `list_item_add` | Missing gold item in a positional list. |
| `list_item_remove` | Excess pred item in a positional list. |
| `list_item_missing` | Missing gold item in a reorder list. Path is list-level. |
| `list_item_excess` | Excess pred item in a reorder list. Path is list-level. |
| `key_add` | Missing gold key. `score_delta` lumps key + value contributions. |
| `key_remove` | Excess pred key. Same lumping. |
| `key_rename_remove` | First half of a fuzzy-key-rename pair. `score_delta = 0`. |
| `key_rename_add` | Second half; carries the full key + value gain. Shares `pair_id` with the remove. |
| `ref_fix` | Wrong `ref` leaf. `value` is the pred-side id from the derived bijection. |
| `subtree_replace` | Whole-subtree replacement in `granularity="subtree"` / `"all"`. |

---

## Caveats

### Non-additivity under Hungarian / DP re-pairing

For schemas with `order: "align"`, `keyScore: "jaro"`, or `idScope`/`ref`,
the alignment is chosen by Hungarian/DP/bijection optimizers. Tree-walk
`score_delta` is **exact under the alignment's chosen pairing** but does
*not* predict re-pairing.

Concretely: applying op #1 and re-running `metric()` may not produce
exactly the score change predicted by `score_delta(op_1)` if the alignment
shifts as a result. In the worst case, $\sum \mathrm{score\_delta}$ over
ranked ops can exceed $1 - S$.

For prompt-optimizer feedback this is fine — a conservative ranking is
more actionable than a fragile exact gradient. For sequential auto-repair
pipelines, prefer evaluating after each application rather than trusting
the summed deltas.

### Key-rename pairs only work when applied together

The `key_rename_remove` op has `score_delta = 0`. Applying *only* the `add`
side leaves the old key in place (the pred ends up with both keys). Always
apply the pair together (`apply_to` does this automatically; bespoke
appliers should respect `pair_id`).

### Reorder list paths are not RFC 6902-strict

`list_item_missing` / `list_item_excess` / `primitive_replace_reorder` ops
point their `path` at the list itself, not a specific index. This is by
design — the schema declared `order: "align"`, so positions are
semantically meaningless. `apply_to` handles these by scanning the list
for `op.pred` and acting on the matched item.

### `granularity="all"` is for inspection, not application

Subtree ops contain leaf ops; applying both layers double-counts. If you
need to *apply* a complete set, choose `granularity="leaf"` (or
`"subtree"`) explicitly.

---

## See also

- [`attribution.md`](attribution.md) — same tree-walk math, structured
  rather than prescriptive.
- [`feedback.md`](feedback.md) — top-K prescriptive feedback rendered from a
  `RepairResult`.
- [`metric.md`](metric.md) — the surrounding evaluation call.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

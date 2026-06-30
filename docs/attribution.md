# 🎯 Per-Property Score Attribution

[Docs](index.md) › Per-Property Score Attribution

`metric()` and `align()` answer *how well* a candidate matches the gold. **`attribute()`** answers a different question: *where exactly is the deficit, and how big is each piece?*

Given the same `(gold, pred)` you'd feed into `metric()`, `attribute()` returns a ranked, path-keyed decomposition of $1 - \mathrm{s}$ — one entry per schema-relevant location, sorted by how much of the deficit lives there.

This is the deterministic, no-LLM-judge backbone for:

- **prompt-optimizer feedback** (GEPA / DSPy): emit the top-k locations a reflective optimizer should focus on next;
- **regression debugging**: when a score drops from 0.91 to 0.78, see *which paths moved*;
- **dataset triage**: rank samples by the attribution of one specific path (e.g., "which examples is my year-extractor failing on?");
- **hierarchical drill-down UIs**: surface a subtree-level summary first, drill into individual leaves on demand.

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
result = aligner.attribute(
    gold={"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]},
    pred={"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]},
)
print(f"score = {result.score:.4f}  deficit = {1 - result.score:.4f}")
for e in result.entries:
    print(f"  {e.path:14s}  contrib={e.contribution:.4f}  score={e.score:.3f}")
```

Output:

```
score = 0.6708  deficit = 0.3292
  /year           contrib=0.2500  score=0.000
  /genres/1       contrib=0.0625  score=0.500
  /title          contrib=0.0167  score=0.967
  /genres/0       contrib=0.0000  score=1.000
```

`/year` accounts for $0.25 / 0.33 \approx 76\%$ of the loss. Fix the year extractor first.

---

## The formula

Every internal node in the match tree writes its score as a convex combination of its children's scores under the chosen Hungarian/DP assignment $\pi^\star$:

$$
s_v \;=\; \sum_{u \in \mathrm{children}(v)} \alpha_{v,u}\, s_u,
\qquad \alpha_{v,u} \ge 0,\quad \sum_u \alpha_{v,u} = 1.
$$

The $\alpha$ coefficients are derived from the schema (`valueWeight`, `keyImportance`, `prefixWeights`, …) and the post-alignment structure — they do **not** depend on the child scores themselves. Multiplying $\alpha$ along the path from root to a leaf gives that leaf's **effective weight** $c_L$:

$$
c_L \;=\; \prod_{(p,c)\,\in\,\mathrm{root}\to L} \alpha_{p,c}.
$$

The leaf's contribution to the deficit is $c_L \cdot (1 - s_L)$, and these contributions partition the total deficit exactly:

$$
\boxed{\;
1 - \mathrm{s} \;=\; \sum_{L \,\in\, \mathrm{leaves}(T)} c_L\,(1 - s_L)
\;}
$$

Object Aligner exposes this directly: each `AttributionEntry` has a `weight` ($c_w$) and a `contribution` ($c_w \cdot (1 - s_w)$). This per-location contribution is the same quantity the paper writes as a repair operation's score delta, $\Delta(\mathrm{op}) = c_w\,(1 - s_w)$ (see [repair](repair.md) and [feedback](feedback.md)); $c_L$ above is the effective weight $c_w$ specialized to a leaf $L$.

---

## Shared setup for the examples

Every code block below was executed; numbers in the output are real. The
examples share an import and a tiny printing helper, defined once here and
reused throughout:

```python
from object_aligner import ObjectAligner

def show(result):
    header = f"score = {result.score:.4f}   deficit = {1 - result.score:.4f}"
    if abs(result.residual) > 1e-9:
        header += f"   residual = {result.residual:+.4f}"
    print(header)
    for e in result.entries:
        marker = ""
        if e.part == "key":
            marker = "  [key]"
        elif not e.is_leaf:
            marker = f"  [subtree {e.node_kind}]"
        elif e.leaf_kind:
            marker = f"  [{e.leaf_kind}]"
        path = e.path or "(root)"
        print(f"  {path:22s} score={e.score:.3f}  weight={e.weight:.3f}  contrib={e.contribution:.4f}{marker}")
```

Each example below builds its own schema inline (the chapter walks through
a different shape per example) but reuses `ObjectAligner` and `show` from
this block. (A few examples have very long output and are truncated for
readability — the helper itself prints every entry.)

## Examples

### Example 1 — A single primitive

```python
schema = {"type": "string", "score": "jaro"}
r = ObjectAligner(schema).attribute("hello", "hallo")
show(r)
```

```
score = 0.8667   deficit = 0.1333
  (root)                 score=0.867  weight=1.000  contrib=0.1333
```

The root *is* the leaf. Path is `""` (empty pointer = root per RFC 6901).

### Example 2 — Flat dict, uniform weights

```python
schema = {
    "type": "object",
    "keyScore": "exact",
    "keyImportance": 0,
    "valueImportance": 1,
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer", "score": "exact"},
    },
}
r = ObjectAligner(schema).attribute(
    {"name": "Alice", "age": 30},
    {"name": "Alicia", "age": 29},
)
show(r)
```

```
score = 0.4111   deficit = 0.5889
  /age                   score=0.000  weight=0.500  contrib=0.5000
  /name                  score=0.822  weight=0.500  contrib=0.0889
```

With uniform `valueWeight` and two properties, each leaf gets $c = 1/2$. The exact-integer comparator scores $\mathrm{age}=0$, so it dominates.

### Example 3 — Non-uniform `valueWeight`

```python
schema = {
    "type": "object",
    "keyScore": "exact",
    "keyImportance": 0,
    "valueImportance": 1,
    "properties": {
        "id":   {"type": "string", "score": "exact", "valueWeight": 4.0},
        "note": {"type": "string", "valueWeight": 1.0},
    },
}
r = ObjectAligner(schema).attribute(
    {"id": "X1", "note": "fine"},
    {"id": "X2", "note": "find"},
)
show(r)
```

```
score = 0.1667   deficit = 0.8333
  /id                    score=0.000  weight=0.800  contrib=0.8000
  /note                  score=0.833  weight=0.200  contrib=0.0333
```

`id` gets $c = 4/(4+1) = 0.8$. A `0.8` weight on a fully failed comparator explains 96 % of the deficit.

### Example 4 — `keyScore` with noisy keys

```python
schema = {
    "type": "object",
    "keyScore": "jaro",
    "keyImportance": 1,
    "valueImportance": 1,
    "properties": {
        "userName":    {"type": "string"},
        "phoneNumber": {"type": "string"},
    },
}
r = ObjectAligner(schema).attribute(
    {"userName": "alice", "phoneNumber": "555-1234"},
    {"username": "alice", "phone":       "555-1234"},   # different keys!
)
show(r)
```

```
score = 0.9337   deficit = 0.0663
  /phoneNumber           score=0.818  weight=0.250  contrib=0.0455  [key]
  /userName              score=0.917  weight=0.250  contrib=0.0208  [key]
  /userName              score=1.000  weight=0.250  contrib=0.0000
  /phoneNumber           score=1.000  weight=0.250  contrib=0.0000
```

With `keyImportance=1`, key fuzzy-match scores show up as their own entries (`part="key"`). Values are perfect once Hungarian pairs the noisy keys correctly. The `phoneNumber` key is the noisiest (`phone` vs `phoneNumber`), so it accounts for the bulk of the (small) deficit.

### Example 5 — Fixed-order list, mismatched lengths

```python
schema = {"type": "array", "items": {"type": "string", "score": "jaro"}}
r = ObjectAligner(schema).attribute(["a", "b", "c"], ["a", "c"])
show(r)
```

```
score = 0.6667   deficit = 0.3333
  /1                     score=0.000  weight=0.333  contrib=0.3333
  /0                     score=1.000  weight=0.333  contrib=0.0000
  /2                     score=1.000  weight=0.333  contrib=0.0000
```

DP aligns `a↔a` and `c↔c`; the missing `b` shows up as an unmatched leaf at index 1 with `pred=None`. Each of three counted positions gets $c = 1/3$.

> **Note on list indices:** for reorder lists, indices refer to **post-alignment** positions, not original gold indices. With `order: "align"`, the Hungarian may put gold[2] in `/0` if that's where it best paired up.

### Example 6 — Reorder list with `ignoreExcess`

```python
schema = {
    "type": "array",
    "items": {"type": "string", "score": "exact"},
    "order": "align",
    "ignoreExcess": True,
}
r = ObjectAligner(schema).attribute(["apple", "pear"], ["pear", "apple", "banana"])
show(r)
```

```
score = 1.0000   deficit = 0.0000
  /0                     score=1.000  weight=0.500  contrib=0.0000
  /1                     score=1.000  weight=0.500  contrib=0.0000
```

Reorder finds both gold items in `pred`; the excess `banana` is dropped from $D$, so $D = 2$ and the gold items each get $c = 1/2$. Perfect score; nothing to attribute.

### Example 7 — Prefix-only list, weighted positions

```python
schema = {
    "type": "array",
    "prefixItems": [
        {"type": "string"},
        {"type": "integer", "score": "exact"},
        {"type": "string"},
    ],
    "prefixWeights": [3.0, 1.0, 1.0],
}
r = ObjectAligner(schema).attribute(["alpha", 7, "beta"], ["alphz", 8, "betz"])
show(r)
```

```
score = 0.6867   deficit = 0.3133
  /1                     score=0.000  weight=0.200  contrib=0.2000
  /0                     score=0.867  weight=0.600  contrib=0.0800
  /2                     score=0.833  weight=0.200  contrib=0.0333
```

Normalized weights $\tilde w = (0.6, 0.2, 0.2)$ become the $c$ values. The integer at index 1 wholly mismatches; even at $c = 0.2$ it's the largest contribution.

### Example 8 — Combined `prefixItems` + `items`

```python
schema = {
    "type": "array",
    "prefixItems": [{"type": "string"}, {"type": "string"}],
    "items": {"type": "string", "score": "exact"},
    "prefixImportance": 1.0,
    "restImportance":   3.0,
}
r = ObjectAligner(schema).attribute(
    ["sku", "name", "tag1", "tag2"],
    ["sku", "Name", "tag1", "tag3"],
)
show(r)
```

```
score = 0.4792   deficit = 0.5208
  /3                     score=0.000  weight=0.250  contrib=0.2500
  /4                     score=0.000  weight=0.250  contrib=0.2500
  /1                     score=0.833  weight=0.125  contrib=0.0208
  /0                     score=1.000  weight=0.125  contrib=0.0000
  /2                     score=1.000  weight=0.250  contrib=0.0000
```

`prefixImportance / restImportance = 1 / 3` means the rest block carries $\bar{w}_r = 0.75$ of the total mass, the prefix block carries $\bar{w}_p = 0.25$. Each prefix child gets $0.25 \cdot 0.5 = 0.125$; each rest child gets $0.75 / 2 = 0.375$ — but `exact` on `"tag3" vs "tag2"` scores 0 *and* the aggregator splits that zero-cell into two unmatched leaves (`/3` for gold-side, `/4` for pred-side), each carrying $c = 0.250$.

### Example 9 — Nested movie schema (the canonical worked example)

Same schema and data as the [Quickstart](#quickstart) above; this time we use `show(r)` for the unified format:

```
score = 0.6708   deficit = 0.3292
  /year                  score=0.000  weight=0.250  contrib=0.2500
  /genres/1              score=0.500  weight=0.125  contrib=0.0625
  /title                 score=0.967  weight=0.500  contrib=0.0167
  /genres/0              score=1.000  weight=0.125  contrib=0.0000
```

Effective weights compose multiplicatively:
$c_\text{title} = 1 \cdot \frac{2}{4} = 0.5$, $c_\text{year} = 1 \cdot \frac{1}{4} = 0.25$, $c_{\text{genres}/i} = 1 \cdot \frac{1}{4} \cdot \frac{1}{2} = 0.125$, summing to $1$.

### Example 10 — Deeper nesting: drill down with `granularity="subtree"`

```python
schema = {
    "type": "object",
    "keyScore": "exact",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "keyScore": "exact",
                "properties": {
                    "kind":  {"type": "string", "score": "exact"},
                    "actor": {"type": "string"},
                    "year":  {"type": "integer", "score": "exact"},
                },
            },
            "order": "align",
        },
    },
}
gold = {"events": [
    {"kind": "win",  "actor": "Alice", "year": 2020},
    {"kind": "loss", "actor": "Bob",   "year": 2021},
]}
pred = {"events": [
    {"kind": "win",  "actor": "Alise", "year": 2020},   # nearly right
    {"kind": "loss", "actor": "X",     "year": 9999},   # mostly wrong (only the kind matches)
]}
aligner = ObjectAligner(schema)
```

**Leaf granularity** — atomic primitives, ranked:

```python
r = aligner.attribute(gold, pred)             # default granularity="leaf"
show(r)
```

```
score = 0.6444   deficit = 0.3556
  /events/1/actor        score=0.000  weight=0.167  contrib=0.1667
  /events/1/year         score=0.000  weight=0.167  contrib=0.1667
  /events/0/actor        score=0.867  weight=0.167  contrib=0.0222
  /events/0/kind         score=1.000  weight=0.167  contrib=0.0000
  /events/0/year         score=1.000  weight=0.167  contrib=0.0000
  /events/1/kind         score=1.000  weight=0.167  contrib=0.0000
```

**Subtree granularity** — pick the right level of abstraction first:

```python
r = aligner.attribute(gold, pred, granularity="subtree")
show(r)
```

```
score = 0.6444   deficit = 0.3556   residual = +0.7111
  (root)                 score=0.644  weight=1.000  contrib=0.3556  [subtree dict]
  /events                score=0.644  weight=1.000  contrib=0.3556  [subtree list:reorder]
  /events/1              score=0.333  weight=0.500  contrib=0.3333  [subtree dict]
  /events/0              score=0.956  weight=0.500  contrib=0.0222  [subtree dict]
```

Reading top-down: 100 % of the deficit lives under `/events`; within that, ~94 % is concentrated in event index 1. *Fix event 1.* From there you can re-run with `granularity="leaf"` to see which field of event 1 hurts most. (The `residual` in the header is the expected by-product of subtree mode — the nested entries are not additive; see the note below.)

> **Subtree entries are nested — do not sum across them.** The root's `0.3556` and `/events`'s `0.3556` aren't additive; one *contains* the other. `total_contribution` over subtree entries is **not** the deficit; the `residual` value on the result reflects that.

### Example 11 — Threshold clipping

```python
schema = {"type": "string", "score": "jaro", "threshold": 0.95}
r = ObjectAligner(schema).attribute("hello", "hallo")
show(r)
```

```
score = 0.0000   deficit = 1.0000
  (root)                 score=0.000  weight=1.000  contrib=1.0000
```

Jaro of `"hello"` vs `"hallo"` is $\approx 0.867$ — below the threshold $0.95$, so the leaf is clipped to $0$. Attribution uses the post-clip score (matching what `metric()` reports), so the clipped leaf shows up at full strength.

### Example 12 — Referential alignment (`idScope` / `ref`)

```python
schema = {
    "type": "object",
    "keyScore": "exact",
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
                    # A matching scalar (the edge label) so the reorder
                    # Hungarian pairs the relation with its gold counterpart;
                    # the swapped refs then surface as [ref] leaves.
                    "label":  {"type": "string", "score": "exact"},
                    "source": {"type": "integer", "ref": "person"},
                    "target": {"type": "integer", "ref": "person"},
                }}},
    },
}
gold = {
    "people":    [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
    "relations": [{"label": "knows", "source": 1, "target": 2}],
}
pred = {  # ids are arbitrary; relation source/target swapped
    "people":    [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}],
    "relations": [{"label": "knows", "source": 124, "target": 53}],
}
r = ObjectAligner(schema).attribute(gold, pred)
show(r)
```

```
score = 0.6667   deficit = 0.3333
  /relations/0/source    score=0.000  weight=0.167  contrib=0.1667  [ref]
  /relations/0/target    score=0.000  weight=0.167  contrib=0.1667  [ref]
  /people/0/id           score=1.000  weight=0.125  contrib=0.0000  [id]
  /people/0/name         score=1.000  weight=0.125  contrib=0.0000
  /people/1/id           score=1.000  weight=0.125  contrib=0.0000  [id]
  /people/1/name         score=1.000  weight=0.125  contrib=0.0000
  /relations/0/label     score=1.000  weight=0.167  contrib=0.0000
```

- `[id]` leaves always score 1 — they contribute nothing.
- `[ref]` leaves score 0 or 1 based on the derived bijection. The swapped source/target each carry a full $c_L$ of deficit.

### Example 13 — Vacuous dual-None positions

A `prefixItems` position that is **absent on both sides** (gold and pred both
ran out before reaching it) is *vacuous*: it carries **zero weight** and is
excluded from the normalization denominator. This keeps the identity
`metric(g, g) == 1` for lists shorter than `prefixItems`.

```python
schema = {
    "type": "array",
    "prefixItems": [
        {"type": "string"},
        {"type": "string"},
        {"type": "string"},
    ],
}
# Both gold and pred stop at index 2 → position /2 is absent on both sides.
r = ObjectAligner(schema).attribute(["a", "b"], ["a", "b"])
show(r)
```

Default (`include_empty_positions=False`):

```
score = 1.0000   deficit = 0.0000
  /0                     score=1.000  weight=0.500  contrib=0.0000
  /1                     score=1.000  weight=0.500  contrib=0.0000
```

The two present positions are perfect and the missing third position is not
penalized — the list scores a perfect `1.0`. The both-absent sentinel at `/2`
is filtered out entirely.

With `include_empty_positions=True` the sentinel reappears, but at **zero
weight** (`[absent]` marker) so it still contributes nothing:

```python
r = ObjectAligner(schema).attribute(["a", "b"], ["a", "b"], include_empty_positions=True)
show(r)
```

```
score = 1.0000   deficit = 0.0000
  /0                     score=1.000  weight=0.500  contrib=0.0000
  /1                     score=1.000  weight=0.500  contrib=0.0000
  /2                     score=0.000  weight=0.000  contrib=0.0000  [absent]
```

`include_empty_positions=True` is purely for inspection — it never changes the
score or introduces deficit, because the sentinel's weight is `0`.

---

## API reference

Canonical signatures, parameter descriptions, and field tables live in
[`api.md`](api.md). This section only links into them and documents the
chapter-specific tables (granularity modes) that have no natural home there.

- [`ObjectAligner.attribute()`](api.md#objectalignerattribute) — runs
  `align()` then walks the match tree.
- [`ObjectAligner.attribute_from_match()`](api.md#objectalignerattribute_from_match)
  — same walk against a pre-computed match tree.
- [`tree_walk_attribution()`](api.md#tree_walk_attribution) — low-level
  functional entry; takes a `MatchItem` / `MatchList` / `MatchDict` directly.
- [`AttributionEntry`](api.md#attributionentry) and
  [`AttributionResult`](api.md#attributionresult) — result types. Iterable
  and indexable over `entries`.

### Granularity modes

| Mode | Emits | Sum invariant |
|---|---|---|
| `"leaf"` *(default)* | every primitive leaf; key leaves; synthetic leaves at non-decomposable nodes | $\sum_L \mathrm{contrib} = 1 - \mathrm{s}$ (within float precision; up to filtered sentinels) |
| `"subtree"` | every internal node | **No** — entries are nested; treat each as a stand-alone "deficit attributable to this subtree." |
| `"all"` | leaves *and* internals | No — same caveat as `"subtree"`. The `is_leaf` field on each entry separates the two. |

---

## Caveats

### The fixed-assignment view

Tree-walk attribution is **exact** under the assignment $\pi^\star$ that `align()` actually chose. It is a true decomposition of the deficit you observe — never an estimate.

What it *cannot* see: if a perturbation flipped the Hungarian's optimal pairing on a `order: "align"` list, or re-routed a DP traceback on a `order: "fixed"` list, the alignment would change and the tree-walk weights $c_L$ would be different. Wherever a discrete optimizer (Hungarian over list items, DP traceback, dict-key Hungarian, scope bijection) sits above a leaf, tree-walk is a *first-order linearization* of what a real perturbation would do.

In practice this is fine — and arguably preferable — for prompt-optimizer feedback: a single conservative direction ("fix the year extractor first") is more actionable than a fragile exact gradient. But it does mean the tree-walk number is not directly comparable to the *counterfactual* gain you'd see by actually rewriting the candidate at that path.

---

## See also

- [`metric.md`](metric.md) — the surrounding evaluation call.
- [`repair.md`](repair.md) — ranked structured repair ops over the same tree.
- [`feedback.md`](feedback.md) — prompt-optimizer-shaped projection.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

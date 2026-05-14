# 15. Alignment Confidence

[Docs](index.md) › Alignment Confidence

Object Aligner runs the **Hungarian algorithm** at two points: matching the
items of an `order: "align"` list, and matching the keys of two objects.
The similarity matrix used to pick the best one-to-one pairing carries
information beyond the chosen pairs — it tells us *how committed* the
algorithm was to each match. A pair where the chosen column was the
clear winner is *stable*; a pair where the second-best column was nearly
identical is *fragile*.

This chapter documents the v1 **alignment confidence** feature:
a per-pair stability score in $[0, 1]$, aggregated up the match tree
onto `MatchItem.confidence` / `MatchList.confidence` /
`MatchDict.confidence`, and consumed by `feedback()` and `describe()`
through opt-in flags. None of it changes default behaviour — when
`compute_confidence=False` (the default), every `confidence` field is
`1.0` and `feedback()` / `describe()` produce byte-identical output to
pre-confidence releases.

---

## Quickstart

```python
from object_aligner import ObjectAligner

schema = {"type": "object", "properties": {
    "name": {"type": "string"},
    "age":  {"type": "integer"},
    "email": {"type": "string"},
}}

aligner = ObjectAligner(schema, compute_confidence=True)

gold = {"name": "Ada", "age": 42,  "email": "ada@x"}
pred = {"naem": "Ad",  "years": 41, "mail":  "ada@x"}

# 1. Read confidence off the match tree
match = aligner.align(gold, pred)
print(round(match.confidence, 3))           # 0.612 — dict-level confidence

# 2. Use it to rerank feedback by expected gain (Δs × c)
print(aligner.feedback(gold, pred, rank_by="expected_gain").text)

# 3. Surface ambiguous pairings as diagnostics
print(aligner.feedback(
    gold, pred,
    include_pairing_ambiguous=True,
    ambiguity_threshold=0.5,
).text)

# 4. Show confidence in describe output
print(aligner.describe(gold, pred, show_confidence=True).text)
```

---

## Shared setup for the examples

```python
from object_aligner import ObjectAligner

schema = {
    "type": "object",
    "properties": {
        "name":  {"type": "string"},
        "age":   {"type": "integer"},
        "email": {"type": "string"},
    },
}

aligner = ObjectAligner(schema, compute_confidence=True)

gold = {"name": "Ada", "age": 42,  "email": "ada@x"}
pred = {"naem": "Ad",  "years": 41, "mail":  "ada@x"}
```

All examples below reuse `aligner`, `gold`, `pred` unless explicitly
stated otherwise.

---

## The model

At one Hungarian site let

- $n, m$ — gold and pred sizes (items, or keys),
- $d = \max(n, m)$,
- $S \in [0, 1]^{d \times d}$ — the zero-padded similarity matrix,
- $\pi^{\star}$ — the assignment returned by
  `linear_sum_assignment(-S)`,
- $\pi^{\star -1}$ — the inverse mapping (column → row).

For each chosen pair $(i, \pi^{\star}(i))$ with both indices in range
($i < n$, $\pi^{\star}(i) < m$), the **per-row margin** is the gap to
the row's second-best column:

$$
m_i^{\text{row}} \;=\; S_{i,\pi^{\star}(i)} \;-\; \max_{j \ne \pi^{\star}(i)} S_{ij}.
$$

The symmetric **per-column margin** is

$$
m_j^{\text{col}} \;=\; S_{\pi^{\star -1}(j),\, j} \;-\; \max_{i \ne \pi^{\star -1}(j)} S_{ij}.
$$

The **per-pair confidence (margin, symmetric, clipped to $[0, 1]$)** is
the average of the two:

$$
c_{i,\pi^{\star}(i)} \;=\;
\tfrac{1}{2}\!\left(
\mathrm{clip}_{[0,1]}\bigl(m_i^{\text{row}}\bigr)
\;+\;
\mathrm{clip}_{[0,1]}\bigl(m_{\pi^{\star}(i)}^{\text{col}}\bigr)
\right).
$$

For excess / missing pairs (one side is zero-padding), $c := 1.0$ — the
item is simply unmatched, there is no pairing ambiguity to report.

The **entropy method** (`confidence_method="entropy"`) softmaxes each
row of $S$ over its $m$ unpadded columns with a temperature $\beta$:

$$
p_{ij} \;=\; \frac{\exp(\beta\, S_{ij})}{\sum_{k=0}^{m-1} \exp(\beta\, S_{ik})},
\qquad
H_i \;=\; -\sum_{j=0}^{m-1} p_{ij} \log p_{ij},
$$

$$
c_i^{\text{ent}} \;=\; 1 \;-\; \frac{H_i}{\log m}.
$$

For padded rows ($i \ge n$): $c_i^{\text{ent}} := 1.0$. The temperature
$\beta$ (constructor parameter `confidence_entropy_temperature`,
default $8.0$) controls how sharply the softmax distinguishes nearby
similarities; with $\beta = 8$ a Jaro 0.95 vs Jaro 0.80 row pair gives
roughly a 3:1 probability ratio.

**Node aggregation.** The Hungarian-paired containers (lists with
`order: "align"` and any dict-key matching) take the mean over genuinely
matched pairs:

$$
c_{\text{node}} \;=\; \frac{1}{K} \sum_{i \in \text{matched}} c_{i,\pi^{\star}(i)}.
$$

For dicts the key-pair confidence and the value-subtree confidence are
blended with the same `keyImportance` / `valueImportance` weights used
for the score:

$$
c_{\text{dict}} \;=\;
\frac{w_k \cdot c_{\text{keys}} \;+\; w_v \cdot c_{\text{values}}}{w_k + w_v},
\qquad
c_{\text{values}} \;=\; \sum_p \tilde w_p \cdot c_{\text{child}_p},
$$

where $\tilde w_p$ are the normalised per-property `valueWeight`s.
For non-Hungarian containers (`kind="fixed"` / `"prefix"` /
`"combined"`) the parent confidence is the weighted mean of children;
no ambiguity is introduced at that level.

The Hungarian algorithm itself, and its three call sites in Object
Aligner, are introduced in
[Concepts & Architecture](concepts.md).

---

## Example 1 — Reading confidence off the match tree

```python
match = aligner.align(gold, pred)

print(round(match.confidence, 3))         # 0.612
for key, value in match.children.items():
    print(f"  key={key.gold!r:>8s} ⇄ {key.pred!r:>8s} "
          f"key_conf={key.confidence:.3f} value_score={value.score:.2f}")
```

```
0.612
  key=  'name' ⇄  'naem' key_conf=0.306 value_score=0.89
  key=   'age' ⇄ 'years' key_conf=0.000 value_score=0.50
  key= 'email' ⇄  'mail' key_conf=0.369 value_score=0.56
```

Three observations.

- The dict-level `confidence = 0.612` is the blended (key, value) score
  with the schema's default `keyImportance = valueImportance = 1.0`.
- `age ⇄ years` has `key_conf = 0.0`. That's the symmetric-margin
  formula at work: at least one of the row/column second-bests was at
  least as good as the chosen entry, so the clip sends the margin to
  zero. The Hungarian algorithm still chose this pair (it had to —
  no other column was available for that row) but it had no local
  signal that this *should* be the pair.
- `email ⇄ mail` has a healthy `key_conf = 0.369`. The Jaro similarity
  between `email` and `mail` is high, and the next-best contender
  (`naem`) is much lower.

---

## Example 2 — Margin method (default)

Take a 3×3 dict-key matrix you can hand-compute. With Jaro similarities

|         | naem | years | mail |
|---------|-----:|------:|-----:|
| name    | 0.92 |  0.49 | 0.69 |
| age     | 0.58 |  0.78 | 0.00 |
| email   | 0.78 |  0.51 | 0.93 |

the Hungarian picks the diagonal $(0,0), (1,1), (2,2)$. Per the
equation in §"The model":

$$
m_{0}^{\text{row}} = 0.92 - 0.69 = 0.23,
\quad
m_{0}^{\text{col}} = 0.92 - 0.78 = 0.14,
$$

$$
c_{0,0} = \tfrac{1}{2}(0.23 + 0.14) = 0.185.
$$

The implementation matches this to 1e-9.

---

## Example 3 — Entropy method

The same matrix scored with the entropy method ($\beta = 8$):

```python
aligner_ent = ObjectAligner(
    schema,
    compute_confidence=True,
    confidence_method="entropy",
    confidence_entropy_temperature=8.0,
)
match_ent = aligner_ent.align(gold, pred)
print(round(match_ent.confidence, 3))
```

The entropy method captures *multi-way* ambiguity that the margin
method's "second-best only" view misses. If three columns tie within
$\varepsilon$ of the chosen one, margin reports "stable" (the second-best
is close to the chosen, fine), but entropy correctly reports "highly
ambiguous" (multiple plausible matches). Use entropy when your data has
near-synonym ambiguity (cf. typo-style ambiguity, which margin handles
fine).

---

## Example 4 — Confidence-weighted feedback ranking

`feedback()` exposes three ranking modes via `rank_by=`:

| Mode | Sort key |
|---|---|
| `"score_delta"` *(default — current behaviour)* | $-\Delta_s$, then $(\text{path}, \text{op}, \text{kind})$ |
| `"expected_gain"` | $-\Delta_s \cdot c$, then $-\Delta_s$, then $(\text{path}, \text{op}, \text{kind})$ |
| `"confidence"` | $-c$, then $-\Delta_s$, then $(\text{path}, \text{op}, \text{kind})$ |

`"expected_gain"` is the rational-optimizer choice: an op with a big
$\Delta_s$ but a fragile pairing might just rewire the alignment
without actually improving the score, so it should rank lower than a
smaller-$\Delta_s$ op from a committed pairing.

```python
print("default:")
for op in aligner.repair(gold, pred).ops:
    print(f"  {op.kind:24s} Δs={op.score_delta:.3f} c={op.confidence:.2f}")

print("\nexpected_gain:")
for op in aligner.repair(gold, pred, rank_by="expected_gain").ops:
    print(f"  {op.kind:24s} Δs={op.score_delta:.3f} c={op.confidence:.2f}"
          f" Δs·c={op.score_delta*op.confidence:.4f}")
```

Both modes are exposed on `feedback()` and `repair()` with the same
name and default.

The default mode preserves byte-identical output of pre-confidence
releases — flipping `rank_by` is the only thing that changes the
ordering.

---

## Example 5 — Surfacing ambiguous pairings as diagnostics

A new `pairing_ambiguous` op kind reports Hungarian-paired containers
whose `confidence` falls below `ambiguity_threshold`. The op is
**diagnostic only** — it carries `score_delta = 0` and is rendered in
a separate "Diagnostic notes" trailing section so it doesn't consume
top-K slots:

```python
print(aligner.feedback(
    gold, pred,
    include_pairing_ambiguous=True,
    ambiguity_threshold=0.7,
).text)
```

```
The prediction scored 0.72 (deficit 0.28). Top 3 of 6 fix locations:
1. rename key "" -> "age" at /age (value 42). Fixing this recovers +0.165.
2. rename key "mail" -> "email" at /email (value 'ada@x'). Fixing this recovers +0.085.
3. rename key "naem" -> "name" at /name (value 'Ada'). Fixing this recovers +0.032.
Focus on key-rename errors — they account for 100% of the deficit shown.
Diagnostic notes (low-confidence pairings):
~ /: pairing between gold and predicted items was ambiguous (confidence 61%). Make these items more distinctive before fixing deeper field-level errors.
```

The diagnostic section tells the optimizer: *your prompt is producing
outputs that are right-ish but not distinctive enough for the evaluator
to pin down — work on disambiguating the keys here before chasing
deeper field-level errors.* That signal is uniquely available because
Object Aligner runs the Hungarian in the first place.

Default `include_pairing_ambiguous=False` — fully opt-in, off by
default.

---

## Example 6 — Confidence in describe output

`describe()` mirrors the same two flags:

```python
print(aligner.describe(gold, pred, show_confidence=True).text)
```

```
The predicted output scores overall 72%, let us align the predicted output to the gold and analyze the differences:
The predicted dictionary scores 72%: (low confidence 0.61)
  KEY = The predicted key "naem" does not match the gold "name" (score=92%). (low confidence 0.31)
  VALUE = The predicted value "Ad" does not match the gold "Ada" (score=89%).

  KEY = The predicted key "years" does not match the gold "age" (score=51%). (low confidence 0.00)
  VALUE = ...
```

The suffix is **banded**: silent for $c \ge 0.70$, `" (confidence X)"`
for $0.40 \le c < 0.70$, `" (low confidence X)"` for $c < 0.40$. That
keeps noise out of the prose at high confidence — you only see the
qualifier where it actually matters.

Set `include_ambiguous=True` to also emit a dedicated `NOTE:` line
above any Hungarian-paired container whose confidence is below
`ambiguity_threshold`:

```python
print(aligner.describe(
    gold, pred,
    show_confidence=True,
    include_ambiguous=True,
    ambiguity_threshold=0.7,
).text)
```

```
The predicted output scores overall 72%, ...
The predicted dictionary scores 72%: (low confidence 0.61)
  NOTE: dict key alignment was ambiguous (confidence 61%).
  KEY = ...
```

Both default to `False`. With both off, `describe()` returns
byte-identical output to pre-confidence releases.

---

## API reference

The constructor surface:

```python
ObjectAligner(
    schema, *,
    compute_confidence=False,                 # opt-in master switch
    confidence_method="margin",               # "margin" | "entropy"
    confidence_entropy_temperature=8.0,       # β for entropy method
    ...                                       # (other args unchanged)
)
```

The `feedback()` surface:

```python
aligner.feedback(
    gold, pred, *,
    rank_by="score_delta",                    # "score_delta" | "expected_gain" | "confidence"
    include_pairing_ambiguous=False,
    ambiguity_threshold=0.30,
    ...                                       # (other args unchanged)
)
```

The `describe()` surface:

```python
aligner.describe(
    gold, pred, *,
    show_confidence=False,
    include_ambiguous=False,
    ambiguity_threshold=0.30,
    ...                                       # (other args unchanged)
)
```

`repair()` accepts the same three new flags as `feedback()`. The full
signatures are generated under [API Reference](api.md#objectaligner).

---

## Caveats

- **Margin is a fast approximation, not the LP-dual answer.** The
  symmetric-clip formula uses only the existing similarity matrix; it
  does not re-solve the assignment. The truly correct quantity — "by
  how much would $V^{\star}$ drop if I forbid this specific edge?" —
  costs $O(n)$ extra Hungarian runs per node and is out of scope for
  v1.
- **Entropy needs a temperature.** $\beta = 1$ is too smooth on
  $[0, 1]$-bounded similarities; the default $\beta = 8$ matches
  intuition for Jaro-style scores. Tune via
  `confidence_entropy_temperature` if your similarities live on a
  different scale.
- **No effect on `score`.** Confidence is a *commentary* on the
  alignment, not a re-weighting of it. The score returned by
  `metric()` is unchanged whether or not `compute_confidence` is on.
- **Default-off everywhere.** With default constructor and method
  flags, every `confidence` field is `1.0`, `rank_by="score_delta"`
  reduces to the legacy sort, no `pairing_ambiguous` ops are emitted,
  no confidence suffixes are appended. `feedback()` and `describe()`
  output is byte-identical to releases without this feature.
- **`metric(debug=True)` is *not* in the byte-identical contract.**
  When `compute_confidence=True`, the debug tree gains a `confidence`
  key on nodes whose confidence differs from `1.0`. With the default
  `compute_confidence=False` it remains byte-identical.

---

## See also

- [Lists & Arrays](lists.md) — where the list-reorder Hungarian runs.
- [Dictionaries & Objects](dicts.md) — where the dict-key Hungarian
  runs.
- [Prompt-Optimizer Feedback](feedback.md) — the consumer of
  `rank_by` and `include_pairing_ambiguous`.
- [Plain-English Description](describe.md) — the consumer of
  `show_confidence` and `include_ambiguous`.

[← Documentation home](index.md)

# 🧩 object-aligner

<p align="center">
  <a href="https://github.com/aic-factcheck">
    <img src="https://github.com/aic-factcheck.png" alt="AIC" width="120" />
  </a>
</p>

<p align="center">
  <em>A configurable, deterministic similarity score for structured (JSON-like) data — and a drop-in, model-free reward for LLM prompt optimization.</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13%2B-blue" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
  <img alt="Status" src="https://img.shields.io/badge/status-research%20preview-orange" />
  <a href="docs/index.md"><img alt="Docs" src="https://img.shields.io/badge/docs-read%20the%20guide-blueviolet" /></a>
</p>

LLMs are increasingly asked to emit **JSON conforming to a fixed schema** — for information
extraction, tool calling, agentic planning, and knowledge-graph construction. Measuring how
close such an output is to a gold reference is awkward: exact match is brittle, text
similarity ignores structure, and an LLM judge — powerful and flexible, but costlier to run
and harder to reproduce — is not always the right fit when you need a fast, deterministic,
auditable score. Object Aligner offers a complementary alternative for that case.

**Object Aligner (OA)** scores two JSON objects by **recursively aligning their trees** —
the Hungarian algorithm for unordered collections, sequence alignment for ordered ones —
and awarding **partial credit at the granularity the schema declares**. It is configured
entirely through a compact set of **JSON Schema extensions**, so adapting it to a new task
means *annotating a schema, not writing code*.

A primary use is **prompt optimization**: OA's deterministic, decomposable score makes a
ready reward signal for optimizers such as [GEPA](https://github.com/gepa-ai/gepa) or
[DSPy](https://github.com/stanfordnlp/dspy) — and because the same alignment
localizes every mismatch, it also emits ranked, natural-language feedback for their reflection
slots, with no extra model call.


---

## ✨ Highlights

- 🌳 **Schema-driven recursive alignment** — one deterministic score in `[0, 1]`, with partial
  credit at every node, for arbitrarily nested objects, lists, and primitives.
- 🕸️ **Referential alignment for (hyper)graphs**  — score cross-referenced
  records **up to identifier renumbering**. OA infers a bijection between gold and candidate ids
  and scores every reference through it.
- 🔢 **Per-list sequence semantics** — choose, per list, between **order-agnostic** matching,
  an **order-sensitive** monotone regime (insertions/deletions) for ranking & planning, and
  **positional tuples / prefixes** whose slots carry position-specific meaning.
- 🧭 **Deterministic ranked feedback** — the same alignment that produces the score also
  pinpoints *where* the candidate departs from gold and emits ranked **repair operations**,
  scored by the exact amount of score each recovers — **no LLM call**.
- 🔌 **Drop-in optimizer reward** — plug OA into prompt optimizers like
  [DSPy](https://github.com/stanfordnlp/dspy) or GEPA as a reproducible,
  auditable, model-free reward (and reflection signal).
- 🧬 **Semantic string similarity** *(optional extension)* — score text fields by **meaning**
  rather than character overlap, using OpenAI (or any OpenAI-compatible) embeddings, with
  built-in caching and batching.
- 🧮 **Deterministic & decomposable** — same inputs → same number; the top-level score is an
  explicit weighted aggregate of child scores, which is what makes attribution and feedback exact.

---

## 📦 Installation

Not on PyPI yet — install straight from GitHub (like other
[AIC](https://github.com/aic-factcheck) tools, e.g.
[`aic-nlp-utils`](https://github.com/aic-factcheck/aic-nlp-utils)):

```bash
pip install git+https://github.com/aic-factcheck/object_aligner.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/aic-factcheck/object_aligner.git
```

**Optional extras** (embedding-based semantic string similarity via an OpenAI-compatible API):

```bash
pip install "object-aligner[semantic-openai] @ git+https://github.com/aic-factcheck/object_aligner.git"
```

Requires **Python 3.13+**.

---

## 🚀 Quick start

```python
from object_aligner import ObjectAligner

schema = {"type": "string", "score": "jaro"}
aligner = ObjectAligner(schema)

print(aligner.metric("hello", "hallo"))          # {'score': 0.8667}
```

Score a nested object and ask for human-readable feedback in one call:

```python
result = aligner.metric(gold, pred, generate_feedback=True)
print(result["score"])
print(result["feedback"])   # ranked, prescriptive fix list — deterministic, no LLM
```

---

## 🧠 How it works

OA takes a **gold** object `g`, a **candidate** object `p`, and a **schema** `S`, and returns
`score(g, p | S) ∈ [0, 1]`. Scoring at every internal node runs in two phases:

1. **Alignment** — fix a correspondence between the children of `g` and `p`
   (Hungarian assignment for unordered collections / maps; a sequence-alignment dynamic
   program for ordered lists).
2. **Scoring** — aggregate the per-pair child scores over that correspondence into a single
   number, weighted as the schema declares.

Both branches recurse, so any nesting depth works naturally. Primitives are scored directly
by a configurable comparator; empty values (`null`/`None`) are handled explicitly.

---

## 🔑 Capabilities

| Area | What you get |
|------|--------------|
| 🔤 **Primitives** | Strings (`exact`, `jaro`, `jaro_winkler`, `levenshtein`, `damerau_levenshtein`, `osa`, `indel`, `lcsseq`), numbers (`exact`, `invdiff`, `relative`), per-field thresholds, and **custom metric callables**. See [primitives](docs/primitives.md). |
| 📚 **Lists & sequences** | `order:"fixed"` (positional), `order:"align"` (order-agnostic Hungarian), monotone order-sensitive alignment, `prefixItems`/`prefixWeights` tuples, and `ignoreExcess`/`ignoreMissing`. See [lists](docs/lists.md). |
| 🗂️ **Maps / objects** | Keys matched **by label only** (Hungarian), then values graded recursively; tune with `keyImportance`, `valueImportance`, `valueWeight`. See [dicts](docs/dicts.md). |
| 🕸️ **Referential alignment** | `idScope` / `ref` declare primary/foreign-key-style links; OA scores references **invariant to id relabeling**, with 1-WL tie-breaking for property-identical twins. See [referential](docs/referential.md). |
| 🧭 **Feedback** | `feedback()` → top-K ranked repair string for optimizer reflection slots (GEPA/DSPy/TextGrad). See [feedback](docs/feedback.md). |
| 🩹 **Attribution & repair** | `attribute()` decomposes the deficit into ranked per-path contributions; `repair()` emits RFC-6902-style ops with exact score deltas and `apply_to()`. See [attribution](docs/attribution.md), [repair](docs/repair.md). |
| 🗣️ **Describe** | `describe()` → deterministic plain-English walk of the alignment tree. See [describe](docs/describe.md). |
| 🈳 **Null handling** | Per-field `nullScore` for asymmetric `null`/value mismatches. See [null handling](docs/null_handling.md). |
| 📈 **Confidence** | Opt-in per-pair stability scores harvested from each Hungarian matrix. See [confidence](docs/confidence.md). |
| 🧬 **Semantic similarity** | Opt-in embedding-based string metric with caching, batching, and OpenAI-compatible transport. See [semantic](docs/semantic.md). |

### 🕸️ Referential alignment

Complex structured data is rarely a flat tree: cross-references between records make it a
**graph or hypergraph**, which no prior similarity metric scores once identifiers are
arbitrary. Mark one primitive as an identifier (`idScope`) and others as references (`ref`):

```python
schema = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array", "order": "align",
            "items": {"type": "object", "properties": {
                "id":   {"type": "integer", "idScope": "person"},
                "name": {"type": "string",  "score": "exact", "valueWeight": 2.0},
                "role": {"type": "string",  "score": "exact"},
            }},
        },
        "mentorships": {
            "type": "array", "order": "align", "ignoreExcess": True,
            "items": {"type": "object", "properties": {
                "mentor": {"type": "integer", "ref": "person"},
                "mentee": {"type": "integer", "ref": "person"},
            }},
        },
    },
}
```

OA infers the gold→candidate id bijection (by everything *except* the masked id field),
breaks remaining ties by graph structure with Weisfeiler–Leman color refinement, and scores
every reference through the bijection — so two correct extractions that **renumber and
reorder** their records still match. Recovering the bijection exactly is graph isomorphism,
which OA approximates in near-linear time.

### 🔌 As a prompt-optimization reward

OA is a deterministic, decomposable structural reward — cheap to evaluate at scale,
reproducible, and easy to audit. It complements LLM-as-judge rewards: use a judge for
open-ended semantic grading, and OA when the answer has a known schema (the two can also be
combined). Used as the reward inside **GEPA** across synthetic and real-world datasets, OA
produced consistent gains and never a significant loss — and the same alignment supplies the
natural-language **reflection** signal, so one call returns both *how well* a candidate did
and *what to change*.

---

## 🧾 Schema extensions (cheat sheet)

OA is configured with a small set of keywords layered on top of JSON Schema:

| Keyword | Applies to | Purpose |
|---------|-----------|---------|
| `score` | string / number / integer | Leaf comparator (built-in name or custom metric) |
| `threshold` | string / number / integer | Floor below which a leaf scores 0 |
| `order` | array | `"fixed"` (positional) or `"align"` (order-agnostic) |
| `ignoreExcess` / `ignoreMissing` | array | Drop unmatched candidate / gold items from the denominator |
| `prefixItems` / `prefixWeights` | array | Positional tuple head with per-slot weights |
| `keyImportance` / `valueImportance` | object | Weight of the key term vs. the value term |
| `valueWeight` | object property | Per-property weight in the value aggregate |
| `idScope` | primitive (in an array) | Declare an identifier scope (primary key) |
| `ref` | primitive | Reference into a named scope (foreign key) |
| `nullScore` | any node | Score for an asymmetric `null`/value mismatch |

Full reference: [docs/schema_reference.md](docs/schema_reference.md).

---

## 📖 Documentation

Start at [docs/index.md](docs/index.md). Chapters:

| Chapter | |
|---|---|
| [Concepts & Architecture](docs/concepts.md) | [Primitives](docs/primitives.md) |
| [Lists & Arrays](docs/lists.md) | [Dictionaries & Objects](docs/dicts.md) |
| [Nesting](docs/nesting.md) | [The Metric Function](docs/metric.md) |
| [Referential Alignment](docs/referential.md) | [Feedback](docs/feedback.md) |
| [Attribution](docs/attribution.md) | [Repair](docs/repair.md) |
| [Describe](docs/describe.md) | [Null Handling](docs/null_handling.md) |
| [Confidence](docs/confidence.md) | [Semantic Similarity](docs/semantic.md) |
| [Schema Reference](docs/schema_reference.md) | [API Reference](docs/api.md) |

---

## 🧪 Development

```bash
uv sync          # install dependencies
uv run pytest    # run the test suite
```

The repository ships a comprehensive pytest suite under `tests/` covering primitives, lists,
dicts, nesting, referential alignment, feedback, repair, attribution, and edge cases.

---

## 📜 Citation

If you use Object Aligner in academic work, please cite the paper (in preparation):

```bibtex
@misc{drchal2026objectaligner,
  title  = {Object Aligner: A Configurable JSON Schema Similarity Score for Graphs,
            Applied to LLM Prompt Optimization},
  author = {Drchal, Jan},
  year   = {2026},
  note   = {Reference implementation: https://github.com/aic-factcheck/object_aligner}
}
```

---

## 📝 License

[MIT](LICENSE)

---

## 🕰️ History

This is a cleaned-up, standalone version of the Object Aligner originally developed as part of
the [PromptOpt](https://github.com/aic-factcheck/prompt_opt) prompt-optimization framework. The
original implementation can be found in the
[first commit](https://github.com/aic-factcheck/prompt_opt/commit/a6575fa4255b6217551a6303cb073674c4c2bd86)
of PromptOpt (Dec 20, 2024).

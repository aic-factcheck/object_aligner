# 🕸️ Referential Alignment

[Docs](index.md) › Referential Alignment

This chapter covers **referential alignment** — a scoring mode that is
**relabel-invariant**: selected primitive fields are treated as **id handles**
whose concrete values are arbitrary, so the score does not depend on the
particular identifiers each side chose. Object Aligner discovers a one-to-one
mapping (a *bijection*) between the gold ids and the candidate ids for each
*id scope* you declare, and compares *references* through that mapping instead
of by raw value equality. In the paper's vocabulary, a scope is a named set of
**records** (the items bearing an `idScope` field), like primary and foreign
keys in a relational model.

This lets you match graphs and multi-graphs (parent/child relations, group
memberships, citation networks, …) even when the two sides assign different
id values to the same conceptual entities.

---

## Motivation

Imagine you're evaluating a model that extracts family relations:

```python
gold = {
    "people":   [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
    "relations":[{"source": 1, "target": 2, "type": "knows"}],
}
pred = {
    "people":   [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}],
    "relations":[{"source": 53, "target": 124, "type": "knows"}],
}
```

The two objects describe **the same graph**, but a naive primitive-by-primitive
comparison would penalize the differing id values. Referential alignment fixes
this: by marking `people[*].id` as an id definer and `relations[*].source` /
`target` as references, the aligner derives the bijection `1 ↔ 53, 2 ↔ 124`
and scores both relations as perfect matches.

---

## Schema syntax

Two new schema keywords:

| Keyword | Where | Value |
|---------|-------|-------|
| `idScope` | a `string` / `integer` / `number` primitive inside an array | scope name (string) |
| `ref`     | a `string` / `integer` / `number` primitive anywhere | scope name (string) |

Each scope has **exactly one** `idScope` declaration and **zero or more**
`ref` sites of the same primitive type. Booleans cannot bear `idScope` or
`ref`. The `idScope` site must live inside an array (because its definers
form an alignable list).

`score` / `threshold` declared on `idScope` / `ref` fields are ignored
(with a `UserWarning`); these fields are matched symbolically through the
mapping, not via a primitive metric.

See [Schema Reference](schema_reference.md#referential-ids-idscope-ref) for the keyword summary.

---

## Shared setup for the examples

Every example below builds its own `schema`, `gold`, and `pred` (each
example illustrates a different graph shape) but shares the import:

```python
from object_aligner import ObjectAligner
```

---

## Example 1 — parent/child relations graph

A family graph in which arbitrary integers serve as person ids:

```python
schema = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
            },
        },
        "relations": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer", "ref": "person"},
                    "target": {"type": "integer", "ref": "person"},
                    "type":   {"type": "string"},
                },
            },
        },
    },
}

gold = {
    "people": [
        {"id": 1, "name": "Alice", "age": 52},
        {"id": 2, "name": "Bob",   "age": 54},
        {"id": 3, "name": "Cindy", "age": 26},
        {"id": 4, "name": "Dave",  "age": 78},
    ],
    "relations": [
        {"source": 1, "target": 3, "type": "parent_of"},
        {"source": 2, "target": 3, "type": "parent_of"},
        {"source": 3, "target": 1, "type": "child_of"},
        {"source": 3, "target": 2, "type": "child_of"},
        {"source": 4, "target": 1, "type": "parent_of"},
        {"source": 1, "target": 4, "type": "child_of"},
    ],
}

pred = {
    "people": [
        {"id": 44,  "name": "Cindy", "age": 26},
        {"id": 53,  "name": "Alice", "age": 52},
        {"id": 99,  "name": "Dave",  "age": 78},
        {"id": 124, "name": "Bob",   "age": 54},
    ],
    "relations": [
        {"source": 53,  "target": 44,  "type": "parent_of"},
        {"source": 124, "target": 44,  "type": "parent_of"},
        {"source": 44,  "target": 53,  "type": "child_of"},
        {"source": 44,  "target": 124, "type": "child_of"},
        {"source": 99,  "target": 53,  "type": "parent_of"},
        {"source": 53,  "target": 99,  "type": "child_of"},
    ],
}

ObjectAligner(schema).metric(gold, pred)
# {'score': 1.0}
```

Although `pred` uses entirely different id values (and lists the people in a
different order), the structure of the graph is identical to `gold`, so the
score is `1.0`.

---

## Example 2 — multi-graph with `members` arrays

Here references appear as bare ids inside an array — no surrounding object
needed.

```python
schema = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
            },
        },
        "research_groups": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string"},
                    "members": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    },
}

gold = {
    "people": [
        {"id": 1, "name": "Alice", "age": 40},
        {"id": 2, "name": "Bob",   "age": 35},
        {"id": 3, "name": "Cindy", "age": 18},
        {"id": 4, "name": "Dave",  "age": 73},
    ],
    "research_groups": [
        {"name": "stem cell", "members": [1, 2, 3]},
        {"name": "quantum",   "members": [2, 3, 4]},
    ],
}

pred = {
    "people": [
        {"id": 44,  "name": "Cindy", "age": 18},
        {"id": 53,  "name": "Alice", "age": 40},
        {"id": 99,  "name": "Dave",  "age": 73},
        {"id": 124, "name": "Bob",   "age": 35},
    ],
    "research_groups": [
        {"name": "stem cell", "members": [53, 124, 44]},
        {"name": "quantum",   "members": [124, 44, 99]},
    ],
}

ObjectAligner(schema).metric(gold, pred)
# {'score': 1.0}
```

The `members` array is matched in reorder mode, so members can appear in any
order in the candidate; the bijection on the `person` scope makes both
multi-graphs identical.

---

## Example 3 — definer list contains refs (cross-scope dependency)

This is the case the documentation analysis specifically called out. A paper
has its own id (definer of scope `paper`) but also contains an `authors`
array of references to `person`. The aligner notices that scope `paper`'s
definer subtree contains refs to `person`, so it derives the `person`
mapping **first**, and then uses it when scoring the paper-pair cost matrix.

```python
schema = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array", "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                },
            },
        },
        "papers": {
            "type": "array", "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":      {"type": "integer", "idScope": "paper"},
                    "title":   {"type": "string"},
                    "authors": {
                        "type": "array", "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
        "citations": {
            "type": "array", "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer", "ref": "paper"},
                    "target": {"type": "integer", "ref": "paper"},
                },
            },
        },
    },
}

gold = {
    "people":    [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
    "papers":    [{"id": 100, "title": "X", "authors": [1]},
                  {"id": 200, "title": "Y", "authors": [1, 2]}],
    "citations": [{"source": 100, "target": 200}],
}

pred = {
    "people":    [{"id": 53,  "name": "Alice"}, {"id": 124, "name": "Bob"}],
    "papers":    [{"id": 999, "title": "Y", "authors": [124, 53]},
                  {"id": 888, "title": "X", "authors": [53]}],
    "citations": [{"source": 888, "target": 999}],
}

ObjectAligner(schema).metric(gold, pred)
# {'score': 1.0}
```

Notice that two papers in `pred` have swapped order *and* completely
different ids. With property-only alignment, the two papers would be hard to
distinguish if their titles collided; the resolved `person` mapping
disambiguates by author set.

The topological order is determined automatically from the schema — you
don't list the scope dependencies anywhere.

---

## Example 4 — dangling pred ref

A candidate reference that points at an id not present in `pred`'s definer
list scores `0` in place — it does not raise.

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
                    "source": {"type": "integer", "ref": "person"},
                    "target": {"type": "integer", "ref": "person"},
                    "type":   {"type": "string", "score": "exact"},
                }}},
    },
}
gold = {
    "people": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
    "relations": [{"source": 1, "target": 2, "type": "knows"}],
}
pred = {
    "people": [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}],
    "relations": [{"source": 53, "target": 9999, "type": "knows"}],  # 9999 dangles
}

ObjectAligner(schema).metric(gold, pred)
# {'score': 0.8333333333333333}  — the 9999 ref position contributes 0, the rest match
```

Dangling refs in **gold** are treated as a hard `jsonschema.ValidationError`
because gold defines the contract.

---

## Example 5 — bijection failure (size mismatch on the definer list)

If `pred` defines more or fewer entities than `gold`, the bijection becomes
partial:

```python
gold_people = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
pred_people = [
    {"id": 53,  "name": "Alice"},
    {"id": 124, "name": "Bob"},
    {"id": 999, "name": "Eve"},  # excess
]
```

The Hungarian assignment pairs `1 ↔ 53` and `2 ↔ 124`; `999` is "excess"
and tracked separately. Any pred reference to `999` will compare against
`mapping[some_gold_id]` and never match — so each such ref scores `0`. The
overall metric drops below `1.0` as a result.

The same applies symmetrically to missing candidates (gold has an entity
absent from pred): the corresponding `mapping[gold_id] = None`, so every
gold reference involving that id scores `0`.

---

## Example 6 — property-twins and structural disambiguation

This example uses a different schema (people referenced from group member
lists):

```python
schema = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "integer", "idScope": "person"},
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
            },
        },
        "groups": {
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string"},
                    "members": {
                        "type": "array",
                        "order": "align",
                        "items": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    },
}
```

When two definer items share all of their own non-id, non-ref properties the
property cost matrix has tied rows. The paper calls such items
**property-identical twins** (or just *twins*). By default Object Aligner runs
**Weisfeiler–Leman (WL) color refinement** over the same-scope ref graph and
uses the resulting structural color to break those ties, so twins that are
*structurally* distinct align deterministically:

```python
gold = {
    "people": [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Alice", "age": 30},
    ],
    "groups": [{"name": "g", "members": [1]}],
}
pred = {
    "people": [
        {"id": 10, "name": "Alice", "age": 30},
        {"id": 20, "name": "Alice", "age": 30},
    ],
    "groups": [{"name": "g", "members": [10]}],
}
aligner = ObjectAligner(schema)        # id_disambiguation="wl" (the default)
aligner.metric(gold, pred)
# {'score': 1.0}
```

Only one Alice is a group member, so the two are structurally distinguishable:
WL pins `1↔10` and `2↔20` and the membership list matches.

A **genuine automorphism** — where swapping the twins is itself a valid
isomorphism — cannot (and should not) be resolved. WL leaves it ambiguous, and
the `warn_on_ambiguous_mapping` flag now fires only on this *residual* case:

```python
gold = {
    "people": [{"id": 1,  "name": "Alice", "age": 30},
               {"id": 2,  "name": "Alice", "age": 30}],
    "groups": [{"name": "g", "members": [1, 2]}],      # both Alices in the group
}
pred = {
    "people": [{"id": 10, "name": "Alice", "age": 30},
               {"id": 20, "name": "Alice", "age": 30}],
    "groups": [{"name": "g", "members": [10, 20]}],
}
aligner = ObjectAligner(schema, warn_on_ambiguous_mapping=True)
aligner.metric(gold, pred)
# UserWarning: Residual ambiguous mapping in idScope 'person' after WL
#              refinement: gold ids [1, 2] remain structurally
#              indistinguishable (graph automorphism or 1-WL blind spot);
#              arbitrary assignment used.
```

Either pairing yields the same score here, so the residual ambiguity is
harmless. For property-only cost with an arbitrary tie-break and no structural
refinement, construct the aligner with `id_disambiguation="none"`.

---

## Algorithm summary

1. **Schema pre-pass** at construction time: locate every `idScope` and
   `ref`; build a dependency graph in which scope *A* depends on scope *B*
   iff *A*'s definer subtree contains refs to *B*; topologically sort.
   Cycles trigger a `UserWarning` and fall back to property-only alignment
   for the cycle members.
2. **Gold validation** at the start of each call: id uniqueness per scope
   and ref resolvability are enforced; violations raise
   `jsonschema.ValidationError`.
3. **Mapping derivation per scope, in topological order**: build the n×m
   cost matrix between gold and pred definer items using the regular
   alignment machinery, with the current scope's id field masked (so two
   nodes are compared by everything *except* their id). Same-scope refs are
   not compared directly (that would bootstrap the mapping on itself);
   instead, when `id_disambiguation="wl"` (the default), they seed a
   **Weisfeiler–Leman color** computed independently per side. Because the
   colors are a pure function of each side's local ref structure, no
   cross-side mapping is needed to compute them — the bootstrap is avoided.
   The structural color enters the cost matrix according to `wl_integration`:
   `"tie_break"` (default) breaks only *exact* ties in the property cost, so
   already-determined alignments never move; `"blend"` mixes property cost
   and structural agreement with weight $\lambda$ (`wl_blend_lambda`). Refs to
   already-resolved higher scopes use those scopes' mappings (and feed the WL
   color as relation labels). Run the Hungarian algorithm on `-cost` and read
   off the bijection. Passing `id_disambiguation="none"` restores the prior
   behavior exactly: same-scope refs are masked to `1.0`, the cost is
   property-only, and ties are broken arbitrarily.
4. **Main alignment pass**: the regular recursive alignment runs as
   before, with two overrides — `idScope` fields contribute `1.0`
   (treated as labels) and `ref` fields score `1.0` iff
   `mapping[scope][gold_id] == pred_id` and `pred_id` is in the pred id
   set.

---

## Limitations

- **Property-twin ambiguity.** When two definer items are indistinguishable
  by their own non-id, non-ref properties, the default Weisfeiler–Leman
  refinement resolves them whenever they are *structurally* distinct (e.g.
  one node has an edge the other does not — see Example 6). Only **genuine
  automorphisms** (where swapping the twins is itself a valid isomorphism)
  and the known **1-WL blind spots** (such as one 6-cycle versus two disjoint
  3-cycles — both 2-regular, so 1-WL assigns every vertex the same color)
  remain non-unique; for those the assignment is arbitrary but
  score-invariant, and `warn_on_ambiguous_mapping=True` reports the residual
  case. `id_disambiguation="none"` disables refinement entirely and uses
  property-only cost.
- **Cycles in the scope dependency graph.** Each cycle member is aligned
  using non-ref properties only; the warning notes the affected scopes.
- **Primitives only.** Ids must be `string`, `integer`, or `number`. Tuple
  ids (e.g. composite keys made of two fields) are not supported in this
  version.
- **`keyScore` is not a reference.** Dict keys can't carry `idScope` or
  `ref` — only values can.
- **Schema walk coverage.** `idScope` and `ref` are discovered only under
  `properties`, `items`, and `prefixItems`. Declarations buried inside
  `allOf`, `anyOf`, `oneOf`, `$ref`, `additionalProperties`, or
  `patternProperties` are not picked up — such a `ref` is **silently
  ignored** (scored as an ordinary primitive, with no `idScope`/`ref`
  semantics and no warning), rather than raising at construction. Only a
  *discovered* `ref` (one reachable through `properties`/`items`/`prefixItems`)
  whose scope name is undefined raises `'ref' to undefined idScope`.
- **Per-call context.** Each `ObjectAligner.align()` / `metric()` call
  creates its own `_AlignContext` carrying the per-call referential
  bookkeeping (current mappings, pred id sets, masking flags). No state
  is mutated on the instance, so concurrent calls on a single aligner
  are safe.

---

## Transferable feedback for reference errors

`aligner.feedback()` describes a mis-routed `ref` by the candidate's own ids by
default (`change to 'p1'`). For prompt-optimizer reflection slots that is a
per-sample renumbering, not a reusable lesson. Pass
`referential_feedback="semantic"` (constructor or per-call) to instead describe
the **gold endpoint the reference should connect to** by its discriminative
properties and the relation label — e.g. _"this ':ARG0' reference should point
to the node with concept 'confirm-01', not the node you used (concept
'protein')."_ The score is unchanged; only the feedback text differs. See
[Feedback — Example 7.5](feedback.md#example-75-transferable-referential-feedback-referential_feedbacksemantic).

## See also

- [Schema Reference — Referential ids](schema_reference.md#referential-ids-idscope-ref)
- [The Metric Function](metric.md) for the surrounding API and debug/description output.
- [Feedback](feedback.md) — prompt-optimizer feedback, including the semantic referential mode.
- [`api.md`](api.md) — generated API reference.

[← Documentation home](index.md)

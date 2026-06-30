# CLAUDE.md

Project knowledge for AI coding assistants.

## Commit policy

- **Do not add `Co-Authored-By` trailers (or any other AI-attribution trailer) to commit messages.** Commits should carry only the human author. This applies to every commit, every time.

## Documentation conventions

- **Typeset equations in Markdown files with LaTeX math**: `$…$` for inline and `$$…$$` for display blocks. Do not use ASCII art, plain-text formulas, or unicode-math approximations. Renders natively in GitHub, MkDocs (`pymdownx.arithmatex`), Pandoc, and VS Code preview.

## Documentation conventions for `docs/`

These rules cover how to structure new (and edit existing) chapters under `docs/`. The shape was settled with the cleanup that introduced `docs/api.md` and `docs/describe.md`; do not regress it.

- **Chapter shape** mirrors `docs/feedback.md`: intro paragraph → quickstart → "Shared setup for the examples" → "The model" → numbered examples → "API reference" (short summary that links into `docs/api.md`, never duplicating signatures) → "Caveats" → "See also" → bottom back-link to `index.md`. A "Future work" / "Roadmap" section is **never** part of a chapter.
- **Top and bottom back-links.** Every chapter under `docs/` (except `index.md`) opens with a breadcrumb `[Docs](index.md) › <Chapter title>` immediately under the H1, and ends with `[← Documentation home](index.md)` after the See also block. For `docs/api.md` this is injected by `scripts/gen_api_docs.py` (do not hand-edit).
- **Examples must be paste-runnable.** Each chapter has at most one "Shared setup for the examples" block early on that defines a schema, an `aligner`, and a `gold` / `pred` pair. Subsequent examples may reuse those names directly. Any example that uses a *different* schema must build it inline (its own imports and `ObjectAligner(...)` call) and the prose must say so explicitly ("This example uses a different schema:").
- **Roadmap.** Forward-looking / "not committed yet" notes live in the private `research/` directory (untracked). Do not add Future-work sections to chapter files or to `CLAUDE.md`.
- **`research/` is private.** The `research/` directory is untracked development scratch space (design notes, internal decisions, the roadmap). It MUST NOT be linked or named from anything that ships publicly: no Markdown links from `docs/`, no mentions inside `docs/api.md`, no Markdown links or backtick references in any docstring under `src/object_aligner/` (those land in the generated `docs/api.md`). `CLAUDE.md` itself may reference `research/` because it is internal project knowledge for contributors. Enforce with `grep -RIn "research/" docs/ src/object_aligner/` returning zero hits.
- **API reference is generated plain Markdown.** `docs/api.md` is produced by `scripts/gen_api_docs.py` — stdlib-only Python that introspects `object_aligner.__all__`, parses Google-style docstrings, and emits headings + signature blocks + parameter bullets + field tables as ordinary Markdown. Do not hand-edit the file. To update it: (a) edit the corresponding `"""..."""` block in `src/object_aligner/` using Google style (`Args:` / `Returns:` / `Raises:` / `Attributes:` sections); (b) run `uv run python scripts/gen_api_docs.py`; (c) commit both the source change and the regenerated `docs/api.md`. Anchors are deterministic GitHub-style slugs (`#objectaligner`, `#objectalignermetric`, `#load_templates_from_toml`); chapter cross-links use those. For classes whose constructor parameters are documented on `__init__` (not the class docstring itself), the generator falls back to `__init__`'s `Args:` block so constructor arguments still render under the class heading.
- **Site build (optional HTML bonus).** `pyproject.toml` declares an optional `[project.optional-dependencies] docs` group (`mkdocs`, `mkdocs-material`). Install once with `uv sync --extra docs`, then `uv run mkdocs serve` for a live preview at `http://127.0.0.1:8000/` or `uv run mkdocs build` for a static site under `./site/`. The HTML build is a convenience; the source of truth is the Markdown files themselves. The canonical gate is "build emits no anchor warnings and no missing-doc-file warnings". `--strict` is **not** the gate: any incidental cross-folder link mkdocs cannot resolve is downgraded via `validation.links.not_found: ignore` in `mkdocs.yml`.
- **Nav upkeep.** Adding a new chapter under `docs/` requires also adding it to the `nav:` list in `mkdocs.yml`. Without that the file is reachable on GitHub but invisible in the rendered site.

## Project Overview

**object-aligner** is a Python library for computing similarity scores between structured data objects (JSON-like: dicts, lists, primitives). It aligns a "gold" (reference) object with a "predicted" object and produces a fine-grained similarity score in `[0, 1]`, with optional human-readable descriptions and optional structured debug output.

## Tech Stack

- **Language:** Python 3.13+
- **Package manager:** uv
- **Build system:** uv_build
- **Dependencies:** numpy, rapidfuzz, jsonschema, scipy
- **Test framework:** pytest

## Project Structure

```
├── pyproject.toml                    # Project metadata, dependencies, build config
├── uv.lock                           # Locked dependency versions
├── src/
│   └── object_aligner/
│       ├── __init__.py               # Re-exports ObjectAligner
│       └── object_aligner.py         # All core code (single module)
├── scripts/
│   └── demo.py                       # Demo script
├── notebooks/
│   └── playground_object_aligner.ipynb  # Legacy playground notebook (messy, older version)
├── docs/
│   ├── index.md                      # Documentation home
│   ├── concepts.md                   # Core abstractions & architecture
│   ├── primitives.md                 # String, number, boolean alignment
│   ├── lists.md                      # Array alignment (fixed, reorder, prefix)
│   ├── dicts.md                      # Dictionary/object alignment
│   ├── nesting.md                    # Complex nested structure examples
│   ├── metric.md                     # The metric() function & evaluation
│   └── schema_reference.md           # Complete schema keyword reference
├── tests/                            # Pytest suite
└── README.md
```

## Key Concepts

- **ObjectAligner** — Main class, constructed with:
  - `ObjectAligner(schema, *, custom_metrics=None, generate_description=False, description_templates=None, description_style="default", generate_feedback=False, feedback_templates=None, feedback_style="gepa", referential_feedback="literal", dominant_fraction_threshold=0.60, warn_on_ambiguous_mapping=False, compute_confidence=False, confidence_method="margin", confidence_entropy_temperature=8.0, id_disambiguation="wl", wl_integration="tie_break", wl_rounds=None, wl_blend_lambda=0.5)`
- Primary methods:
  - `align(gold, pred, skip_validation=False)` → returns a `MatchItem` / `MatchList` / `MatchDict` tree
  - `metric(gold, pred, debug=False, generate_description=None, generate_feedback=None)` → returns `{"score": float}` by default, optionally adding `"description"`, `"feedback"`, and/or `"debug"`
- **Schema** — JSON Schema–inspired dict that defines data structure plus custom scoring keywords (`score`, `threshold`, `order`, `keyScore`, `valueWeight`, etc.)
- **Match types** — `MatchItem` (primitives), `MatchList` (arrays), `MatchDict` (objects) — frozen dataclasses with Python `float` scores
- **Custom metrics** — user-supplied named metric callables registered through `custom_metrics`, referenced declaratively from schema `score`

## Primitive Scoring

### Built-in string metrics

Supported string `score` values:
- `exact`
- `jaro` *(default)*
- `jaro_winkler`
- `levenshtein`
- `damerau_levenshtein`
- `osa`
- `indel`
- `lcsseq`

### Built-in numeric metrics

Supported number/integer `score` values:
- `exact`
- `invdiff` *(default)*
- `relative` — scale-invariant `1 - min(1, |a-b| / max(|a|, |b|))`, equal values (incl. `0` vs `0`) score `1.0`

### Custom primitive metrics

- Supported for schema types: `string`, `number`, `integer`
- Registered via:
  - `custom_metrics={"string": {...}, "number": {...}, "integer": {...}}`
- Metric callable contract:
  - `(gold, pred) -> float`
  - must return a real number in `[0, 1]`
- Integer metric lookup order:
  1. built-ins
  2. custom `number` metrics
  3. custom `integer` metrics override same-name `number` metrics
- Boolean scoring remains exact-only
- Object `keyScore` is still only `"exact"` or `"jaro"`

## Architecture

The alignment dispatcher `_align_helper` routes by Python type:
- `bool` → `_align_booleans` (exact only)
- `int/float` → `_align_numbers` (registry-based primitive lookup)
- `str` → `_align_strings` (registry-based primitive lookup)
- `list` → `_align_lists` → dispatches to:
  - `_align_lists_prefix` (positional prefix items with weights)
  - `_align_lists_fixed` (DP sequence alignment, order preserved)
  - `_align_lists_reorder` (Hungarian algorithm via scipy, order-free)
- `dict` → `_align_dicts` (Hungarian key matching + recursive value alignment)

All branches are recursive — any nesting depth works naturally.

## Public API Behavior

- `align()` validates by default unless `skip_validation=True`
- `metric()` always validates `gold`
- if `pred` fails validation, `metric()` returns `score: 0.0`
  - with description disabled: `{"score": 0.0}`
  - with description enabled: `{"score": 0.0, "description": ...}`
- `metric(..., debug=True)` adds a structured `"debug"` alignment tree using only basic Python container/scalar types
- public `score` values should be plain Python `float`, not NumPy scalar types
- `ObjectAligner(..., warn_on_ambiguous_mapping=False)` emits a `UserWarning` when id-mapping derivation is ambiguous (off by default). Under the default `id_disambiguation="wl"` it fires only on *residual* (post-WL) ambiguity — genuine automorphisms / 1-WL blind spots; under `id_disambiguation="none"` it fires on any raw cost tie (legacy text)
- `aligner.attribute(gold, pred, *, granularity="leaf", include_empty_positions=False, skip_validation=False)` returns an `AttributionResult` decomposing the deficit into per-path contributions (tree-walk; see `docs/attribution.md`). `aligner.attribute_from_match(match_tree, ...)` skips re-alignment.
- `aligner.repair(gold, pred, *, granularity="leaf", min_contribution=0.0, skip_validation=False)` returns a `RepairResult` with ranked `RepairOp`s (RFC 6902-flavored `add`/`remove`/`replace` with `score_delta`); see `docs/repair.md`. `RepairResult.apply_to(pred)` returns a patched deep copy. `aligner.repair_from_match(match_tree, gold, pred, mappings, ...)` skips re-alignment.
- `aligner.feedback(gold, pred, *, top_k=5, min_score_delta=0.0, style=None, include_synthesis_line=True, include_metadata=False, dominant_fraction_threshold=None, granularity="leaf", skip_validation=False, referential_feedback=None)` returns a `FeedbackResult` with a top-K prescriptive feedback string for prompt-optimizer reflection slots; see `docs/feedback.md`. Deterministic, no LLM. `aligner.feedback_from_match(match_tree, gold, pred, mappings, ...)` skips re-alignment. Constructor accepts `generate_feedback=False`, `feedback_templates=None`, `feedback_style="gepa"`, `referential_feedback="literal"`, `dominant_fraction_threshold=0.60`. `metric(..., generate_feedback=True)` adds a `"feedback"` string key; `generate_feedback="full"` adds the structured dict shape from `FeedbackResult.to_dict()`. Any other value raises `ValueError`.
- `ObjectAligner(..., referential_feedback="literal"|"semantic")` (also a per-call `feedback()` / `feedback_from_match()` override, default `None` → instance default) selects how `feedback()` renders `ref` / `idScope` mismatches. `"literal"` (default) uses opaque ids — byte-identical to earlier releases. `"semantic"` instead describes the gold endpoint node the reference should connect to by its discriminative direct-child scalar props (the same fields the referential cost matrix compares) plus the relation/edge label read from the prediction's carrier; only `feedback().text` changes (scores / `metric` / `attribute` / `repair` are untouched). Falls back to the literal line per-op when the gold endpoint has no discriminator (hedges on property-twins), is a no-op on schemas without `ref`/`idScope`, and never raises. Descriptors are built by `_build_ref_endpoint_descriptors` (the `_ReferentialFeedbackMixin` in `src/object_aligner/_aligner_reffeedback.py`, reusing `_walk_data` / `_get_schema_node` / `_carrier_path` / `_exact_scalars`); rendering uses the `feedback.refsem.*` fragments + `feedback.op.{ref_fix,ref_fix_no_target}.semantic` skeletons via `_render_ref_semantic` in `src/object_aligner/feedback.py`. Constructor validation mirrors the `feedback_style` pattern.
- `aligner.describe(gold, pred, *, style=None, skip_validation=False)` returns a `DescriptionResult` with `.text` (deterministic indented prose walk of the match tree) and `.entries` (one `DescriptionEntry` per visited node — `path`, `depth`, `match_kind`, `outcome`, `score`, `text`); see `docs/describe.md`. Deterministic, no LLM. Two styles ship: `"default"` (prose) and `"json"` (empty `.text`, populated `.entries`). `aligner.describe_from_match(match_tree, ...)` skips re-alignment (no `mappings` needed — describe walks the match tree only). Constructor accepts `generate_description=False`, `description_templates=None`, `description_style="default"`. `metric(..., generate_description=True)` adds a `"description"` string key; `generate_description="full"` adds the structured dict shape from `DescriptionResult.to_dict()`. Any other value raises `ValueError`.
- `MatchList.kind` is `"reorder"` / `"fixed"` / `"prefix"` / `"combined"` (default `""`); used by attribution, repair, and describe (for path-emission rules) to select the per-aggregator α schedule. Live in `src/object_aligner/attribution.py`, `src/object_aligner/repair.py`, and `src/object_aligner/describe.py`.
- Template validation (unknown keys, bad placeholders, non-string values) is shared between `description_templates` and `feedback_templates` via `src/object_aligner/_templates.py:validate_templates`. Internal-only; not part of the public API.
- Default template strings live as TOML data under `src/object_aligner/templates/` (`describe.toml`, `feedback.toml`, `feedback.compact.toml`) and are loaded at module-import time via `_templates.py:_load_packaged_template` (uses `importlib.resources` + stdlib `tomllib`). Python source only holds the placeholder allowlists (`_DESCRIPTION_PLACEHOLDERS` in `describe.py`, `_FEEDBACK_PLACEHOLDERS` in `feedback.py`) and the renderer code. Editing template wording is a `.toml` edit, not a code change. `tests/_legacy_template_snapshots.py` holds a frozen byte-identical copy of the shipped defaults so accidental drift is caught by `tests/test_templates.py`.
- `load_templates_from_toml(path)` (public, re-exported from the package root) loads a user-supplied TOML override file and returns a flat `dict[str, str]` suitable for passing as `description_templates=` or `feedback_templates=` to `ObjectAligner(...)`. Accepts both flat (`"feedback.op.key_add" = "..."`) and nested-table (`[feedback.op]`) TOML styles.

## Roadmap

Planned but uncommitted work is tracked in the private `research/`
directory (untracked, contributor-internal). Do not add Future-work
bullets here or in chapter files under `docs/`.

## Custom Schema Keywords (beyond JSON Schema)

| Keyword | Applies to | Values | Default |
|---------|-----------|--------|---------|
| `score` | string, number, integer | built-in metric name or registered custom metric name | `"jaro"` / `"invdiff"` |
| `threshold` | string, number, integer | float | `0.0` |
| `order` | array (items) | `"fixed"`, `"align"` | `"fixed"` |
| `ignoreExcess` | array | bool | `false` |
| `ignoreMissing` | array | bool | `false` |
| `prefixItems` | array | list of schemas | — |
| `prefixWeights` | array | list of floats | all 1.0 |
| `prefixImportance` | array | float | — |
| `restImportance` | array | float | — |
| `keyScore` | object | `"jaro"`, `"exact"` | `"jaro"` |
| `keyThreshold` | object | float | `0.0` |
| `keyImportance` | object | float | `0.0` |
| `valueImportance` | object | float | `1.0` |
| `valueWeight` | object property | float | `1.0` |
| `idScope` | string/integer/number primitive (inside an array) | scope name (string) | — |
| `ref` | string/integer/number primitive | scope name (string) | — |
| `nullScore` | any schema node (primitive, object, or array) | float in `[0, 1]` | `0.0` |

## Referential Alignment

- `idScope: "<name>"` declares a primitive as the definer of a named id scope; exactly one definer per scope, must be inside an array.
- `ref: "<name>"` declares a primitive as a reference into a named id scope; any depth, any number per scope.
- Comparison happens via a discovered bijection between gold and pred ids derived per scope (Hungarian over the definer list with the id field masked); scopes are resolved in topological order of their inter-scope dependencies.
- Cycles in the dependency graph → `UserWarning`; cycle members align using non-ref properties only (and skip WL — see below).
- Gold id duplicates or dangling gold refs raise `jsonschema.ValidationError`; pred-side analogs score 0 in place.
- `ObjectAligner(..., id_disambiguation="wl"|"none", wl_integration="tie_break"|"blend", wl_rounds=None, wl_blend_lambda=0.5)` selects the id-bijection disambiguation strategy. The default `"wl"` runs 1-WL color refinement over the same-scope ref graph, computed independently per side, so attribute-less / property-tied definers align by structure (up-to-renumbering) rather than emission order; `"none"` reproduces the pre-WL behavior byte-for-byte. `tie_break` (default) only breaks exact property-cost ties (sub-gap ε); `blend` mixes property cost and structural agreement as `(1-λ)·cost + λ·w` with `λ=wl_blend_lambda`. The pure refinement (disjoint-union joint refinement with a shared per-round signature→token dict, so per-side colors are comparable without any cross-side mapping → no bootstrapping) lives in `src/object_aligner/_wl.py` (`RefGraph`, `_RefEdge`, `wl_tokens`); the ref-graph construction (`_build_ref_graph`, `_carrier_path`, `_carrier_label`, `_exact_scalars`, `_emit_incidences`) and cost integration (`_apply_wl`) are methods on `ObjectAligner`. `_carrier_path` finds the carrier object owning each ref site (edge `{source,target}` → directed edge; symmetric/k-ary `members` → star to a synthetic hub vertex). 1-WL blind spots (6-cycle vs two 3-cycles) and genuine automorphisms remain residual-ambiguous by design. Constructor validation mirrors the `confidence_method`/`confidence_entropy_temperature` pattern.
- See `docs/referential.md` for worked examples.

## Common Commands

```bash
uv sync                  # Install dependencies
uv run pytest            # Run test suite
uv run python <file>     # Run a Python file in the venv
uv run python -c "..."   # Run inline Python
```

## Testing

A pytest suite is present under `tests/`.

Useful subsets:

```bash
uv run pytest tests/test_primitives.py
uv run pytest tests/test_api.py
uv run pytest
```

## Important Notes

- All core implementation lives in a single module: `src/object_aligner/object_aligner.py`
- `__init__.py` re-exports `ObjectAligner` from the submodule
- The notebook `playground_object_aligner.ipynb` is from an older version and is messy — use only for inspiration
- Dict key matching ignores value types — aligned gold/pred values with different Python types score `0.0` in place (soft-zero; consistent across `align()` and `metric()`). A gold key not declared in the schema's `properties` also soft-zeros (weight `1.0`) and emits a `UserWarning` instead of crashing; recommend `additionalProperties: false` for closed-world scoring
- `ignoreExcess` and `ignoreMissing` are mutually exclusive on the same array node — both set raises `ValueError` at construction (`_validate_ignore_flags` in `_aligner_schema.py`; the combination would reward omitting hard items). When the normalization denominator `D` is `0` (every unmatched entry ignored, or both lists empty), the list scores a vacuous `1.0` — including the fixed-list `n==0`/`m==0` early returns under the matching flag
- `prefixItems` positions absent from both sides are vacuous: excluded from the prefix-weight normalization (identity `metric(g, g) == 1` holds for lists shorter than `prefixItems`) and emitted as `kind="absent"` sentinel children; attribution gives them alpha `0` (emitted with zero weight only under `include_empty_positions=True`)
- Booleans must be checked before numbers in the dispatcher because `isinstance(True, int)` is `True` in Python
- Unsupported primitive metric names should raise clear `ValueError`s rather than relying on `assert`
- Custom metric registry validation happens at construction time
- `MatchItem.kind` is `"id"` for `idScope` fields, `"ref"` for `ref` fields, `"null"` when one or both of gold/pred is `None`, `"absent"` for prefix positions missing from both sides, and `""` (default) otherwise; the debug tree surfaces this as a `"marker"` field when non-empty. The null-aware leaf is produced by `_align_null`, called from `_align_helper` after the `idScope`/`ref` short-circuits but before the type dispatch. `nullScore` (default `0.0`) is consulted only for the asymmetric case; both-`None` always scores `1.0`. The construction-time `_validate_null_scores` walker rejects out-of-range or non-real values via `_iter_schema_children`. Repair emits a corresponding `RepairOp(kind="null_value_replace")`; the dedicated feedback template key is `feedback.op.null_value_replace`; describe emits `describe.null.match` / `describe.null.mismatch`.
- `_align_helper` short-circuits on `idScope` (always scores 1.0) and `ref` (scores via the bijection) **before** the type dispatch, so the order in that method matters — see the comment at the top of the method.
- Per-call referential state (`current_mappings`, `pred_ids`, `gold_ids`, `pred_excess_ids`, `mask_scope`, `mask_all_refs`, `skip_validation`) lives in an `_AlignContext` dataclass that `align()` creates per call and threads through `_align_helper` and the recursive `_align_*` methods; concurrent `align()` / `metric()` calls on the same instance are safe.
- The JSON Schema validator is built once at construction (`self._validator = validator_for(schema)(schema)`) and reused across `align()` / `metric()` calls.
- Weisfeiler–Leman color refinement disambiguates property-twin / attribute-less definer cases and is the default (`id_disambiguation="wl"`); see the Referential Alignment section above. The private `research/` notes track the remaining doors (callable strategy protocol, `ref_informed`, `k_wl`, graded `w`).
- `ObjectAligner(..., compute_confidence=True, confidence_method="margin"|"entropy", confidence_entropy_temperature=8.0)` enables per-pair stability scores at every Hungarian site, surfaced on `MatchItem/MatchList/MatchDict.confidence`, `RepairOp.confidence` (gain-weighted for key-rename pairs), `FeedbackEntry.confidence`, and `DescriptionEntry.confidence`. Consumed by `feedback(rank_by="score_delta"|"expected_gain"|"confidence", include_pairing_ambiguous=False, ambiguity_threshold=0.30)` and `describe(show_confidence=False, include_ambiguous=False, ambiguity_threshold=0.30)`. All flags default off so existing `feedback()` / `describe()` output stays byte-identical. The debug tree emits `confidence` only when the value differs from `1.0`. Helpers live in `_hungarian_confidence` + `_with_confidence` near the top of `src/object_aligner/object_aligner.py`; the ambiguous-pairings emitter is `_emit_pairing_ambiguous` in `src/object_aligner/repair.py`. Full chapter: [`docs/confidence.md`](docs/confidence.md).
- Embedding-based semantic similarity for string fields lives in the opt-in subpackage `object_aligner.semantic` (not re-exported at the top level). Three layers, composed by injection: `Embedder` (protocol) — `HashMockEmbedder` / `DictMockEmbedder` / `OpenAIEmbedder`; `EmbeddingCache` — `InMemoryEmbeddingCache` / `SQLiteEmbeddingCache` (stdlib `sqlite3`, WAL mode); `cosine_similarity_metric(cache, sign_convention="clip"|"affine")` returning the OA-shaped `(gold, pred) -> float` callable. The callable carries `.cache` + `.kind` attributes so `precompute(aligner, *objects)` can walk the schema, find every string node whose `score` resolves to a cache-backed metric, and batch-embed the union of relevant strings in one upstream call. `OpenAIEmbedder` requires the optional `semantic-openai` extra (`pip install object-aligner[semantic-openai]` → adds `openai>=1.0`); the import succeeds always but construction raises a clean `ImportError` without the extra. Tests use the deterministic `HashMockEmbedder` (BLAKE2b → seeded RNG) — never reach `localhost:8333`. Full chapter: [`docs/semantic.md`](docs/semantic.md).

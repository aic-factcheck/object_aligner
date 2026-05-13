# CLAUDE.md

Project knowledge for AI coding assistants.

## Commit policy

- **Do not add `Co-Authored-By` trailers (or any other AI-attribution trailer) to commit messages.** Commits should carry only the human author. This applies to every commit, every time.

## Project Overview

**object-aligner** is a Python library for computing similarity scores between structured data objects (JSON-like: dicts, lists, primitives). It aligns a "gold" (reference) object with a "predicted" object and produces a fine-grained similarity score in `[0, 1]`, with optional human-readable reasoning and optional structured debug output.

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
  - `ObjectAligner(schema, *, custom_metrics=None, generate_reasoning=False, reasoning_templates=None)`
- Primary methods:
  - `align(gold, pred, skip_validation=False)` → returns a `MatchItem` / `MatchList` / `MatchDict` tree
  - `metric(gold, pred, debug=False, generate_reasoning=None)` → returns `{"score": float}` by default, optionally adding `"reasoning"` and/or `"debug"`
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
  - with reasoning disabled: `{"score": 0.0}`
  - with reasoning enabled: `{"score": 0.0, "reasoning": ...}`
- `metric(..., debug=True)` adds a structured `"debug"` alignment tree using only basic Python container/scalar types
- public `score` values should be plain Python `float`, not NumPy scalar types
- `ObjectAligner(..., warn_on_ambiguous_mapping=False)` emits a `UserWarning` when id-mapping derivation has tied costs (off by default)

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
| `keyImportance` | object | float | `1.0` |
| `valueImportance` | object | float | `1.0` |
| `valueWeight` | object property | float | `1.0` |
| `idScope` | string/integer/number primitive (inside an array) | scope name (string) | — |
| `ref` | string/integer/number primitive | scope name (string) | — |

## Referential Alignment

- `idScope: "<name>"` declares a primitive as the definer of a named id scope; exactly one definer per scope, must be inside an array.
- `ref: "<name>"` declares a primitive as a reference into a named id scope; any depth, any number per scope.
- Comparison happens via a discovered bijection between gold and pred ids derived per scope (Hungarian over the definer list with the id field masked); scopes are resolved in topological order of their inter-scope dependencies.
- Cycles in the dependency graph → `UserWarning`; cycle members align using non-ref properties only.
- Gold id duplicates or dangling gold refs raise `jsonschema.ValidationError`; pred-side analogs score 0 in place.
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
- Dict key matching ignores value types — a `ValueError` is raised if aligned gold/pred values have different Python types
- Booleans must be checked before numbers in the dispatcher because `isinstance(True, int)` is `True` in Python
- Unsupported primitive metric names should raise clear `ValueError`s rather than relying on `assert`
- Custom metric registry validation happens at construction time
- `MatchItem.kind` is `"id"` for `idScope` fields, `"ref"` for `ref` fields, and `""` (default) otherwise; the debug tree surfaces this as a `"marker"` field when non-empty.
- `_align_helper` short-circuits on `idScope` (always scores 1.0) and `ref` (scores via the bijection) **before** the type dispatch, so the order in that method matters — see the comment at the top of the method.
- Per-call referential state (`current_mappings`, `pred_ids`, `gold_ids`, `pred_excess_ids`, `mask_scope`, `mask_all_refs`, `skip_validation`) lives in an `_AlignContext` dataclass that `align()` creates per call and threads through `_align_helper` and the recursive `_align_*` methods; concurrent `align()` / `metric()` calls on the same instance are safe.
- The JSON Schema validator is built once at construction (`self._validator = validator_for(schema)(schema)`) and reused across `align()` / `metric()` calls.
- Future v2: Weisfeiler–Lehman color refinement could disambiguate property-twin definer cases (out of scope for v1).

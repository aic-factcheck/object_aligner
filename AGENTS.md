# AGENTS.md

Project knowledge for AI coding assistants.

## Project Overview

**object-aligner** is a Python library for computing similarity scores between structured data objects (JSON-like: dicts, lists, primitives). It aligns a "gold" (reference) object with a "predicted" object and produces a fine-grained similarity score in [0, 1] along with a human-readable explanation of differences.

## Tech Stack

- **Language:** Python 3.13+
- **Package manager:** uv
- **Build system:** uv_build
- **Dependencies:** numpy, rapidfuzz, jsonschema, scipy

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
└── README.md
```

## Key Concepts

- **ObjectAligner** — Main class, constructed with `(id_, schema)`. Has two primary methods:
  - `align(gold, pred, skip_validation=False)` → returns MatchItem/MatchList/MatchDict tree
  - `metric(gold, pred, debug=False)` → returns `{"score": float, "reasoning": str}`
- **Schema** — JSON Schema–inspired dict that defines data structure + custom scoring keywords (`score`, `threshold`, `order`, `keyScore`, `valueWeight`, etc.)
- **Match types** — `MatchItem` (primitives), `MatchList` (arrays), `MatchDict` (objects) — all frozen dataclasses with `score` field in [0, 1]

## Architecture

The alignment dispatcher `_align_helper` routes by Python type:
- `bool` → `_align_booleans` (exact only)
- `int/float` → `_align_numbers` (exact or invdiff: `1/(1+|a-b|)`)
- `str` → `_align_strings` (exact or Jaro similarity via rapidfuzz)
- `list` → `_align_lists` → dispatches to:
  - `_align_lists_prefix` (positional prefix items with weights)
  - `_align_lists_fixed` (DP sequence alignment, order preserved)
  - `_align_lists_reorder` (Hungarian algorithm via scipy, order-free)
- `dict` → `_align_dicts` (Hungarian key matching + recursive value alignment)

All branches are recursive — any nesting depth works naturally.

## Custom Schema Keywords (beyond JSON Schema)

| Keyword | Applies to | Values | Default |
|---------|-----------|--------|---------|
| `score` | string, number, integer | `"jaro"`, `"exact"`, `"invdiff"` | `"jaro"` / `"invdiff"` |
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

## Common Commands

```bash
uv sync                  # Install dependencies
uv run python <file>     # Run a Python file in the venv
uv run python -c "..."   # Run inline Python
```

## Testing

No test framework is currently set up. There are no automated tests yet.

## Important Notes

- All code lives in a single module: `src/object_aligner/object_aligner.py`
- `__init__.py` re-exports `ObjectAligner` from the submodule
- The notebook `playground_object_aligner.ipynb` is from an older version and is messy — use only for inspiration
- `metric()` validates both gold and pred against the schema; if pred fails validation, it returns `score: 0.0`
- Dict key matching ignores value types — a `ValueError` is raised if aligned gold/pred values have different Python types
- Booleans must be checked before numbers in the dispatcher because `isinstance(True, int)` is `True` in Python

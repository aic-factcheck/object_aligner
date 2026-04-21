# Plan: Extensive Testing for Object Aligner

## Context

Object Aligner is a single-module library for computing similarity scores between structured data objects. It currently has **zero automated tests**. This plan covers creating an extensive test suite using pytest (`tests/` directory with ~6–8 modules, plus a regression suite) that validates: every scoring mode, all alignment algorithms, schema validation behavior, nesting, edge cases, and all documented examples.

## Prerequisites

- [ ] Python 3.13+ environment with `uv` installed
- [ ] Current source tree reviewed (`src/object_aligner/object_aligner.py`)

## Assumptions

- pytest will be added as a dev dependency in `pyproject.toml`.
- Float comparisons use `pytest.approx` or explicit tolerance checks.
- Regression tests mirror the documented examples from `docs/` but also go deeper (more edge cases, boundary conditions, and adversarial inputs).

---

## Execution Steps

### Phase 1: Test Framework Setup

**Goal:** Enable running tests via `uv run pytest`.

1. **Add pytest to dev dependencies**
   - Edit `pyproject.toml` to add `dependencies = [ ... ]` under `[dependency-groups]` or `[project.optional-dependencies]` (whichever matches uv conventions).
   - Run `uv sync` to lock the dependency.

2. **Create `tests/` directory layout**
   ```
   tests/
   ├── __init__.py
   ├── test_utilities.py          # Standalone utility functions + match dataclasses
   ├── test_primitives.py         # bool, int/float, string
   ├── test_lists.py              # fixed, reorder, prefix, combined
   ├── test_dicts.py              # key matching, weights, importances, type mismatch
   ├── test_api.py                # align(), metric(), validation, reasoning
   ├── test_nesting.py            # Deep / mixed structures
   └── test_regression.py         # Every documented example from docs/ + extras
   ```

3. **Verify pytest runs**
   - Run `uv run pytest` — should discover 0 tests and pass.

**Verification:** `uv run pytest` returns exit code 0.

---

### Phase 2: Utility Functions & Match Types

**Goal:** Cover un-gated helper functions and frozen dataclasses.

1. **`tests/test_utilities.py`**

   | Test case | Rationale |
   |-----------|-----------|
   | `similarity_exact(5, 5) == 1.0` | Identity |
   | `similarity_exact(5, 6) == 0.0` | Distinct |
   | `similarity_exact("a", "a") == 1.0` | Works for strings too |
   | `similarity_exact("a", "b") == 0.0` | Distinct strings |
   | `similarity_num_inv_diff(50, 50) == 1.0` | Identity |
   | `similarity_num_inv_diff(50, 51) == 0.5` | Unit step |
   | `similarity_num_inv_diff(50, 52) == pytest.approx(1/3)` | Two steps |
   | `similarity_num_inv_diff(50, 100) == pytest.approx(1/51)` | Large diff |
   | `similarity_string_jaro("hello", "hello") == 1.0` | Identity |
   | `similarity_string_jaro("hello", "hallo") > 0` | Near match |
   | `similarity_string_jaro("hello", "world") == 0.0` | No overlap |
   | `MatchItem(score=0.5, gold="a", pred="b")` | Dataclass instantiation |
   | `MatchList(score=0.5, children=[...])` | Dataclass instantiation |
   | `MatchDict(score=0.5, children={...})` | Dataclass instantiation |
   | Frozen dataclass immutability (`match.score = 1.0` must raise `FrozenInstanceError`) | Contract check |

**Verification:** `uv run pytest tests/test_utilities.py` passes.

---

### Phase 3: Primitive Alignment

**Goal:** Exhaustively test `_align_booleans`, `_align_numbers`, `_align_strings`.

1. **`tests/test_primitives.py`**

   **Booleans**
   - `(True, True)` → score `1.0`
   - `(False, False)` → score `1.0`
   - `(True, False)` → score `0.0`
   - `(False, True)` → score `0.0`
   - No schema keywords beyond `{"type": "boolean"}` matter.

   **Numbers (int and float, exact mode)**
   - `(42, 42)` → `1.0`
   - `(42, 43)` → `0.0`
   - `(3.14, 3.14)` → `1.0`
   - `(3.14, 2.71)` → `0.0`

   **Numbers (invdiff mode)**
   - `(50, 50)` → `1.0`
   - `(50, 51)` → `0.5`
   - `(50, 52)` → `pytest.approx(1/3)`
   - `(50, 100)` → `pytest.approx(1/51)`
   - `(0, 0)` → `1.0`
   - `(0, 5)` → `pytest.approx(1/6)`
   - Negative numbers: `(-5, -3)` → `pytest.approx(1/3)`
   - Float invdiff: `(1.0, 1.5)` → `pytest.approx(1/1.5)`

   **Numbers (threshold)**
   - Schema `{"type": "integer", "score": "invdiff", "threshold": 0.5}` → `(50, 51)` score `0.5`, `(50, 52)` score `0.0`
   - Threshold on exact mode: `{"type": "integer", "score": "exact", "threshold": 0.5}` → `(42, 42)` score `1.0`, `(42, 43)` score `0.0` (threshold irrelevant but allowed)

   **Strings (exact mode)**
   - `("hello", "hello")` → `1.0`
   - `("hello", "hallo")` → `0.0`
   - `("", "")` → `1.0`
   - `("a", "")` → `0.0`

   **Strings (jaro mode)**
   - `("hello", "hello")` → `1.0`
   - `("hello", "hallo")` → score `> 0.0`
   - `("hello", "world")` → `0.0`
   - `("MARTHA", "MARHTA")` → high Jaro score (classic transposition case)
   - `("DWAYNE", "DUANE")` → moderate-high Jaro score
   - Unicode strings: `("café", "cafè")` → Jaro score `> 0.0`
   - Empty string: `("", "abc")` → `0.0`
   - `("abc", "")` → `0.0`

   **Strings (threshold)**
   - Schema `{"type": "string", "score": "jaro", "threshold": 0.7}` → a pair scoring `0.65` under Jaro must return `0.0`; a pair scoring `0.75` must return `0.75`.

**Verification:** `uv run pytest tests/test_primitives.py` passes.

---

### Phase 4: List Alignment

**Goal:** Exhaustively test `_align_lists_fixed`, `_align_lists_reorder`, `_align_lists_prefix`, combined prefix+items, and edge cases.

1. **`tests/test_lists.py`**

   **Fixed-order (`order: "fixed"`) — basic**
   - Identical short lists: `[1, 2, 3]` vs `[1, 2, 3]` → `1.0`
   - `[1, 2, 3]` vs `[1, 3, 2]` → lower score because order is preserved
   - `[1, 2]` vs `[1, 2, 99]` → one excess item, score `< 1.0`
   - `[1, 2, 99]` vs `[1, 2]` → one missing item, score `< 1.0`
   - Empty vs empty: `[]` vs `[]` → `1.0`
   - Empty vs non-empty: `[]` vs `[1]` → `0.0`

   **Fixed-order — gaps / DP behavior**
   - `gold = [1, 2, 4]`, `pred = [2, 3]` — from docs, assert exact expected score and structure.
   - `gold = [1, 2, 3, 4]`, `pred = [1, 3, 4]` — skip one middle element.
   - `gold = [1, 2, 3]`, `pred = [2, 3]` — missing prefix.

   **Fixed-order — with primitive scoring**
   - Strings with `score: "jaro"` inside fixed list.
   - Numbers with `score: "invdiff"` inside fixed list.

   **Reorder (`order: "align"`) — basic**
   - Identical lists (any order): `["a", "b", "c"]` vs `["c", "a", "b"]` → score `1.0`
   - Shuffle + tiny typos: `["Python", "JS", "SQL"]` vs `["Pythn", "SQL", "JavaScrypt"]` — from docs.
   - `["weight", "name", "age"]` vs `["name", "ages", "title"]` — from docs (threshold 0.5, excess `"title"`, missing `"weight"`).

   **Reorder — sizes**
   - Both empty → `1.0`
   - Gold empty, pred non-empty → `0.0`
   - Gold non-empty, pred empty → `0.0`
   - 1×1 match: `["a"]` vs `["a"]` → `1.0`
   - 1×1 mismatch: `["a"]` vs `["b"]` → `0.0` (with exact string)
   - 2×1: `["a", "b"]` vs `["a"]` → one missing
   - 1×2: `["a"]` vs `["a", "b"]` → one excess

   **Reorder — thresholds on items**
   - Items with `threshold: 0.5`; ensure below-threshold pairings are reported as mismatches / unaligned.

   **Prefix items (`prefixItems`)**
   - Exact positional schemas: `prefixItems: [{"type":"integer"}, {"type":"string"}]`, gold/pred `[42, "hello"]` → perfect.
   - Prefix mismatch: `[42, "hello"]` vs `[99, "hello"]` → exactly first position is penalized.
   - `prefixWeights`: `[3.0, 1.0]` to test weighted prefix scoring.

   **Prefix + items combined**
   - Both `prefixItems` and `items` present with `prefixImportance` and `restImportance`.
   - Test that combined score is a weighted average.
   - Example: prefix matches perfectly but tail is terrible → score close to `prefixImportance / (prefixImportance + restImportance)`.

   **ignoreExcess / ignoreMissing**
   - `ignoreExcess: true` → extra pred items should not lower denominator.
   - `ignoreMissing: true` → missing gold items should not lower denominator.
   - Combined both true on a mismatched list.

**Verification:** `uv run pytest tests/test_lists.py` passes.

---

### Phase 5: Dictionary Alignment

**Goal:** Exhaustively test `_align_dicts`, key matching, weights, importances, and expected errors.

1. **`tests/test_dicts.py`**

   **Exact key matching (`keyScore: "exact"`)**
   - Identical: `{"a":1, "b":2}` vs `{"a":1, "b":2}` → `1.0`
   - Value differs: `{"a":1, "b":2}` vs `{"a":1, "b":99}` -> `< 1.0`
   - Missing key: `{"a":1}` missing in pred → score `< 1.0`
   - Extra key: pred has `"c":3` not in gold → score `< 1.0`
   - Keys shuffled but exact only: `{"a":1, "b":2}` vs `{"b":2, "a":1}` → still `1.0` because values line up when keys are exact.

   **Fuzzy key matching (`keyScore: "jaro"`)**
   - `{"weight":1, "name":"A"}` vs `{"name":"A", "ages":2, "title":"X"}` — from docs.
   - Typos intentionally matched: `{"colour": "red"}` vs `{"color": "red"}` → Jaro key match > 0, then exact value.

   **keyThreshold**
   - High threshold prevents poor-spelling key alignment: `"colour"` vs `"clr"` threshold 0.7 → no match, treated as missing+excess.
   - Low threshold allows more fuzzy matches.

   **keyImportance / valueImportance**
   - `keyImportance=0.0, valueImportance=1.0` → only values matter.
   - `keyImportance=1.0, valueImportance=0.0` → only keys matter.
   - Unbalanced weights to confirm formula: `(key_importance * keys_score + value_importance * values_score) / (key_importance + value_importance)`.

   **valueWeight**
   - One property weighted 3× and another 1×; confirm weighted average of value scores.
   - Missing/extra properties that receive default weight 1.0 still behave correctly.

   **Missing / extra values**
   - Missing key → value match is `MatchItem(0.0, gold=value, pred=None)`.
   - Extra key → value match is `MatchItem(0.0, gold=None, pred=value)`.
   - Zero keys in gold, non-empty pred → handle gracefully.
   - Zero keys in pred, non-empty gold → handle gracefully.

   **Type mismatch raises ValueError**
   - `{"a": 1}` (int) vs `{"a": "one"}` (string) must raise `ValueError` when keys align (either exact or fuzzy).
   - Both with exact keys and fuzzy keys to ensure the path that does `type(ag) != type(ap)` is always hit.

   **Empty dicts**
   - `{}` vs `{}` → `1.0`
   - `{}` vs `{"a":1}` → `0.0`
   - `{"a":1}` vs `{}` → `0.0`

**Verification:** `uv run pytest tests/test_dicts.py` passes.

---

### Phase 6: API Surface (`align` vs `metric`, Validation, Reasoning)

**Goal:** Cover `ObjectAligner.align()`, `ObjectAligner.metric()`, `get_name()`, `_alignment2reasoning`, and JSON Schema validation.

1. **`tests/test_api.py`**

   | Test | Expected behavior |
   |------|-------------------|
   | `aligner.get_name()` returns `id_` | Direct equality |
   | `align(gold, pred)` returns `MatchItem/List/Dict` depending on schema type | `isinstance` checks |
   | `align()` validates both by default; valid identical objects → perfect match | score `1.0` |
   | `align(gold, pred, skip_validation=True)` skips `jsonschema.validate` | Does not raise on schema-mismatching inputs |
   | `metric(gold, pred)` returns `{"score": float, "reasoning": str}` | Keys exist, `score` in `[0, 1]` |
   | `metric()` with invalid gold raises `ValidationError` | `pytest.raises(ValidationError)` |
   | `metric()` with invalid pred catches error and returns `score: 0.0` | assert `score == 0.0` and `"JSON Schema validation failed" in reasoning` |
   | Perfect match reasoning | `"perfectly matches the gold"` in reasoning |
   | Imperfect match reasoning starts with `"The predicted output scores overall"` | String prefix check |
   | Reasoning for lists mentions excessive / missing items | Substring checks |
   | Reasoning for dicts mentions KEY and VALUE lines | Substring checks |
   | Score bounds after every test case | `0.0 <= score <= 1.0` |

2. **Validation edge cases**
   - Pred violates `required` fields → `score == 0.0`.
   - Pred violates `type` constraint in a nested property → `score == 0.0`.
   - Pred violates `minItems` / `maxItems` if present in schema → `score == 0.0`.
   - Gold violates schema → unhandled `ValidationError` (assert it propagates).

**Verification:** `uv run pytest tests/test_api.py` passes.

---

### Phase 7: Nesting & Complex Structures

**Goal:** Verify recursion works at arbitrary depth and with mixed types.

1. **`tests/test_nesting.py`**

   - **List of dicts** (from docs): Students list with `order: "align"`.
     - `gold = [{"name":"Alice","score":95}, {"name":"Bob","score":82}]`
     - `pred = [{"name":"Alice","score":93}, {"name":"Bobby","score":82}]`
     - Assert overall score around expected value.
   - **Dict of lists**: `{"tags": ["a", "b"], "numbers": [1, 2]}` with independent scoring per property.
   - **Mixed depth 3+**:
     - `{"items": [{"labels": ["x", "y"]}, {"labels": ["z"]}]}`
     - Test fixed order inside reorder inside dict.
   - **Deep nesting sanity**: 5-level nesting (dict → list → dict → list → string). Both identical objects and with intentional typos at the deepest leaf. Ensure score drops only at that leaf.
   - **Prefix items inside nested list**:
     - Outer `order: "align"`, inner list uses `prefixItems`.
   - **All identical nested structure** → score `1.0`.
   - **Completely off nested structure** → score `0.0` or very near it.

**Verification:** `uv run pytest tests/test_nesting.py` passes.

---

### Phase 8: Regression / Documented Examples + Extras

**Goal:** Turn every meaningful example from `docs/` into a test and add adversarial / boundary cases beyond the docs.

1. **`tests/test_regression.py`**

   **From `primitives.md`**
   - Boolean perfect / mismatch.
   - Number exact and invdiff examples.
   - Number threshold example (50 vs 51 passes, 50 vs 52 fails).
   - String Jaro examples.

   **From `lists.md`**
   - Quiz fixed-order `[42, 7, 13]` vs `[99, 7, 13]`.
   - Skills reorder `["Python", "JS", "SQL"]` vs `["Pythn", "SQL", "JavaScrypt"]`.
   - Field-names reorder with threshold `"weight"/"name"/"age"` vs `"name"/"ages"/"title"`.

   **From `dicts.md`**
   - Exact keys person (`name`+`age`) example.
   - Fuzzy keys person (`weight`/`name`/`age` vs `name`/`ages`/`title`) example.

   **From `nesting.md`**
   - Students list example.
   - Product catalog example (if enough detail exists in doc).

   **Adversarial / beyond docs**
   - Lists of length 100 with `order: "align"` and `order: "fixed"` to ensure performance and correctness.
   - Dict with 20+ keys and fuzzy matching — ensure Hungarian handles it.
   - String threshold at boundary: pair scoring exactly `threshold` must be kept (the code uses `< threshold` to zero out).
   - Very large numeric difference (millions) → `invdiff` approaches `0.0`.
   - Very small numeric difference (0.001) → `invdiff` approaches `1.0`.
   - Booleans passed as numbers (`1` / `0`) — if schema says `"boolean"`, `isinstance(g, bool)` check will reject because `bool` is checked first; test that Python `True` is treated as boolean, not number.

**Verification:** `uv run pytest tests/test_regression.py` passes.

---

### Phase 9: Edge Cases & Stress

**Goal:** Catch boundary and structural weirdness.

1. **`tests/test_edge_cases.py`**

   | Case | Detail |
   |------|--------|
   | Identical primitives, lists, dicts | All return `1.0` |
   | Completely disjoint primitives, lists, dicts | Return `0.0` or very close |
   | Single-element list fixed/reorder | `["a"]` vs `["b"]` |
   | Single-key dict exact/fuzzy | `{"x":1}` vs `{"y":1}` |
   | Empty string exact/jaro | `""` vs `""` and `""` vs `"abc"` |
   | Unicode / emoji strings | `{"name": "🔥"}` vs `{"name": "🔥"}` → `1.0`; `{"name": "🔥"}` vs `{"name": "❄️"}` → low score |
   | Float precision | `(1.0000001, 1.0)` with invdiff → very high score |
   | Mixed int/float in number schema | Schema `{"type": "number"}`, compare `1` (int) and `1.0` (float). `_align_helper` branches on `type(g)`; if `g` is `int` but schema says `"number"`, assert it still works (because `isinstance(g, (int, float))` catches both). |
   | Schema `{"type": "integer"}` with float input | Validation should fail (handled by `jsonschema`), not alignment code. Test via `metric()` returning `0.0` if pred is float. |
   | Very deep empty nesting | `{}` inside `[]` inside `{}` repeated 4× — identical and with one missing leaf. |
   | Large homogeneous list | 50 identical strings `"x"` vs 50 identical strings `"x"` → `1.0` under both fixed and reorder. |
   | List where `items` schema has nested object | Reorder alignment of list of objects where one object field uses `valueWeight: 10.0`. |

**Verification:** `uv run pytest tests/test_edge_cases.py` passes.

---

### Phase 10: Final Integration Run & CI Hardening

**Goal:** Make the suite robust and easy to run.

1. **Add `pytest.ini` or `pyproject.toml` pytest config** with standard settings: `testpaths = tests`, `python_files = test_*.py`.
2. **Run the full suite** (`uv run pytest`) and ensure 100 % pass.
3. **Optional: add coverage** — install `pytest-cov` and run `uv run pytest --cov=object_aligner --cov-report=term-missing`. Aim for high coverage; note any uncovered lines for future work.

**Verification:** `uv run pytest` passes; coverage report shows which branches in `object_aligner.py` are hit.

---

## Verification (Global)

- [ ] `uv sync` installs pytest.
- [ ] `uv run pytest` discovers and runs all test modules.
- [ ] All assertions pass.
- [ ] Every public method (`align`, `metric`, `get_name`) and every private alignment path (`_align_booleans`, `_align_numbers`, `_align_strings`, `_align_lists_fixed`, `_align_lists_reorder`, `_align_lists_prefix`, `_align_dicts`) is exercised by at least one test.
- [ ] All documented examples from `docs/` exist as regression tests.
- [ ] Edge-case coverage includes empty containers, perfect matches, total mismatches, unicode, large sizes, deep nesting, and threshold boundaries.

## Rollback Notes

- If anything breaks during setup, revert `pyproject.toml` and `uv.lock`.
- Deleting the `tests/` directory removes the entire test suite without affecting library code.

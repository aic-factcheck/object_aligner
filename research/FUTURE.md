# Future work / roadmap

This file is the single source of truth for planned but uncommitted work in
the `object-aligner` project. Do **not** add Future-work sections to chapter
documents under `docs/`, to `CLAUDE.md`, or anywhere else — link here instead.

Items are organized by feature area, not by source file, so that bullets that
previously appeared in more than one place collapse to a single entry.

---

## Repair

- **`mode="exact"` for `repair()`.** Apply each candidate op, re-run
  `metric()`, surface the true score delta. Captures Hungarian re-pairing
  exactly; one extra `align()` call per candidate. Will ship together with
  `attribute(mode="counterfactual")` since they share the patch-and-evaluate
  primitive. See `research/opus47_json_patch.md` §2 and §5.
- **`move` op support.** Replace the two-op `key_rename_remove` +
  `key_rename_add` pair with a single `op="move"`. Also potentially detect
  list-item misplacements in `order: "fixed"` lists. See
  `research/opus47_json_patch.md` §3.1 and §6.3.
- **Greedy-exact sequential ranking.** A mode that produces a *sequence* of
  patches whose deltas actually compose, by re-running `metric()` between
  applications.

---

## Attribution

- **`mode="counterfactual"`.** Re-run `align()` with each leaf temporarily
  patched to perfect. Trades cost for exact deltas — and surfaces a non-zero
  `residual` reflecting Hungarian re-pairings — without changing the public
  surface. Shares the patch-and-evaluate primitive with `repair(mode="exact")`
  above. See `research/opus47_json_patch.md` §2 and §5.

---

## Feedback

- **DSPy / GEPA adapter modules** (`object_aligner.dspy`,
  `object_aligner.gepa`) as optional installs
  (`pip install object-aligner[dspy]`, `[gepa]`). Thin wrappers exposing
  `feedback()` through each framework's `Metric` / `Reflector` callable
  contract. Kept separate from the core so we don't couple the library to
  framework availability. See `research/opus47_promptopt_feedback.md` §6.
- **`experiments/dspy_gepa_demo.py`** — a runnable demo comparing DSPy + OA
  feedback vs DSPy + LLM-judge rationale, the empirical centerpiece referenced
  in the research document. Kept outside `src/` to avoid coupling the library
  to LLM API availability. See `research/opus47_promptopt_feedback.md` §6.2.
- **`"verbose"` style preset.** A feedback style that additionally includes
  the metric name (`jaro`, `invdiff`) and the α-weight chain that produced
  each contribution. Useful for ablation work; not default. See
  `research/opus47_promptopt_feedback.md` §4.4.

---

## Templates

- **Template-key stability policy** for `feedback_templates` and
  `description_templates`. Keys may be **added** (with sensible English
  defaults so existing user overrides keep working) when the underlying
  op-kind / match-type taxonomy grows; renames and removals are not
  permitted within a major version. See
  `research/opus47_promptopt_feedback.md` §7.5.

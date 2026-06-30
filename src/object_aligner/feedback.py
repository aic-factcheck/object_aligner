"""Prompt-optimizer feedback rendering.

Turns a ``RepairResult`` (produced by ``object_aligner.repair``) into a
top-K ranked, prescriptive, optimizer-shaped feedback string suitable for
DSPy / GEPA / TextGrad reflection slots — *deterministically, no LLM*.

The module is a *renderer*, not a generator: every numeric value
(``score_delta``, ``score_delta_pct``) is read off the ``RepairOp``
records that ``object_aligner.repair.generate_repairs`` already produced.

Three style presets are supported:

* ``"gepa"`` (default) — multi-line, prescriptive, with a trailing
  synthesis line ("Focus on … errors").
* ``"compact"`` — single line per entry, no synthesis by default.
* ``"json"`` — bypasses rendering entirely; ``FeedbackResult.text`` is
  empty and every ``FeedbackEntry.text`` is empty, but all structured
  fields are populated.

Templates are validated at construction time via
``object_aligner._templates.validate_templates``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from object_aligner._templates import _load_packaged_template, validate_templates
from object_aligner.repair import RepairOp, RepairResult, _filter_paired_ops

# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------

__all__ = [
    "FeedbackEntry",
    "FeedbackResult",
    "render_feedback",
    "DEFAULT_FEEDBACK_TEMPLATES",
]


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_VALID_STYLES = ("gepa", "compact", "json")
_DEFAULT_VALUE_REPR_CAP = 80
_DEFAULT_DOMINANT_FRACTION = 0.60

# Internal op_kind -> human-readable label used by the synthesis line.
# `key_rename_add` and `key_rename_remove` collapse to one kind so the
# dominant-kind logic treats the pair as a single error class; ditto the
# ordered vs reorder list variants.
_OP_KIND_HUMAN = {
    "primitive_replace": "primitive-value",
    "primitive_replace_reorder": "list-item-value",
    "key_add": "missing-key",
    "key_remove": "extraneous-key",
    "key_rename_add": "key-rename",
    "key_rename_remove": "key-rename",
    "list_item_add": "missing-list-item",
    "list_item_remove": "extraneous-list-item",
    "list_item_missing": "missing-list-item",
    "list_item_excess": "extraneous-list-item",
    "ref_fix": "reference",
    "ref_fix_no_target": "unresolved-reference",
    "subtree_replace": "subtree",
    "null_value_replace": "null-value",
}


# -----------------------------------------------------------------------------
# Default templates and placeholder allowlists
# -----------------------------------------------------------------------------

# Default templates and compact-style overlay ship as TOML data under
# ``src/object_aligner/templates/``. The Python source only holds the
# placeholder allowlist (the renderer's API contract).
DEFAULT_FEEDBACK_TEMPLATES = _load_packaged_template("feedback.toml")
_COMPACT_OVERRIDES = _load_packaged_template("feedback.compact.toml")

# Optional confidence placeholders. Available on every per-op template
# key — user-supplied overrides may include them; the shipped defaults do
# not, so confidence is omitted from output when compute_confidence=False.
_CONFIDENCE_PLACEHOLDERS = frozenset({"confidence", "confidence_pct"})


def _op_placeholders(*names) -> frozenset:
    """Per-op placeholder set, with the optional confidence pair added."""
    return frozenset(names) | _CONFIDENCE_PLACEHOLDERS


# Placeholders allowed in each template key. Used by construction-time
# validation to reject typos in user-supplied overrides. Keep in sync with
# the rendering code below.
_FEEDBACK_PLACEHOLDERS = {
    "feedback.intro.perfect": frozenset({"score", "score_pct"}),
    "feedback.intro.imperfect": frozenset({
        "score", "score_pct", "deficit", "deficit_pct",
        "n_shown", "n_total",
    }),
    "feedback.op.primitive_replace": _op_placeholders(
        "rank", "path", "gold", "pred", "score_delta", "score_delta_pct",
    ),
    "feedback.op.primitive_replace_reorder": _op_placeholders(
        "rank", "list_path", "gold", "pred", "score_delta", "score_delta_pct",
    ),
    "feedback.op.key_add": _op_placeholders(
        "rank", "path", "key", "gold", "score_delta", "score_delta_pct",
    ),
    "feedback.op.key_remove": _op_placeholders(
        "rank", "path", "key", "pred", "score_delta", "score_delta_pct",
    ),
    "feedback.op.key_rename_add": _op_placeholders(
        "rank", "gold_path", "pred_path", "gold_key", "pred_key",
        "gold", "score_delta", "score_delta_pct",
    ),
    "feedback.op.key_rename_remove": _op_placeholders(
        "rank", "pred_path", "pred_key", "pred",
    ),
    "feedback.op.list_item_add": _op_placeholders(
        "rank", "path", "gold", "score_delta", "score_delta_pct",
    ),
    "feedback.op.list_item_remove": _op_placeholders(
        "rank", "path", "pred", "score_delta", "score_delta_pct",
    ),
    "feedback.op.list_item_missing": _op_placeholders(
        "rank", "list_path", "gold", "score_delta", "score_delta_pct",
    ),
    "feedback.op.list_item_excess": _op_placeholders(
        "rank", "list_path", "pred", "score_delta", "score_delta_pct",
    ),
    "feedback.op.ref_fix": _op_placeholders(
        "rank", "path", "pred", "value", "score_delta", "score_delta_pct",
    ),
    "feedback.op.ref_fix_no_target": _op_placeholders(
        "rank", "path", "pred", "score_delta", "score_delta_pct",
    ),
    # Opt-in semantic referential feedback (referential_feedback="semantic").
    "feedback.refsem.prop": frozenset({"key", "value"}),
    "feedback.refsem.relation": frozenset({"relation_label"}),
    "feedback.refsem.target": frozenset({"scope", "gold_props"}),
    "feedback.refsem.target_ambiguous": frozenset({"scope", "gold_props"}),
    "feedback.refsem.used": frozenset({"scope", "pred_props"}),
    "feedback.refsem.used_dangling": frozenset(),
    "feedback.op.ref_fix.semantic": _op_placeholders(
        "rank", "path", "scope", "relation", "target", "used",
        "score_delta", "score_delta_pct",
    ),
    "feedback.op.ref_fix_no_target.semantic": _op_placeholders(
        "rank", "path", "scope", "relation", "target",
        "score_delta", "score_delta_pct",
    ),
    "feedback.op.subtree_replace": _op_placeholders(
        "rank", "path", "score_delta", "score_delta_pct",
    ),
    "feedback.op.null_value_replace": _op_placeholders(
        "rank", "path", "gold", "pred", "score_delta", "score_delta_pct",
    ),
    # Diagnostic-only — pairing_ambiguous has no score_delta and is not a fix.
    "feedback.op.pairing_ambiguous": frozenset({
        "rank", "path", "confidence", "confidence_pct",
    }),
    "feedback.diagnostics.intro": frozenset(),
    "feedback.synthesis.single_dominant": frozenset({
        "dominant_kind", "dominant_kind_human",
        "dominant_fraction", "dominant_fraction_pct",
    }),
    "feedback.synthesis.mixed": frozenset({"top_kinds"}),
    "feedback.empty": frozenset({"score", "score_pct"}),
    "feedback.validation_error": frozenset({"path", "message"}),
}

# Self-check: surface typos in feedback.toml / feedback.compact.toml at
# import time rather than at first render. Treat each shipped file as a
# user-supplied override map; bad placeholders raise here.
validate_templates(
    DEFAULT_FEEDBACK_TEMPLATES,
    DEFAULT_FEEDBACK_TEMPLATES,
    _FEEDBACK_PLACEHOLDERS,
    kind="feedback",
)
validate_templates(
    _COMPACT_OVERRIDES,
    DEFAULT_FEEDBACK_TEMPLATES,
    _FEEDBACK_PLACEHOLDERS,
    kind="feedback",
)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedbackEntry:
    """One rendered feedback line plus its structured backing fields.

    `rank` is the *visible* rank: silenced entries (those whose template
    renders to empty text, like the default `key_rename_remove`) inherit
    the rank of their paired `key_rename_add` so the visible numbering
    stays contiguous. Standalone silenced entries get rank `0`.

    Attributes:
        rank: Visible 1-based rank in the rendered feedback message.
        op_kind: Fine-grained op discriminator (mirrors `RepairOp.kind`).
        op: RFC 6902 op (`"add"` / `"remove"` / `"replace"`).
        path: RFC 6901 JSON Pointer at which the repair would apply.
        score_delta: Approximate deficit closure if the op were applied.
        score_delta_pct: `score_delta * 100`, rounded for display.
        gold: Gold value at the patch site (informational).
        pred: Predicted value at the patch site (informational).
        text: Rendered template body — what the consumer prints.
        pair_id: Non-empty for paired ops that must be applied together.
    """

    rank: int
    op_kind: str
    op: str
    path: str
    score_delta: float
    score_delta_pct: float
    gold: Any
    pred: Any
    text: str
    pair_id: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class FeedbackResult:
    """Result of a single `render_feedback()` call.

    `str(result)` returns `result.text`. Iterable: `for entry in result`
    yields `FeedbackEntry`s in visible-rank order.

    Attributes:
        score: Overall similarity in `[0, 1]`.
        text: Fully rendered feedback string, suitable for pasting into
            a DSPy / GEPA / TextGrad reflection slot.
        entries: Structured backing entries, in visible-rank order.
        style: The style name used to render (`"gepa"`, `"compact"`,
            etc.).
        truncated: `True` if the underlying repair list contained more
            ops than `top_k` allowed.
        n_total_ops: Total candidate ops before top-K truncation.
        error_breakdown: Per-op-kind contribution summary as a dict —
            used to drive synthesis-line phrasing.
    """

    score: float
    text: str
    entries: tuple = field(default_factory=tuple)
    style: str = "gepa"
    truncated: bool = False
    n_total_ops: int = 0
    error_breakdown: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def __str__(self):
        return self.text

    def to_dict(self) -> dict:
        """Serialize to a basic-types dict — usable as JSON.

        Returns:
            Dict with the same shape as `FeedbackResult` but with
            `entries` as a list of dicts. `gold` / `pred` values are
            passed through as-is (caller is responsible for ensuring
            they are JSON-serializable if that matters).
        """
        return {
            "score": self.score,
            "text": self.text,
            "entries": [
                {
                    "rank": e.rank,
                    "op_kind": e.op_kind,
                    "op": e.op,
                    "path": e.path,
                    "score_delta": e.score_delta,
                    "score_delta_pct": e.score_delta_pct,
                    "gold": e.gold,
                    "pred": e.pred,
                    "text": e.text,
                    "pair_id": e.pair_id,
                    "confidence": e.confidence,
                }
                for e in self.entries
            ],
            "style": self.style,
            "truncated": self.truncated,
            "n_total_ops": self.n_total_ops,
            "error_breakdown": dict(self.error_breakdown),
        }


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _validate_style(style: str) -> None:
    if style not in _VALID_STYLES:
        raise ValueError(
            f"style must be one of {_VALID_STYLES!r}, got {style!r}"
        )


def _default_value_formatter(value: Any) -> str:
    """``repr(value)`` truncated to 80 chars with trailing ellipsis."""
    s = repr(value)
    if len(s) > _DEFAULT_VALUE_REPR_CAP:
        # 79 chars + a single-character ellipsis = 80 displayed.
        return s[: _DEFAULT_VALUE_REPR_CAP - 1] + "…"
    return s


def _decode_pointer_token(token: str) -> str:
    """RFC 6901 decode: ``~1`` -> ``/``, then ``~0`` -> ``~`` (order matters)."""
    return token.replace("~1", "/").replace("~0", "~")


def _last_path_token(path: str) -> str:
    """Return the last (decoded) token of a JSON Pointer path, or ``''``."""
    if not path or path == "/":
        return ""
    return _decode_pointer_token(path.rsplit("/", 1)[-1])


def _parent_path(path: str) -> str:
    """Return the parent JSON Pointer path of ``path``, or ``''``."""
    if not path or "/" not in path:
        return ""
    parent = path.rsplit("/", 1)[0]
    return parent


def merge_feedback_templates(user_templates) -> dict:
    """Validate user overrides and merge with defaults.

    Public helper for ``ObjectAligner`` to call at construction time. Returns
    a fully-populated dict (defaults overlaid with user overrides). The
    style preset is **not** applied here — that happens per-call in
    ``render_feedback`` so the same instance can render in multiple styles.
    """
    return validate_templates(
        user_templates,
        DEFAULT_FEEDBACK_TEMPLATES,
        _FEEDBACK_PLACEHOLDERS,
        kind="feedback",
    )


def _build_templates(style: str, user_templates_or_merged) -> dict:
    """Build the final template dict for a render call.

    ``user_templates_or_merged`` may be ``None``, a partial user-overrides
    dict, or a fully-merged dict (the form ``ObjectAligner`` stores). Either
    way we re-merge with the default + compact-overrides baseline to make
    sure the user's choices win over preset choices.
    """
    templates = dict(DEFAULT_FEEDBACK_TEMPLATES)
    if style == "compact":
        templates.update(_COMPACT_OVERRIDES)
    if user_templates_or_merged:
        # Layer only the keys that *differ* from the package defaults — that
        # way a fully-merged dict (where every key equals its default unless
        # explicitly overridden) doesn't undo the compact overrides.
        for key, val in user_templates_or_merged.items():
            if val != DEFAULT_FEEDBACK_TEMPLATES.get(key):
                templates[key] = val
    return templates


def _select_top_k(
    ops: list,
    top_k: int | None,
    min_score_delta: float,
) -> tuple[list, bool]:
    """Apply ``min_score_delta`` and ``top_k`` to a sorted ops list.

    Returns ``(selected, truncated)``. Ops come in sorted descending by
    ``score_delta`` (and tie-broken by path/op/kind) — same as
    ``generate_repairs`` emits.

    Key-rename pairs are atomic: the pair is kept iff the ``add`` half
    passes ``min_score_delta``; ``top_k`` counts each member of a kept
    pair separately (so a pair takes two "slots"). Standalone silenced
    ops (``key_rename_remove`` without a kept partner) never appear.
    """
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be >= 0 or None, got {top_k!r}")
    if min_score_delta < 0.0:
        raise ValueError(
            f"min_score_delta must be >= 0, got {min_score_delta!r}"
        )

    filtered = _filter_paired_ops(ops, min_score_delta)

    if top_k is None:
        selected = filtered
    else:
        selected = filtered[:top_k]

    truncated = len(selected) < len(ops)
    return selected, truncated


def _compute_error_breakdown(ops) -> dict:
    """Sum ``score_delta`` by ``op_kind`` for an iterable of ops/entries."""
    out: dict[str, float] = {}
    for op in ops:
        key = op.op_kind if isinstance(op, FeedbackEntry) else op.kind
        out[key] = out.get(key, 0.0) + float(op.score_delta)
    return out


def _compute_dominant_kind(
    entries: list,
    threshold: float,
) -> tuple[str | None, float]:
    """Return ``(kind, fraction)`` where ``fraction >= threshold``.

    "kind" here is the human-collapsed bucket from ``_OP_KIND_HUMAN``
    (so ``key_rename_add`` and ``key_rename_remove`` count together).
    Returns ``(None, 0.0)`` if no bucket meets the threshold or the
    deficit shown is zero.
    """
    if not entries:
        return None, 0.0
    by_bucket: dict[str, float] = {}
    total = 0.0
    for e in entries:
        bucket = _OP_KIND_HUMAN.get(e.op_kind, e.op_kind)
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + float(e.score_delta)
        total += float(e.score_delta)
    if total <= 0.0:
        return None, 0.0
    best_bucket = max(by_bucket, key=by_bucket.get)
    fraction = by_bucket[best_bucket] / total
    if fraction >= threshold:
        return best_bucket, fraction
    return None, 0.0


def _compute_top_kinds(entries: list, n: int = 3) -> str:
    """Comma-joined list of the top-``n`` human kinds by share of deficit."""
    by_bucket: dict[str, float] = {}
    for e in entries:
        bucket = _OP_KIND_HUMAN.get(e.op_kind, e.op_kind)
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + float(e.score_delta)
    ranked = sorted(by_bucket.items(), key=lambda kv: -kv[1])
    return ", ".join(name for name, _ in ranked[:n])


# -----------------------------------------------------------------------------
# Per-op rendering
# -----------------------------------------------------------------------------

def _format_value(value: Any, fmt: Callable[[Any], str]) -> str:
    if value is None:
        return "None"
    return fmt(value)


def _render_entry_text(
    op: RepairOp,
    rank: int,
    templates: dict,
    fmt: Callable[[Any], str],
    pair_paths: dict,
    ref_mode: str = "literal",
    ref_endpoints: dict | None = None,
) -> str:
    """Render the per-op feedback line. Returns ``""`` if the op's template
    is empty/whitespace (silenced)."""
    # Opt-in semantic referential feedback: render ref ops by the gold
    # endpoint's discriminative properties instead of opaque ids. Falls through
    # to the literal branch below when disabled, when no descriptor is present,
    # or when the descriptor is unusable (no discriminator).
    if ref_mode == "semantic" and op.kind in ("ref_fix", "ref_fix_no_target") and ref_endpoints:
        desc = ref_endpoints.get(op.path)
        if desc is not None and desc.usable:
            semantic = _render_ref_semantic(op, rank, desc, templates, fmt)
            if semantic is not None:
                return semantic

    template_key = f"feedback.op.{op.kind}"
    if template_key not in templates:
        raise KeyError(
            f"no feedback template registered for op kind {op.kind!r}"
        )
    template = templates[template_key]
    if not template.strip():
        return ""

    kwargs: dict[str, Any] = {
        "rank": rank,
        "score_delta": op.score_delta,
        "score_delta_pct": 100.0 * op.score_delta,
        "confidence": float(op.confidence),
        "confidence_pct": 100.0 * float(op.confidence),
    }
    gold_str = _format_value(op.gold, fmt)
    pred_str = _format_value(op.pred, fmt)

    if op.kind == "primitive_replace":
        kwargs.update(path=op.path, gold=gold_str, pred=pred_str)
    elif op.kind == "primitive_replace_reorder":
        kwargs.update(list_path=op.path, gold=gold_str, pred=pred_str)
    elif op.kind == "key_add":
        kwargs.update(
            path=op.path,
            key=_last_path_token(op.path),
            gold=gold_str,
        )
    elif op.kind == "key_remove":
        kwargs.update(
            path=op.path,
            key=_last_path_token(op.path),
            pred=pred_str,
        )
    elif op.kind == "key_rename_add":
        pred_path = pair_paths.get(op.pair_id, "")
        kwargs.update(
            gold_path=op.path,
            pred_path=pred_path,
            gold_key=_last_path_token(op.path),
            pred_key=_last_path_token(pred_path),
            gold=gold_str,
        )
    elif op.kind == "key_rename_remove":
        kwargs.update(
            pred_path=op.path,
            pred_key=_last_path_token(op.path),
            pred=pred_str,
        )
    elif op.kind == "list_item_add":
        kwargs.update(path=op.path, gold=gold_str)
    elif op.kind == "list_item_remove":
        kwargs.update(path=op.path, pred=pred_str)
    elif op.kind == "list_item_missing":
        kwargs.update(list_path=op.path, gold=gold_str)
    elif op.kind == "list_item_excess":
        kwargs.update(list_path=op.path, pred=pred_str)
    elif op.kind == "ref_fix":
        kwargs.update(path=op.path, pred=pred_str, value=_format_value(op.value, fmt))
    elif op.kind == "ref_fix_no_target":
        kwargs.update(path=op.path, pred=pred_str)
    elif op.kind == "subtree_replace":
        kwargs.update(path=op.path)
    elif op.kind == "null_value_replace":
        kwargs.update(path=op.path, gold=gold_str, pred=pred_str)
    elif op.kind == "pairing_ambiguous":
        kwargs.update(path=op.path)
    else:
        raise ValueError(
            f"unknown RepairOp.kind {op.kind!r} — no feedback handler"
        )

    return template.format(**kwargs)


def _render_ref_semantic(
    op: RepairOp,
    rank: int,
    desc,
    templates: dict,
    fmt: Callable[[Any], str],
) -> str | None:
    """Render a ref op via the gold endpoint's discriminative properties.

    Composes the ``target`` / ``used`` / ``relation`` clauses from their
    fragment templates, then formats the op skeleton. Returns ``None`` if a
    required semantic template key is absent, so the caller falls back to the
    literal line.
    """
    skeleton_key = f"feedback.op.{op.kind}.semantic"
    # Fall back to the literal line if the semantic skeleton or any fragment it
    # composes from is absent, so the "never raises" contract holds even if a
    # template path ever bypasses the default-seeded merge.
    required = (
        skeleton_key,
        "feedback.refsem.prop",
        "feedback.refsem.target",
        "feedback.refsem.target_ambiguous",
        "feedback.refsem.relation",
        "feedback.refsem.used",
        "feedback.refsem.used_dangling",
    )
    if any(k not in templates for k in required):
        return None

    def _props(pairs):
        prop_tmpl = templates["feedback.refsem.prop"]
        return ", ".join(
            prop_tmpl.format(key=k, value=fmt(v)) for k, v in pairs
        )

    gold_props = _props(desc.gold_props)
    target_key = (
        "feedback.refsem.target"
        if desc.endpoint_certain
        else "feedback.refsem.target_ambiguous"
    )
    target = templates[target_key].format(scope=desc.scope, gold_props=gold_props)

    if desc.relation_values:
        rel_label = " / ".join(fmt(v) for v in desc.relation_values)
        relation = templates["feedback.refsem.relation"].format(
            relation_label=rel_label
        )
    else:
        relation = ""

    kwargs: dict[str, Any] = {
        "rank": rank,
        "path": op.path,
        "scope": desc.scope,
        "relation": relation,
        "target": target,
        "score_delta": op.score_delta,
        "score_delta_pct": 100.0 * op.score_delta,
        "confidence": float(op.confidence),
        "confidence_pct": 100.0 * float(op.confidence),
    }

    if op.kind == "ref_fix":
        if desc.pred_props is not None:
            used = templates["feedback.refsem.used"].format(
                scope=desc.scope, pred_props=_props(desc.pred_props)
            )
        else:
            used = templates["feedback.refsem.used_dangling"]
        kwargs["used"] = used

    return templates[skeleton_key].format(**kwargs)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def render_feedback(
    repair_result: RepairResult,
    *,
    top_k: int | None = 5,
    min_score_delta: float = 0.0,
    style: str = "gepa",
    include_synthesis_line: bool = True,
    include_metadata: bool = False,
    templates: dict | None = None,
    value_formatter: Callable[[Any], str] | None = None,
    dominant_fraction_threshold: float = _DEFAULT_DOMINANT_FRACTION,
    include_diagnostics: bool = True,
    referential_feedback: str = "literal",
    ref_endpoints: dict | None = None,
) -> FeedbackResult:
    """Render a `RepairResult` as a top-K ranked feedback string.

    See [`docs/feedback.md`](../feedback.md) for the full design and
    examples.

    Args:
        repair_result: Result from `aligner.repair()` or
            `aligner.repair_from_match()`.
        top_k: Maximum number of entries to render. `None` = unlimited;
            `0` renders the `feedback.empty` template only.
        min_score_delta: Drop ops with `score_delta` strictly below this
            value. Default `0.0` keeps every positive-delta op.
        style: `"gepa"` (default), `"compact"`, or `"json"`.
        include_synthesis_line: Whether to append the trailing synthesis
            sentence (single-dominant or mixed).
        include_metadata: Populate `FeedbackResult.error_breakdown` (sum
            by op_kind over the full ops list, not just the displayed
            ones).
        templates: User template overrides. May be `None` (use defaults),
            a partial override dict, or a fully-merged dict.
        value_formatter: `(value) -> str` callable used to format `gold`
            / `pred` values in templates. Default truncates `repr(value)`
            to 80 chars.
        dominant_fraction_threshold: Threshold above which the
            single-dominant synthesis line fires. Default `0.60`.
        include_diagnostics: If `True` (default) and the underlying
            `RepairResult` contains `pairing_ambiguous` ops (only emitted
            when `aligner.feedback(..., include_pairing_ambiguous=True)`
            was used), render them as a trailing "Diagnostic notes"
            section. Set to `False` to suppress the section even when
            such ops are present.
        referential_feedback: `"literal"` (default) renders `ref_fix` /
            `ref_fix_no_target` ops by opaque ids. `"semantic"` renders them by the gold
            endpoint's discriminative properties and the relation label,
            using `ref_endpoints`. Falls back to the literal line per-op
            when no descriptor / discriminator is available.
        ref_endpoints: `{op.path: descriptor}` built by
            `ObjectAligner._build_ref_endpoint_descriptors`. Required for
            `"semantic"` mode to have any effect; ignored otherwise.

    Returns:
        `FeedbackResult` whose `text` is the fully rendered top-K
        feedback string.

    Raises:
        ValueError: If `style` is not a registered style.
    """
    _validate_style(style)
    fmt = value_formatter or _default_value_formatter
    final_templates = _build_templates(style, templates)

    score = float(repair_result.score)
    all_ops_in = list(repair_result.ops)

    # Partition diagnostic-only ops out so they do not consume top-K slots,
    # do not contribute to the synthesis line, and cannot be filtered by
    # min_score_delta (their delta is 0 by design).
    all_ops = [op for op in all_ops_in if op.kind != "pairing_ambiguous"]
    diag_ops = [op for op in all_ops_in if op.kind == "pairing_ambiguous"]
    n_total_ops = len(all_ops)

    selected, truncated = _select_top_k(all_ops, top_k, min_score_delta)
    if top_k == 0 and n_total_ops > 0:
        truncated = True

    # Build pair-paths lookup so key_rename_add can render the pred_path
    # (and vice-versa). Walk selected ops once.
    pair_paths: dict[str, str] = {}
    for op in selected:
        if not op.pair_id:
            continue
        # Each op contributes its own path under the pair_id key:
        # add -> gold_path; remove -> pred_path. We want pred_path indexed
        # by pair_id so key_rename_add can find it.
        if op.kind == "key_rename_remove":
            pair_paths[op.pair_id] = op.path

    # Build FeedbackEntry list (with rank assignment for visible-only items).
    entries: list[FeedbackEntry] = []
    visible_rank = 0
    pair_rank_lookup: dict[str, int] = {}

    for op in selected:
        if style == "json":
            # Bypass rendering; entries still populated.
            text = ""
            if op.pair_id and op.kind == "key_rename_remove":
                this_rank = pair_rank_lookup.get(op.pair_id, 0)
            else:
                visible_rank += 1
                this_rank = visible_rank
                if op.pair_id:
                    pair_rank_lookup[op.pair_id] = this_rank
        else:
            # Pre-check whether this op is silenced via its template.
            template_key = f"feedback.op.{op.kind}"
            if template_key not in final_templates:
                raise KeyError(
                    f"no feedback template registered for op kind "
                    f"{op.kind!r} — add a default for "
                    f"{template_key!r} to DEFAULT_FEEDBACK_TEMPLATES"
                )
            template_str = final_templates[template_key]
            is_silenced = not template_str.strip()
            if is_silenced:
                # Inherit the paired add's rank if available, else 0.
                this_rank = pair_rank_lookup.get(op.pair_id, 0)
                text = ""
            else:
                visible_rank += 1
                this_rank = visible_rank
                if op.pair_id:
                    pair_rank_lookup[op.pair_id] = this_rank
                text = _render_entry_text(
                    op, this_rank, final_templates, fmt, pair_paths,
                    ref_mode=referential_feedback, ref_endpoints=ref_endpoints,
                )

        entries.append(FeedbackEntry(
            rank=this_rank,
            op_kind=op.kind,
            op=op.op,
            path=op.path,
            score_delta=float(op.score_delta),
            score_delta_pct=100.0 * float(op.score_delta),
            gold=op.gold,
            pred=op.pred,
            text=text,
            pair_id=op.pair_id,
            confidence=float(op.confidence),
        ))

    # Build diagnostic entries (pairing_ambiguous). These are NOT counted
    # against top_k and NOT filtered by min_score_delta — they ride on the
    # opt-in include_pairing_ambiguous flag in repair() and are surfaced
    # in a dedicated section by the renderer.
    diag_entries: list[FeedbackEntry] = []
    if include_diagnostics:
        for op in diag_ops:
            if style == "json":
                text = ""
            else:
                text = _render_entry_text(
                    op, rank=0, templates=final_templates,
                    fmt=fmt, pair_paths={},
                )
            diag_entries.append(FeedbackEntry(
                rank=0,
                op_kind=op.kind,
                op=op.op,
                path=op.path,
                score_delta=float(op.score_delta),
                score_delta_pct=100.0 * float(op.score_delta),
                gold=op.gold,
                pred=op.pred,
                text=text,
                pair_id=op.pair_id,
                confidence=float(op.confidence),
            ))

    # Build the final text string.
    if style == "json":
        full_text = ""
    else:
        full_text = _render_full_text(
            score=score,
            entries=entries,
            diag_entries=diag_entries,
            n_total_ops=n_total_ops,
            templates=final_templates,
            include_synthesis_line=include_synthesis_line,
            dominant_fraction_threshold=dominant_fraction_threshold,
        )

    error_breakdown = (
        _compute_error_breakdown(all_ops) if include_metadata else {}
    )

    return FeedbackResult(
        score=score,
        text=full_text,
        entries=tuple(list(entries) + diag_entries),
        style=style,
        truncated=truncated,
        n_total_ops=n_total_ops,
        error_breakdown=error_breakdown,
    )


def _render_full_text(
    *,
    score: float,
    entries: list,
    diag_entries: list,
    n_total_ops: int,
    templates: dict,
    include_synthesis_line: bool,
    dominant_fraction_threshold: float,
) -> str:
    """Render intro + entry lines + optional synthesis + diagnostics."""
    visible = [e for e in entries if e.text]
    visible_diag = [e for e in diag_entries if e.text]
    n_shown = len(visible)

    if n_shown == 0:
        # Either perfect match, or everything filtered/silenced. We still
        # emit a diagnostics section if pairing_ambiguous ops fired —
        # they carry real information even when no fixes are visible.
        if score >= 1.0:
            out = templates["feedback.intro.perfect"].format(
                score=score,
                score_pct=100.0 * score,
            )
        else:
            out = templates["feedback.empty"].format(
                score=score,
                score_pct=100.0 * score,
            )
        if visible_diag:
            out += templates["feedback.diagnostics.intro"]
            out += "\n" + "\n".join(e.text for e in visible_diag)
        return out

    deficit = 1.0 - score
    intro = templates["feedback.intro.imperfect"].format(
        score=score,
        score_pct=100.0 * score,
        deficit=deficit,
        deficit_pct=100.0 * deficit,
        n_shown=n_shown,
        n_total=n_total_ops,
    )

    body = "\n".join(e.text for e in visible)

    out = intro + body
    if include_synthesis_line:
        dom_kind, fraction = _compute_dominant_kind(
            entries, dominant_fraction_threshold,
        )
        if dom_kind is not None:
            out += templates["feedback.synthesis.single_dominant"].format(
                dominant_kind=dom_kind,
                dominant_kind_human=dom_kind,
                dominant_fraction=fraction,
                dominant_fraction_pct=100.0 * fraction,
            )
        else:
            out += templates["feedback.synthesis.mixed"].format(
                top_kinds=_compute_top_kinds(entries),
            )
    if visible_diag:
        out += templates["feedback.diagnostics.intro"]
        out += "\n" + "\n".join(e.text for e in visible_diag)
    return out

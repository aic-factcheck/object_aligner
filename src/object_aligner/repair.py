"""Scored JSON-Patch repair suggestions.

Walks a match tree and emits a ranked list of ``RepairOp`` objects — each one
a RFC 6902-flavored ``add`` / ``remove`` / ``replace`` carrying an estimated
``score_delta`` in :math:`[0, 1-S]` that says how much of the score deficit
the op would close.

v1 is the **approximate** flavor: ``score_delta`` is read from the same per-
aggregator alpha schedules that drive ``tree_walk_attribution`` in
``attribution.py``.

The module also vendors a small ``_apply_op`` utility that supports our op
subset on Python ``dict`` / ``list`` / primitive structures (not JSON
strings). ``RepairResult.apply_to(target)`` calls it to produce a patched
deep copy of ``target``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from object_aligner.attribution import (
    _dict_alphas,
    _encode_pointer_token,
    _is_ignored_child,
    _join_path,
    _list_alphas,
    _resolve_dict_value_schema,
    _resolve_list_child_schema,
)
from object_aligner.object_aligner import MatchDict, MatchItem, MatchList

_VALID_GRANULARITIES = ("leaf", "subtree", "all")
_VALID_RANK_BY = ("score_delta", "expected_gain", "confidence")

_OP_ADD = "add"
_OP_REMOVE = "remove"
_OP_REPLACE = "replace"
# Pseudo-RFC6902 op for diagnostic-only entries (e.g. pairing_ambiguous).
_OP_DESCRIBE = "describe"


@dataclass(frozen=True)
class RepairOp:
    """One scored repair operation.

    Conforms to RFC 6902 JSON Patch vocabulary at the `op` level, plus a
    finer-grained `kind` discriminator the library uses to dispatch on op
    semantics. See [`docs/repair.md`](../repair.md) for the full
    `kind` table.

    Attributes:
        op: One of `"add"` / `"remove"` / `"replace"`.
        path: RFC 6901 JSON Pointer locating the patch site.
        score_delta: Positive — how much of the deficit `1 - S` applying
            this op would close (approximate, v1).
        kind: Finer discriminator (`primitive_replace`, `key_add`,
            `list_item_missing`, `ref_fix`, `ref_fix_no_target`,
            `null_value_replace`, `pairing_ambiguous`, etc.).
            `ref_fix_no_target` is emitted when the gold referent has no
            counterpart in the candidate under the derived bijection; its
            `value` carries the gold-side id as a best-effort apply-time
            replacement (works in concert with a sibling
            `list_item_missing` op), but feedback / describe templates do
            not surface that value so no gold-space id leaks into
            user-visible text.
        value: For `add` / `replace` ops, the value to write.
        gold: Gold value at the patch site (informational, useful for
            rendering feedback).
        pred: Predicted value at the patch site (informational).
        pair_id: Non-empty for ops that must be applied atomically with
            another op (currently `key_rename_remove` + `key_rename_add`
            pairs).
        confidence: Stability score in `[0, 1]` of the alignment that
            produced this op, inherited from the originating match node.
            `1.0` for ops not derived from a Hungarian pairing, and
            `1.0` everywhere when `compute_confidence=False`. Used by
            `rank_by="expected_gain"` / `"confidence"`.
    """

    op: str
    path: str
    score_delta: float
    kind: str
    value: Any = None
    gold: Any = None
    pred: Any = None
    pair_id: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class RepairResult:
    """Result of a single `repair()` call.

    Iterable: `for op in result: ...` yields `RepairOp`s in rank order.
    Indexable: `result[0]` is the highest-`score_delta` op.

    Attributes:
        score: Overall similarity in `[0, 1]` (i.e. the value `metric()`
            would return for the same inputs).
        ops: Ranked tuple of `RepairOp` rows.
        granularity: `"leaf"`, `"subtree"`, or `"all"`.
        total_delta: Sum of `score_delta` across `ops`.
        residual: `total_delta - (1 - score)`. Non-zero indicates the
            ranked ops do not fully close the deficit under the
            approximate flavor.
        notes: Free-form strings flagging when the patch's semantics
            interact with re-pairing (e.g., an `order: "align"` list
            present in the schema).
    """

    score: float
    ops: tuple = field(default_factory=tuple)
    granularity: str = "leaf"
    total_delta: float = 0.0
    residual: float = 0.0
    notes: tuple = field(default_factory=tuple)

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, idx):
        return self.ops[idx]

    def apply_to(self, target: Any) -> Any:
        """Apply every op (in result order) to a deep copy of `target`.

        Args:
            target: The object to patch (typically the same `pred` that
                produced this result).

        Returns:
            A deep copy of `target` with every op applied in result order.
        """
        patched = copy.deepcopy(target)
        for op in self.ops:
            patched = _apply_op(patched, op)
        return patched


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def generate_repairs(
    match_tree,
    schema,
    gold,
    pred,
    mappings: dict | None = None,
    *,
    granularity: str = "leaf",
    min_contribution: float = 0.0,
    rank_by: str = "score_delta",
    include_pairing_ambiguous: bool = False,
    ambiguity_threshold: float = 0.30,
) -> RepairResult:
    """Generate a ranked list of scored repair ops for a match tree.

    Approximate flavor (v1): score deltas come from the same tree-walk
    math as `tree_walk_attribution`.

    Args:
        match_tree: Match tree from `align()`.
        schema: The schema that produced `match_tree`.
        gold: The gold object — read to populate `RepairOp.gold` and
            source values for `add` ops.
        pred: The candidate object (the predicted output) — read to populate
            `RepairOp.pred` and source values for `remove` ops.
        mappings: Per-scope `{gold_id: pred_id}` dict captured from the
            align-time `_AlignContext.current_mappings`. Required for
            emitting `ref_fix` ops; if `None`, no `ref_fix` ops are
            emitted and the walker falls back to using the raw gold ref
            value as the suggested replacement (less useful — pred uses
            arbitrary ids by convention).
        granularity: `"leaf"` (default), `"subtree"`, or `"all"`.
        min_contribution: Drop ops whose `score_delta` falls below this
            threshold. Atomic pairs are kept iff the carrying op passes.
        rank_by: Sort key for the returned ops list. One of
            `"score_delta"` (default, descending by raw deficit closed),
            `"expected_gain"` (descending by `score_delta × confidence`),
            or `"confidence"` (descending by stability of the originating
            pairing). All three modes use the same deterministic
            tiebreaker `(path, op, kind)`. Default preserves byte-for-
            byte behavior of pre-confidence releases.
        include_pairing_ambiguous: If `True`, walk the match tree for
            Hungarian-paired containers whose `confidence` falls below
            `ambiguity_threshold` and append a `pairing_ambiguous` op
            (with `score_delta = 0`) at each such path. Diagnostic
            only — `RepairResult.apply_to` ignores them. Off by default.
        ambiguity_threshold: Confidence threshold for the
            `pairing_ambiguous` walker. Default `0.30`.

    Returns:
        `RepairResult` whose `ops` are ranked under `rank_by`.

    Raises:
        ValueError: If `granularity` or `rank_by` is not one of the
            supported values.
    """
    if granularity not in _VALID_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {_VALID_GRANULARITIES!r}, got {granularity!r}"
        )
    if rank_by not in _VALID_RANK_BY:
        raise ValueError(
            f"rank_by must be one of {_VALID_RANK_BY!r}, got {rank_by!r}"
        )

    state = _WalkState(mappings=mappings or {}, granularity=granularity)
    _walk(
        node=match_tree,
        schema=schema,
        path="",
        gold_subtree=gold,
        pred_subtree=pred,
        parent_list_kind="",
        c_parent=1.0,
        state=state,
    )

    ops = state.ops
    score = float(match_tree.score)
    deficit = 1.0 - score
    notes: list[str] = []

    # Filter by min_contribution. Key-rename pairs are atomic: keep both iff
    # the add (which carries the delta) passes.
    if min_contribution > 0.0:
        ops = _filter_paired_ops(ops, min_contribution)

    # Opt-in diagnostic ops for low-confidence Hungarian containers. Emit
    # AFTER the score-delta filter so they cannot be dropped by it (their
    # score_delta is 0 by design). They are appended here rather than during
    # the main walk so the ordering of fix ops stays stable when this flag
    # toggles.
    if include_pairing_ambiguous:
        ops = list(ops) + list(
            _emit_pairing_ambiguous(match_tree, ambiguity_threshold)
        )

    # Sort according to rank_by. Tiebreaker is the deterministic
    # (path, op, kind) tuple in every mode so within-tie ordering is stable.
    if rank_by == "score_delta":
        ops.sort(key=lambda o: (-o.score_delta, o.path, o.op, o.kind))
    elif rank_by == "expected_gain":
        ops.sort(
            key=lambda o: (
                -o.score_delta * o.confidence,
                -o.score_delta,
                o.path,
                o.op,
                o.kind,
            )
        )
    else:  # rank_by == "confidence"
        ops.sort(
            key=lambda o: (
                -o.confidence,
                -o.score_delta,
                o.path,
                o.op,
                o.kind,
            )
        )

    total = float(sum(op.score_delta for op in ops))
    residual = total - deficit

    if state.has_reorder_list:
        notes.append(
            "schema contains at least one order='align' list; "
            "list_item_missing / list_item_excess / primitive_replace_reorder "
            "ops use semantic paths (path points at the list, not a specific index)."
        )

    return RepairResult(
        score=score,
        ops=tuple(ops),
        granularity=granularity,
        total_delta=total,
        residual=residual,
        notes=tuple(notes),
    )


# -----------------------------------------------------------------------------
# Internal walker state
# -----------------------------------------------------------------------------

@dataclass
class _WalkState:
    mappings: dict
    granularity: str
    ops: list = field(default_factory=list)
    pair_counter: int = 0
    has_reorder_list: bool = False

    def next_pair_id(self) -> str:
        self.pair_counter += 1
        return f"repair_pair_{self.pair_counter}"


# -----------------------------------------------------------------------------
# Recursive walker
# -----------------------------------------------------------------------------

def _walk(
    node,
    schema,
    path: str,
    gold_subtree: Any,
    pred_subtree: Any,
    parent_list_kind: str,
    c_parent: float,
    state: _WalkState,
):
    if isinstance(node, MatchItem):
        _walk_item(node, schema, path, parent_list_kind, c_parent, state)
        return
    if isinstance(node, MatchList):
        _walk_list(node, schema, path, gold_subtree, pred_subtree, c_parent, state)
        return
    if isinstance(node, MatchDict):
        _walk_dict(node, schema, path, gold_subtree, pred_subtree, c_parent, state)
        return
    raise TypeError(f"unsupported match node type: {type(node).__name__}")


def _walk_item(
    node: MatchItem,
    schema: Any,
    path: str,
    parent_list_kind: str,
    c_parent: float,
    state: _WalkState,
):
    """Emit ops for a primitive leaf based on (kind, score, gold/pred presence)."""
    kind = node.kind

    # id leaves are always perfect — no op.
    if kind == "id":
        return

    # null-aware leaves: emitted by `_align_null` when one or both sides are
    # None. Both-None is a perfect match (score 1.0) and produces no op;
    # asymmetric emits a `null_value_replace` carrying `gold` as the target
    # value (which may itself be None).
    if kind == "null":
        if node.score >= 1.0:
            return
        state.ops.append(
            RepairOp(
                op=_OP_REPLACE,
                path=path,
                value=node.gold,
                score_delta=c_parent * (1.0 - node.score),
                kind="null_value_replace",
                gold=node.gold,
                pred=node.pred,
                confidence=float(node.confidence),
            )
        )
        return

    # ref leaves: emit ref_fix when the bijection resolves to a concrete
    # pred-space id, ref_fix_no_target when it does not. The mapped pred id
    # rides on `node.aux["mapped_pred"]` (populated by `_align_helper`); we
    # fall back to a re-lookup against `state.mappings` if a caller built the
    # match tree by hand without `aux`.
    if kind == "ref":
        if node.score == 1.0:
            return
        mapped_pred_id = None
        if node.aux is not None and "mapped_pred" in node.aux:
            mapped_pred_id = node.aux["mapped_pred"]
        else:
            ref_scope = schema.get("ref") if isinstance(schema, dict) else None
            if ref_scope and ref_scope in state.mappings:
                mapped_pred_id = state.mappings[ref_scope].get(node.gold)
        if mapped_pred_id is not None:
            state.ops.append(
                RepairOp(
                    op=_OP_REPLACE,
                    path=path,
                    value=mapped_pred_id,
                    score_delta=c_parent * (1.0 - node.score),
                    kind="ref_fix",
                    gold=node.gold,
                    pred=node.pred,
                    confidence=float(node.confidence),
                )
            )
        else:
            # No pred-space counterpart for the gold referent under the
            # derived bijection. Carry `value=node.gold` so apply_to chained
            # with the sibling list_item_missing op can still reach 1.0; the
            # feedback / describe templates do not surface `value` for this
            # op kind, so no gold-space id leaks into user-visible text.
            state.ops.append(
                RepairOp(
                    op=_OP_REPLACE,
                    path=path,
                    value=node.gold,
                    score_delta=c_parent * (1.0 - node.score),
                    kind="ref_fix_no_target",
                    gold=node.gold,
                    pred=node.pred,
                    confidence=float(node.confidence),
                )
            )
        return

    # Standard primitive leaf.
    if node.score >= 1.0:
        return

    # One-side-None cases: these arise as children of MatchList (unmatched
    # gold or excess pred). The parent _walk_list emits the right op kind
    # depending on parent_list_kind; this branch is for the leaf children
    # that the dict aggregator emits for one-side-None key positions, which
    # are handled at the dict level instead. So at MatchItem level, if both
    # sides are None we silently skip (the dual-None prefix sentinel case).
    if node.gold is None and node.pred is None:
        return

    # Direct list-child unmatched cases are handled by _walk_list before
    # recursing; we only get here for "both present, imperfect score".
    if node.gold is not None and node.pred is not None:
        if parent_list_kind == "reorder":
            op_kind = "primitive_replace_reorder"
        else:
            op_kind = "primitive_replace"
        state.ops.append(
            RepairOp(
                op=_OP_REPLACE,
                path=path,
                value=node.gold,
                score_delta=c_parent * (1.0 - node.score),
                kind=op_kind,
                gold=node.gold,
                pred=node.pred,
                confidence=float(node.confidence),
            )
        )


def _walk_list(
    node: MatchList,
    schema: Any,
    path: str,
    gold_subtree: Any,
    pred_subtree: Any,
    c_parent: float,
    state: _WalkState,
):
    list_kind = node.kind or "fixed"
    if list_kind == "reorder":
        state.has_reorder_list = True

    # Emit a subtree_replace if requested (and if subtree imperfect).
    if state.granularity in ("subtree", "all") and node.score < 1.0:
        state.ops.append(
            RepairOp(
                op=_OP_REPLACE,
                path=path,
                value=gold_subtree,
                score_delta=c_parent * (1.0 - node.score),
                kind="subtree_replace",
                gold=gold_subtree,
                pred=pred_subtree,
                confidence=float(node.confidence),
            )
        )

    # In subtree-only mode we skip leaf-level emission entirely.
    if state.granularity == "subtree":
        return

    # Empty list (e.g. MatchList(score=1.0, children=[])): nothing to recurse.
    if not node.children:
        return

    alphas = _list_alphas(node, schema)

    for i, (child, alpha) in enumerate(zip(node.children, alphas)):
        if alpha == 0.0:
            continue
        c_child = c_parent * alpha
        child_schema = _resolve_list_child_schema(node, schema, i)
        child_path = _join_path(path, i)

        # MatchItem children: handle directly so we can choose the right
        # path (list-level for reorder, child-level otherwise).
        if isinstance(child, MatchItem):
            if _emit_list_unmatched_op(child, list_kind, path, child_path, c_child, node, state):
                continue
            # Both sides present, imperfect score: emit primitive replace.
            if child.score < 1.0:
                if list_kind == "reorder":
                    op_kind = "primitive_replace_reorder"
                    op_path = path  # list-level semantic path
                else:
                    op_kind = "primitive_replace"
                    op_path = child_path
                state.ops.append(
                    RepairOp(
                        op=_OP_REPLACE,
                        path=op_path,
                        value=child.gold,
                        score_delta=c_child * (1.0 - child.score),
                        kind=op_kind,
                        gold=child.gold,
                        pred=child.pred,
                        confidence=float(child.confidence),
                    )
                )
            continue

        # Nested MatchDict / MatchList child: recurse.
        # For reorder lists with nested children, child paths use post-
        # Hungarian indices and are NOT stable into the user's original
        # pred array — documented in docs/repair.md.
        _walk(
            node=child,
            schema=child_schema,
            path=child_path,
            gold_subtree=_safe_index(gold_subtree, i),
            pred_subtree=_safe_index(pred_subtree, i),
            parent_list_kind=list_kind,
            c_parent=c_child,
            state=state,
        )


def _emit_list_unmatched_op(
    child: MatchItem,
    list_kind: str,
    list_path: str,
    child_path: str,
    c_child: float,
    parent_list: MatchList,
    state: _WalkState,
) -> bool:
    """Emit add/remove for a list child where one side is None. Returns True if handled."""
    if child.gold is None and child.pred is None:
        # Dual-None prefix sentinel: nothing useful to emit.
        return True

    # Unmatched list items are excess/missing, not paired — their stability
    # is the parent list's container-level confidence (the Hungarian decided
    # to leave this one without a partner).
    parent_conf = float(parent_list.confidence)

    if child.pred is None and child.gold is not None:
        if list_kind == "reorder":
            op_kind = "list_item_missing"
            op_path = list_path
        else:
            op_kind = "list_item_add"
            op_path = child_path
        state.ops.append(
            RepairOp(
                op=_OP_ADD,
                path=op_path,
                value=child.gold,
                score_delta=c_child * (1.0 - child.score),
                kind=op_kind,
                gold=child.gold,
                pred=None,
                confidence=parent_conf,
            )
        )
        return True

    if child.gold is None and child.pred is not None:
        if list_kind == "reorder":
            op_kind = "list_item_excess"
            op_path = list_path
        else:
            op_kind = "list_item_remove"
            op_path = child_path
        state.ops.append(
            RepairOp(
                op=_OP_REMOVE,
                path=op_path,
                value=None,
                score_delta=c_child * (1.0 - child.score),
                kind=op_kind,
                gold=None,
                pred=child.pred,
                confidence=parent_conf,
            )
        )
        return True

    return False  # Both sides present — regular primitive_replace path.


def _walk_dict(
    node: MatchDict,
    schema: Any,
    path: str,
    gold_subtree: Any,
    pred_subtree: Any,
    c_parent: float,
    state: _WalkState,
):
    # Subtree-replace at the dict level.
    if state.granularity in ("subtree", "all") and node.score < 1.0:
        state.ops.append(
            RepairOp(
                op=_OP_REPLACE,
                path=path,
                value=gold_subtree,
                score_delta=c_parent * (1.0 - node.score),
                kind="subtree_replace",
                gold=gold_subtree,
                pred=pred_subtree,
                confidence=float(node.confidence),
            )
        )

    if state.granularity == "subtree":
        return

    if not node.children:
        return

    key_alphas, value_alphas = _dict_alphas(node, schema)

    for (key_match, value_match), a_key, a_value in zip(
        node.children.items(), key_alphas, value_alphas
    ):
        gk = key_match.gold
        pk = key_match.pred
        c_key = c_parent * a_key
        c_value = c_parent * a_value

        # Case 1: excess pred key (gold key None).
        if gk is None and pk is not None:
            # The whole pred entry is excess; emit a single remove covering
            # both key and value contributions. Excess/missing keys are
            # unmatched pairs; their stability is the parent dict's
            # container-level confidence.
            value_contribution = c_value * (1.0 - value_match.score) if isinstance(value_match, MatchItem) else 0.0
            state.ops.append(
                RepairOp(
                    op=_OP_REMOVE,
                    path=_join_path(path, pk),
                    value=None,
                    score_delta=c_key * (1.0 - key_match.score) + value_contribution,
                    kind="key_remove",
                    gold=None,
                    pred=pred_subtree.get(pk) if isinstance(pred_subtree, dict) else None,
                    confidence=float(node.confidence),
                )
            )
            continue

        # Case 2: missing gold key (pred key None).
        if pk is None and gk is not None:
            value_contribution = c_value * (1.0 - value_match.score) if isinstance(value_match, MatchItem) else 0.0
            gold_value = gold_subtree.get(gk) if isinstance(gold_subtree, dict) else None
            state.ops.append(
                RepairOp(
                    op=_OP_ADD,
                    path=_join_path(path, gk),
                    value=gold_value,
                    score_delta=c_key * (1.0 - key_match.score) + value_contribution,
                    kind="key_add",
                    gold=gold_value,
                    pred=None,
                    confidence=float(node.confidence),
                )
            )
            continue

        # Case 3: keys match exactly (or both None — impossible per aggregator).
        if gk == pk:
            value_path = _join_path(path, gk)
            value_schema = _resolve_dict_value_schema(schema, gk)
            child_gold = gold_subtree.get(gk) if isinstance(gold_subtree, dict) else None
            child_pred = pred_subtree.get(pk) if isinstance(pred_subtree, dict) else None
            _walk(
                node=value_match,
                schema=value_schema,
                path=value_path,
                gold_subtree=child_gold,
                pred_subtree=child_pred,
                parent_list_kind="",
                c_parent=c_value,
                state=state,
            )
            continue

        # Case 4: fuzzy key match — keys differ but Hungarian paired them.
        pair_id = state.next_pair_id()
        gold_value = gold_subtree.get(gk) if isinstance(gold_subtree, dict) else None
        pred_value = pred_subtree.get(pk) if isinstance(pred_subtree, dict) else None

        # The value-side gain has to be attributed to the add op (the remove
        # is a no-op alone). Walk the value child into a side list to capture
        # its score_delta sum without polluting the main ops list.
        side_ops: list[RepairOp] = []
        side_state = _WalkState(
            mappings=state.mappings,
            granularity=state.granularity,
            ops=side_ops,
            pair_counter=state.pair_counter,
            has_reorder_list=state.has_reorder_list,
        )
        value_schema = _resolve_dict_value_schema(schema, gk)
        _walk(
            node=value_match,
            schema=value_schema,
            path=_join_path(path, gk),  # Use gold key path for inner ops.
            gold_subtree=gold_value,
            pred_subtree=pred_value,
            parent_list_kind="",
            c_parent=c_value,
            state=side_state,
        )
        # Adopt the side state's bookkeeping.
        state.pair_counter = side_state.pair_counter
        state.has_reorder_list = state.has_reorder_list or side_state.has_reorder_list

        value_gain = sum(op.score_delta for op in side_ops)
        key_gain = c_key * (1.0 - key_match.score)

        # Gain-weighted blend of key-pair and value-subtree confidences.
        # When total_gain is zero, both halves are perfect; rename_conf
        # defaults to 1.0 (the rename has nothing to fix and is fully
        # committed).
        total_gain = key_gain + value_gain
        if total_gain > 0:
            value_conf_avg = (
                sum(op.score_delta * op.confidence for op in side_ops) / value_gain
                if value_gain > 0
                else 1.0
            )
            rename_conf = (
                key_gain * float(key_match.confidence)
                + value_gain * value_conf_avg
            ) / total_gain
            rename_conf = min(1.0, max(0.0, rename_conf))
        else:
            rename_conf = 1.0

        # Emit the remove (score_delta = 0; paired).
        state.ops.append(
            RepairOp(
                op=_OP_REMOVE,
                path=_join_path(path, pk),
                value=None,
                score_delta=0.0,
                kind="key_rename_remove",
                gold=None,
                pred=pred_value,
                pair_id=pair_id,
                confidence=rename_conf,
            )
        )
        # Emit the add carrying the gold value (fixes both key and content).
        state.ops.append(
            RepairOp(
                op=_OP_ADD,
                path=_join_path(path, gk),
                value=gold_value,
                score_delta=key_gain + value_gain,
                kind="key_rename_add",
                gold=gold_value,
                pred=pred_value,
                pair_id=pair_id,
                confidence=rename_conf,
            )
        )
        # We intentionally do NOT re-emit side_ops: the add carries all the
        # content gain already, and any leaf-level fixes inside the renamed
        # value would target a path that doesn't yet exist (the key is being
        # added by this op). The user applies the rename atomically.


# -----------------------------------------------------------------------------
# Pairing-ambiguous diagnostic walker (opt-in via include_pairing_ambiguous)
# -----------------------------------------------------------------------------

def _emit_pairing_ambiguous(match_tree, threshold: float):
    """Yield ``RepairOp(kind="pairing_ambiguous")`` for every Hungarian-paired
    container in ``match_tree`` whose ``confidence`` falls below ``threshold``.

    Hungarian-paired containers are ``MatchList(kind="reorder")`` and any
    ``MatchDict``. Other ``MatchList`` aggregators (fixed / prefix /
    combined) aggregate child confidences without introducing pairing
    ambiguity at their own level, so they are skipped.
    """
    def walk(node, path: str):
        if isinstance(node, MatchList):
            if node.kind == "reorder" and float(node.confidence) < threshold:
                yield RepairOp(
                    op=_OP_DESCRIBE,
                    path=path or "/",
                    score_delta=0.0,
                    kind="pairing_ambiguous",
                    confidence=float(node.confidence),
                )
            for i, child in enumerate(node.children):
                yield from walk(child, _join_path(path, i))
            return
        if isinstance(node, MatchDict):
            if float(node.confidence) < threshold:
                yield RepairOp(
                    op=_OP_DESCRIBE,
                    path=path or "/",
                    score_delta=0.0,
                    kind="pairing_ambiguous",
                    confidence=float(node.confidence),
                )
            for key_match, value_match in node.children.items():
                if key_match.gold is not None:
                    yield from walk(value_match, _join_path(path, key_match.gold))
            return
        # MatchItem leaf: no container to be ambiguous about.
        return

    yield from walk(match_tree, "")


# -----------------------------------------------------------------------------
# Score-delta filter shared with feedback.py (key-rename pairs are atomic)
# -----------------------------------------------------------------------------

def _filter_paired_ops(ops, threshold: float) -> list:
    """Keep ops with ``score_delta >= threshold``; key-rename pairs are kept
    atomically (iff the ``key_rename_add`` half passes the threshold)."""
    pair_pass: dict[str, bool] = {}
    for op in ops:
        if op.pair_id and op.kind == "key_rename_add":
            pair_pass[op.pair_id] = op.score_delta >= threshold

    kept = []
    for op in ops:
        if op.pair_id:
            if pair_pass.get(op.pair_id, False):
                kept.append(op)
            continue
        if op.score_delta >= threshold:
            kept.append(op)
    return kept


# -----------------------------------------------------------------------------
# Vendored patch applier
# -----------------------------------------------------------------------------

def _apply_op(obj: Any, op: RepairOp) -> Any:
    """Apply ``op`` to ``obj`` and return the (possibly modified) result.

    Mutates the tree in place where possible; for root-level replacements
    returns a new value. Caller is responsible for deep-copying inputs if
    needed; ``RepairResult.apply_to`` does this.
    """
    # Semantic ops on reorder lists need different handling.
    if op.kind == "list_item_missing":
        target = _resolve_pointer(obj, op.path)
        if not isinstance(target, list):
            raise TypeError(f"list_item_missing target is not a list at {op.path!r}")
        target.append(op.value)
        return obj
    if op.kind == "list_item_excess":
        target = _resolve_pointer(obj, op.path)
        if not isinstance(target, list):
            raise TypeError(f"list_item_excess target is not a list at {op.path!r}")
        try:
            target.remove(op.pred)
        except ValueError as e:
            raise ValueError(
                f"list_item_excess: pred value {op.pred!r} not found in list at {op.path!r}"
            ) from e
        return obj
    if op.kind == "primitive_replace_reorder":
        target = _resolve_pointer(obj, op.path)
        if not isinstance(target, list):
            raise TypeError(f"primitive_replace_reorder target is not a list at {op.path!r}")
        try:
            idx = target.index(op.pred)
        except ValueError as e:
            raise ValueError(
                f"primitive_replace_reorder: pred value {op.pred!r} not found in list at {op.path!r}"
            ) from e
        target[idx] = op.value
        return obj

    # Strict-path ops: add / remove / replace at a specific location.
    if op.path == "":
        # Root replacement (rare in repair; subtree_replace at root path).
        if op.op == _OP_REPLACE:
            return copy.deepcopy(op.value)
        raise ValueError(f"cannot {op.op!r} at root path")

    parent_path, _, last_token = op.path.rpartition("/")
    parent = _resolve_pointer(obj, parent_path)
    key = _decode_pointer_token(last_token)

    if isinstance(parent, list):
        index = _list_index(parent, key, op.op)
        if op.op == _OP_ADD:
            parent.insert(index, op.value)
        elif op.op == _OP_REMOVE:
            del parent[index]
        elif op.op == _OP_REPLACE:
            parent[index] = op.value
        else:
            raise ValueError(f"unsupported list op: {op.op!r}")
    elif isinstance(parent, dict):
        # JSON Pointer tokens in dicts may need to map back to the original
        # key (non-string keys are stringified on the way out).
        actual_key = _match_dict_key(parent, key)
        if op.op == _OP_ADD:
            parent[actual_key] = op.value
        elif op.op == _OP_REMOVE:
            if actual_key not in parent:
                raise KeyError(f"key {actual_key!r} not in dict at {parent_path!r}")
            del parent[actual_key]
        elif op.op == _OP_REPLACE:
            if actual_key not in parent:
                raise KeyError(f"key {actual_key!r} not in dict at {parent_path!r}")
            parent[actual_key] = op.value
        else:
            raise ValueError(f"unsupported dict op: {op.op!r}")
    else:
        raise TypeError(
            f"cannot resolve op {op.op!r} at {op.path!r}: parent is {type(parent).__name__}"
        )
    return obj


def _resolve_pointer(obj: Any, path: str) -> Any:
    """Navigate a JSON Pointer path through ``obj`` and return the referenced object."""
    if path == "":
        return obj
    if not path.startswith("/"):
        raise ValueError(f"invalid JSON Pointer: {path!r}")
    tokens = path[1:].split("/")
    cur = obj
    for tok in tokens:
        key = _decode_pointer_token(tok)
        if isinstance(cur, list):
            cur = cur[_list_index(cur, key, op_for_resolve=True)]
        elif isinstance(cur, dict):
            actual_key = _match_dict_key(cur, key)
            if actual_key not in cur:
                raise KeyError(f"key {key!r} not in dict while resolving {path!r}")
            cur = cur[actual_key]
        else:
            raise TypeError(
                f"cannot resolve token {tok!r} through {type(cur).__name__} while resolving {path!r}"
            )
    return cur


def _decode_pointer_token(tok: str) -> str:
    # Inverse of `_encode_pointer_token`: ~1 → /, then ~0 → ~ (order matters).
    return tok.replace("~1", "/").replace("~0", "~")


def _list_index(lst: list, token: str, op=None, op_for_resolve: bool = False) -> int:
    """Resolve a JSON Pointer token to a list index. ``-`` means append-position."""
    if token == "-":
        if op == _OP_ADD:
            return len(lst)
        raise ValueError("'-' index is only valid for add ops")
    try:
        idx = int(token)
    except ValueError as e:
        raise ValueError(f"non-integer list index {token!r}") from e
    if op == _OP_ADD:
        # add allows index == len(lst) (append).
        if idx < 0 or idx > len(lst):
            raise IndexError(f"index {idx} out of range for list of length {len(lst)}")
    else:
        if idx < 0 or idx >= len(lst):
            raise IndexError(f"index {idx} out of range for list of length {len(lst)}")
    return idx


def _match_dict_key(d: dict, encoded_key: str) -> Any:
    """Map an encoded JSON Pointer token back to the original dict key.

    Encoding stringifies non-string keys (``str(key)``), so for non-string
    keys we must reverse-match by stringifying every existing key.
    """
    if encoded_key in d:
        return encoded_key
    for k in d:
        if _encode_pointer_token(k) == encoded_key:
            return k
    return encoded_key  # Will surface as KeyError downstream.


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------

def _safe_index(obj: Any, i: int) -> Any:
    """Index into a list/None, returning None when out of range."""
    if isinstance(obj, list) and 0 <= i < len(obj):
        return obj[i]
    return None

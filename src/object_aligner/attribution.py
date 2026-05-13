"""Tree-walk score attribution.

Walks a match tree returned by ``ObjectAligner.align()`` and produces a ranked
per-path decomposition of the score deficit ``(1 - S)``.

Every internal node's score is a convex combination of its children's
scores under the chosen Hungarian/DP assignment, so

    S = sum_L c_L * s_L,
    1 - S = sum_L c_L * (1 - s_L),

where ``c_L`` (the *effective weight*) is the product of per-aggregator alpha
factors along the path from the root to leaf ``L``. This module produces, for
each leaf (and optionally each internal subtree), the tuple
``(path, score, weight, contribution)`` with ``contribution = c_w * (1 - s_w)``.

Counterfactual attribution is intentionally left out of v1; the API shape
(``AttributionResult.residual``) leaves room for a future ``mode="counterfactual"``
that re-runs ``align()`` on patched inputs.
"""

from dataclasses import dataclass, field
from typing import Any

from object_aligner.object_aligner import MatchDict, MatchItem, MatchList


_VALID_GRANULARITIES = ("leaf", "subtree", "all")


@dataclass(frozen=True)
class AttributionEntry:
    """One row of an attribution result.

    `contribution` is positive: the size of the deficit attributable to this
    path under the fixed-assignment view. Sort descending to surface the
    biggest losers first.

    Attributes:
        path: RFC 6901 JSON Pointer locating the node in the gold tree.
        score: Local similarity in `[0, 1]` at this node.
        weight: Accumulated weight along the root-to-node path.
        contribution: Share of the overall deficit `1 - S` owed to this
            node, equal to `weight * (1 - score)`.
        gold: Gold value at this node (or sentinel for missing positions).
        pred: Predicted value at this node (or sentinel).
        is_leaf: `True` if this row is a leaf (vs. a subtree rollup).
        leaf_kind: For leaves, `"item"` / `"ref"` / `"id"` / `""`.
        node_kind: For non-leaves, the match-node kind passed through to
            attribution consumers.
        part: `"key"` or `"value"` for dict-child rows; otherwise `"value"`.
    """

    path: str
    score: float
    weight: float
    contribution: float
    gold: Any
    pred: Any
    is_leaf: bool
    leaf_kind: str = ""
    node_kind: str = ""
    part: str = "value"


@dataclass(frozen=True)
class AttributionResult:
    """Result of a single `attribute()` call.

    Iterable: `for entry in result: ...` yields `AttributionEntry`s in rank
    order. Indexable: `result[0]` is the highest-contribution row.

    Attributes:
        score: Overall similarity in `[0, 1]` (i.e. the value
            `metric()` would return).
        entries: Ranked tuple of `AttributionEntry` rows.
        granularity: `"leaf"`, `"subtree"`, or `"all"` — controls what
            kind of rows are emitted.
        total_contribution: Sum of entry contributions. For
            `granularity="leaf"` this equals `1 - score` up to `residual`.
        residual: `total_contribution - (1 - score)`. For `"leaf"`,
            non-zero values reflect filtered empty positions.
    """

    score: float
    entries: tuple = field(default_factory=tuple)
    granularity: str = "leaf"
    total_contribution: float = 0.0
    residual: float = 0.0

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


# -----------------------------------------------------------------------------
# JSON Pointer encoding (RFC 6901)
# -----------------------------------------------------------------------------

def _encode_pointer_token(token: Any) -> str:
    """Stringify and escape a single JSON Pointer token per RFC 6901."""
    s = str(token)
    # Per RFC 6901: '~' -> '~0' (must come first), '/' -> '~1'.
    return s.replace("~", "~0").replace("/", "~1")


def _join_path(parent: str, token: Any) -> str:
    return parent + "/" + _encode_pointer_token(token)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def tree_walk_attribution(
    match_tree,
    schema,
    *,
    granularity: str = "leaf",
    include_empty_positions: bool = False,
    sort: bool = True,
) -> AttributionResult:
    """Decompose a match tree's deficit into per-path contributions.

    Implements the tree-walk attribution algorithm (exact under the
    assignment that `align()` produced). See
    [`docs/attribution.md`](../attribution.md) for examples.

    Args:
        match_tree: Root `MatchItem` / `MatchList` / `MatchDict` from
            `align()`.
        schema: The schema that produced `match_tree`. Needed to resolve
            `valueWeight` / `ignoreExcess` / `ignoreMissing` /
            `prefixWeights` and similar keywords.
        granularity: `"leaf"` (default), `"subtree"`, or `"all"`. `"leaf"`
            decomposes only down to primitive nodes; `"subtree"` emits
            roll-up rows for non-leaf nodes; `"all"` emits both.
        include_empty_positions: When `False` (default), dual-None prefix
            sentinels are filtered out of the entry list. Their summed
            contribution shows up in `residual`.
        sort: When `True` (default), entries are returned sorted by
            `contribution` descending.

    Returns:
        `AttributionResult` whose `entries` rank paths by per-path
        contribution.

    Raises:
        ValueError: If `granularity` is not one of `"leaf"` / `"subtree"`
            / `"all"`.
    """
    if granularity not in _VALID_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {_VALID_GRANULARITIES!r}, got {granularity!r}"
        )

    score = float(match_tree.score)
    leaf_entries: list[AttributionEntry] = []
    subtree_entries: list[AttributionEntry] = []
    filtered_contribution = 0.0

    def _emit_leaf(entry: AttributionEntry, is_filtered: bool):
        nonlocal filtered_contribution
        if is_filtered:
            filtered_contribution += entry.contribution
            return
        leaf_entries.append(entry)

    _walk(
        node=match_tree,
        schema=schema,
        path="",
        c_parent=1.0,
        emit_leaf=_emit_leaf,
        emit_subtree=subtree_entries.append,
        include_empty_positions=include_empty_positions,
    )

    if granularity == "leaf":
        entries = leaf_entries
    elif granularity == "subtree":
        entries = subtree_entries
    else:  # "all"
        entries = leaf_entries + subtree_entries

    if sort:
        entries = sorted(entries, key=lambda e: e.contribution, reverse=True)

    total = float(sum(e.contribution for e in entries))
    # Residual is the difference between what we sum and the deficit. It's
    # meaningful only for granularity="leaf"; for "subtree"/"all" the sum is
    # not invariant by construction so we still report the raw diff but the
    # caller should not interpret it as an error metric.
    deficit = 1.0 - score
    residual = total - deficit

    return AttributionResult(
        score=score,
        entries=tuple(entries),
        granularity=granularity,
        total_contribution=total,
        residual=residual,
    )


# -----------------------------------------------------------------------------
# Recursive walker
# -----------------------------------------------------------------------------

def _walk(
    node,
    schema,
    path: str,
    c_parent: float,
    *,
    emit_leaf,
    emit_subtree,
    include_empty_positions: bool,
):
    """Dispatch on node type, compute alphas, recurse."""
    if isinstance(node, MatchItem):
        _walk_item(node, path, c_parent, emit_leaf, include_empty_positions)
        return
    if isinstance(node, MatchList):
        _walk_list(
            node,
            schema,
            path,
            c_parent,
            emit_leaf=emit_leaf,
            emit_subtree=emit_subtree,
            include_empty_positions=include_empty_positions,
        )
        return
    if isinstance(node, MatchDict):
        _walk_dict(
            node,
            schema,
            path,
            c_parent,
            emit_leaf=emit_leaf,
            emit_subtree=emit_subtree,
            include_empty_positions=include_empty_positions,
        )
        return
    raise TypeError(f"unsupported match node type: {type(node).__name__}")


def _walk_item(node: MatchItem, path: str, c_parent: float, emit_leaf, include_empty_positions: bool):
    contribution = c_parent * (1.0 - node.score)
    is_filtered = (
        not include_empty_positions
        and node.gold is None
        and node.pred is None
    )
    emit_leaf(
        AttributionEntry(
            path=path,
            score=node.score,
            weight=c_parent,
            contribution=contribution,
            gold=node.gold,
            pred=node.pred,
            is_leaf=True,
            leaf_kind=node.kind,
            node_kind="",
            part="value",
        ),
        is_filtered=is_filtered,
    )


def _walk_list(
    node: MatchList,
    schema,
    path: str,
    c_parent: float,
    *,
    emit_leaf,
    emit_subtree,
    include_empty_positions: bool,
):
    node_kind = f"list:{node.kind}" if node.kind else "list"

    # Empty list — degenerate leaf.
    if not node.children:
        emit_leaf(
            AttributionEntry(
                path=path,
                score=node.score,
                weight=c_parent,
                contribution=c_parent * (1.0 - node.score),
                gold=None,
                pred=None,
                is_leaf=True,
                leaf_kind="",
                node_kind=node_kind,
                part="value",
            ),
            is_filtered=False,
        )
        return

    alphas = _list_alphas(node, schema)
    # Detect D=0 (all children either ignored or zero-weight). When that
    # happens the subtree is non-decomposable: emit a synthetic leaf at the
    # subtree's own path so the invariant sum still holds.
    if all(a == 0.0 for a in alphas):
        emit_leaf(
            AttributionEntry(
                path=path,
                score=node.score,
                weight=c_parent,
                contribution=c_parent * (1.0 - node.score),
                gold=None,
                pred=None,
                is_leaf=True,
                leaf_kind="",
                node_kind=node_kind,
                part="value",
            ),
            is_filtered=False,
        )
        return

    # Subtree entry for "subtree"/"all" granularity.
    emit_subtree(
        AttributionEntry(
            path=path,
            score=node.score,
            weight=c_parent,
            contribution=c_parent * (1.0 - node.score),
            gold=None,
            pred=None,
            is_leaf=False,
            leaf_kind="",
            node_kind=node_kind,
            part="value",
        )
    )

    # Recurse into each child with its alpha.
    for i, (child, alpha) in enumerate(zip(node.children, alphas)):
        child_path = _join_path(path, i)
        child_schema = _resolve_list_child_schema(node, schema, i)
        c_child = c_parent * alpha
        if alpha == 0.0:
            # Ignored child — skip recursion entirely (it doesn't contribute
            # to the parent score, so it shouldn't show up in attribution).
            continue
        _walk(
            child,
            child_schema,
            child_path,
            c_child,
            emit_leaf=emit_leaf,
            emit_subtree=emit_subtree,
            include_empty_positions=include_empty_positions,
        )


def _walk_dict(
    node: MatchDict,
    schema,
    path: str,
    c_parent: float,
    *,
    emit_leaf,
    emit_subtree,
    include_empty_positions: bool,
):
    if not node.children:
        emit_leaf(
            AttributionEntry(
                path=path,
                score=node.score,
                weight=c_parent,
                contribution=c_parent * (1.0 - node.score),
                gold=None,
                pred=None,
                is_leaf=True,
                leaf_kind="",
                node_kind="dict",
                part="value",
            ),
            is_filtered=False,
        )
        return

    key_alphas, value_alphas = _dict_alphas(node, schema)

    # Subtree entry.
    emit_subtree(
        AttributionEntry(
            path=path,
            score=node.score,
            weight=c_parent,
            contribution=c_parent * (1.0 - node.score),
            gold=None,
            pred=None,
            is_leaf=False,
            leaf_kind="",
            node_kind="dict",
            part="value",
        )
    )

    for (key_match, value_match), a_key, a_value in zip(
        node.children.items(), key_alphas, value_alphas
    ):
        # Use gold key for path when present, else pred key.
        gk = key_match.gold
        pk = key_match.pred
        key_token = gk if gk is not None else pk
        child_path = _join_path(path, key_token)

        # Emit the key MatchItem as its own leaf (part="key").
        c_key = c_parent * a_key
        if a_key != 0.0:
            emit_leaf(
                AttributionEntry(
                    path=child_path,
                    score=key_match.score,
                    weight=c_key,
                    contribution=c_key * (1.0 - key_match.score),
                    gold=gk,
                    pred=pk,
                    is_leaf=True,
                    leaf_kind=key_match.kind,
                    node_kind="",
                    part="key",
                ),
                is_filtered=False,
            )

        # Recurse into the value with its alpha.
        c_value = c_parent * a_value
        value_schema = _resolve_dict_value_schema(schema, gk)
        if a_value == 0.0:
            continue
        _walk(
            value_match,
            value_schema,
            child_path,
            c_value,
            emit_leaf=emit_leaf,
            emit_subtree=emit_subtree,
            include_empty_positions=include_empty_positions,
        )


# -----------------------------------------------------------------------------
# Alpha schedules
# -----------------------------------------------------------------------------

def _is_ignored_child(child, schema) -> bool:
    """Mirrors ``_list_norm`` in the aligner."""
    if not isinstance(child, MatchItem):
        return False
    ignore_excess = schema.get("ignoreExcess", False)
    ignore_missing = schema.get("ignoreMissing", False)
    if child.gold is None and ignore_excess:
        return True
    if child.pred is None and ignore_missing:
        return True
    return False


def _list_alphas(node: MatchList, schema) -> list[float]:
    """Return the alpha (per-child weight) schedule for a MatchList."""
    kind = node.kind
    if kind == "reorder" or kind == "fixed" or kind == "":
        return _list_alphas_uniform(node.children, schema)
    if kind == "prefix":
        return _list_alphas_prefix(node.children, schema)
    if kind == "combined":
        return _list_alphas_combined(node.children, schema)
    raise ValueError(f"unknown MatchList.kind: {kind!r}")


def _list_alphas_uniform(children, schema) -> list[float]:
    """1/D schedule for reorder/fixed lists (D = post-ignore count)."""
    counted = [not _is_ignored_child(c, schema) for c in children]
    D = sum(1 for x in counted if x)
    if D == 0:
        return [0.0] * len(children)
    return [(1.0 / D) if x else 0.0 for x in counted]


def _list_alphas_prefix(children, schema) -> list[float]:
    """Normalized prefixWeights schedule."""
    n = len(children)
    raw = schema.get("prefixWeights", None)
    if raw is None:
        weights = [1.0] * n
    else:
        # Truncate / pad defensively to match aggregator behavior. The
        # aggregator uses ``np.ones(len(aligned_matches))`` as the default, so
        # we replicate by taking the first ``n`` entries.
        weights = [float(w) for w in raw[:n]]
        while len(weights) < n:
            weights.append(1.0)
    total = sum(weights)
    if total <= 0.0:
        return [0.0] * n
    return [w / total for w in weights]


def _list_alphas_combined(children, schema) -> list[float]:
    """π̄·w̃_i for prefix block, ρ̄/D_rest for the rest block."""
    prefix_items = schema.get("prefixItems", [])
    prefix_len = len(prefix_items)
    n = len(children)
    pi = float(schema.get("prefixImportance", 1.0))
    ri = float(schema.get("restImportance", 1.0))
    total_imp = pi + ri
    pi_bar = pi / total_imp
    ri_bar = ri / total_imp

    prefix_children = children[:prefix_len]
    rest_children = children[prefix_len:]
    prefix_alphas = _list_alphas_prefix(prefix_children, schema)

    rest_schema = {
        "ignoreExcess": schema.get("ignoreExcess", False),
        "ignoreMissing": schema.get("ignoreMissing", False),
    }
    rest_alphas = _list_alphas_uniform(rest_children, rest_schema)

    out = [pi_bar * a for a in prefix_alphas] + [ri_bar * a for a in rest_alphas]
    if len(out) != n:
        raise RuntimeError("internal: combined list alpha schedule length mismatch")
    return out


def _dict_alphas(node: MatchDict, schema):
    """Return (key_alphas, value_alphas), one entry per child pair."""
    n_pairs = len(node.children)
    kappa = float(schema.get("keyImportance", 1.0))
    nu = float(schema.get("valueImportance", 1.0))
    total_imp = kappa + nu
    kappa_bar = kappa / total_imp
    nu_bar = nu / total_imp

    # Value weights: per-property valueWeight (default 1.0).
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    raw_value_weights = []
    for key_match in node.children.keys():
        gk = key_match.gold
        if gk is not None and gk in properties:
            raw_value_weights.append(float(properties[gk].get("valueWeight", 1.0)))
        else:
            raw_value_weights.append(1.0)
    total_vw = sum(raw_value_weights)
    if total_vw <= 0.0:
        norm_value_weights = [0.0] * n_pairs
    else:
        norm_value_weights = [w / total_vw for w in raw_value_weights]

    key_alphas = [kappa_bar / n_pairs] * n_pairs
    value_alphas = [nu_bar * w for w in norm_value_weights]
    return key_alphas, value_alphas


# -----------------------------------------------------------------------------
# Schema resolution
# -----------------------------------------------------------------------------

def _resolve_list_child_schema(node: MatchList, schema, i: int):
    """Resolve the schema for the i-th child of a MatchList."""
    kind = node.kind
    if kind == "prefix":
        return schema.get("prefixItems", [{}])[i] if i < len(schema.get("prefixItems", [])) else {}
    if kind == "combined":
        prefix_items = schema.get("prefixItems", [])
        prefix_len = len(prefix_items)
        if i < prefix_len:
            return prefix_items[i]
        return schema.get("items", {})
    # reorder / fixed / "": items schema for everyone.
    return schema.get("items", {}) if isinstance(schema, dict) else {}


def _resolve_dict_value_schema(schema, gold_key):
    """Resolve the schema for a dict value, given its gold key."""
    if not isinstance(schema, dict):
        return {}
    if gold_key is None:
        return {}
    properties = schema.get("properties", {})
    return properties.get(gold_key, {})

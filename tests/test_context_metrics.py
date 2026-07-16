"""Context-aware custom leaf comparators.

Covers the opt-in three-argument metric signature: the ``context_metric``
decorator, the ``ScoreContext`` handed to such a metric, and correct
threading of parent/root/path through every alignment path (dict values,
reorder/fixed/prefix lists, the referential id-derivation cost matrix).
"""
import copy
import functools

import pytest

from object_aligner import ObjectAligner, ScoreContext, context_metric


# --------------------------------------------------------------------------
# Decorator / public surface
# --------------------------------------------------------------------------
def test_decorator_sets_flag_and_returns_same_object():
    def m(g, p, ctx):
        return 1.0

    out = context_metric(m)
    assert out is m
    assert m.wants_context is True


def test_decorator_preserves_other_attributes():
    def m(g, p, ctx):
        return 1.0

    m.cache = object()
    sentinel = m.cache
    context_metric(m)
    assert m.wants_context is True
    assert m.cache is sentinel


def test_public_imports_available():
    from object_aligner import ScoreContext as SC
    from object_aligner import context_metric as cm

    assert SC is ScoreContext
    assert cm is context_metric


def test_score_context_is_frozen():
    sc = ScoreContext(gold_parent={"a": 1})
    with pytest.raises(Exception):
        sc.gold_parent = {"b": 2}


# --------------------------------------------------------------------------
# Backward compatibility: plain (gold, pred) metrics untouched
# --------------------------------------------------------------------------
def test_plain_two_arg_metric_still_called_with_two_args():
    calls = []

    def plain(g, p):
        calls.append((g, p))
        return 1.0 if g == p else 0.0

    aligner = ObjectAligner(
        {"type": "string", "score": "plain"},
        custom_metrics={"string": {"plain": plain}},
    )
    assert aligner.metric("x", "x")["score"] == 1.0
    assert calls == [("x", "x")]


# --------------------------------------------------------------------------
# ScoreContext contents at a dict leaf
# --------------------------------------------------------------------------
def _capture_metric():
    seen = {}

    @context_metric
    def capture(g, p, ctx):
        seen["ctx"] = ctx
        return 1.0

    return capture, seen


def test_dict_leaf_sees_parents_and_roots_and_path():
    capture, seen = _capture_metric()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "string", "score": "exact"},
            "b": {"type": "string", "score": "capture"},
        },
    }
    aligner = ObjectAligner(schema, custom_metrics={"string": {"capture": capture}})
    gold = {"a": "sib_g", "b": "leaf_g"}
    pred = {"a": "sib_p", "b": "leaf_p"}
    aligner.metric(gold, pred)

    ctx = seen["ctx"]
    assert isinstance(ctx, ScoreContext)
    assert ctx.gold_parent is gold
    assert ctx.pred_parent is pred
    assert ctx.gold_root is gold
    assert ctx.pred_root is pred
    assert ctx.gold_parent["a"] == "sib_g"
    assert ctx.pred_parent["a"] == "sib_p"
    assert ctx.path == ("b",)


def test_number_and_integer_union_leaf():
    seen = {}

    @context_metric
    def num_capture(g, p, ctx):
        seen.setdefault("ctxs", []).append(ctx)
        return 1.0

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "n": {"type": "number", "score": "num_capture"},
            "i": {"type": ["integer", "null"], "score": "num_capture"},
        },
    }
    aligner = ObjectAligner(schema, custom_metrics={"number": {"num_capture": num_capture}})
    aligner.metric({"n": 1.5, "i": 3}, {"n": 1.5, "i": 3})
    assert len(seen["ctxs"]) == 2
    for ctx in seen["ctxs"]:
        assert ctx.gold_root == {"n": 1.5, "i": 3}


# --------------------------------------------------------------------------
# Reorder list cross-pairing: parent must be the specific paired dict
# --------------------------------------------------------------------------
def test_reorder_cross_pairing_parent_is_paired_dict():
    """A sibling-dependent metric on a reorder list must see the sibling of
    the *paired* pred item, not gold's — proving parents travel cross-paired
    through the n*m cost matrix."""

    @context_metric
    def sibling_match(g, p, ctx):
        return 1.0 if ctx.gold_parent["tag"] == ctx.pred_parent["tag"] else 0.0

    schema = {
        "type": "array",
        "order": "align",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tag": {"type": "string", "score": "exact"},
                "v": {"type": "string", "score": "sibling_match"},
            },
        },
    }
    aligner = ObjectAligner(schema, custom_metrics={"string": {"sibling_match": sibling_match}})
    # pred is shuffled; Hungarian pairs by 'tag' (exact). The metric returns
    # 1.0 only when the paired dicts share a tag — so a perfect score proves
    # each cell saw its own cross-paired parent.
    gold = [{"tag": "a", "v": "x"}, {"tag": "b", "v": "x"}, {"tag": "c", "v": "x"}]
    pred = [{"tag": "c", "v": "y"}, {"tag": "a", "v": "y"}, {"tag": "b", "v": "y"}]
    assert aligner.metric(gold, pred)["score"] == 1.0


# --------------------------------------------------------------------------
# Fixed-order list: gold-side path, parent identity
# --------------------------------------------------------------------------
def test_fixed_list_parent_and_path():
    seen = {}

    @context_metric
    def cap(g, p, ctx):
        seen.setdefault("by_val", {})[g] = ctx
        return 1.0

    schema = {
        "type": "array",
        "order": "fixed",
        "items": {"type": "string", "score": "cap"},
    }
    aligner = ObjectAligner(schema, custom_metrics={"string": {"cap": cap}})
    gold = ["p", "q", "r"]
    pred = ["p", "q", "r"]
    aligner.metric(gold, pred)
    # parent is the whole list; path is the gold index.
    assert seen["by_val"]["p"].gold_parent is gold
    assert seen["by_val"]["p"].pred_parent is pred
    assert seen["by_val"]["p"].path == (0,)
    assert seen["by_val"]["r"].path == (2,)


# --------------------------------------------------------------------------
# prefixItems + combined path: offset correctness, original-list parent
# --------------------------------------------------------------------------
def test_combined_prefix_and_items_path_offset_and_parent_identity():
    seen = {}

    @context_metric
    def cap(g, p, ctx):
        seen.setdefault("by_val", {})[g] = ctx
        return 1.0

    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "string", "score": "cap"},
            {"type": "string", "score": "cap"},
        ],
        "items": {"type": "string", "score": "cap"},
        "order": "fixed",
        "prefixImportance": 1.0,
        "restImportance": 1.0,
    }
    aligner = ObjectAligner(schema, custom_metrics={"string": {"cap": cap}})
    gold = ["P0", "P1", "R0", "R1"]
    pred = ["P0", "P1", "R0", "R1"]
    aligner.metric(gold, pred)
    # prefix positions keep their index; rest positions are offset by prefix_len.
    assert seen["by_val"]["P0"].path == (0,)
    assert seen["by_val"]["P1"].path == (1,)
    assert seen["by_val"]["R0"].path == (2,)
    assert seen["by_val"]["R1"].path == (3,)
    # Parent is the ORIGINAL full list, not a slice.
    for v in ("P0", "P1", "R0", "R1"):
        assert seen["by_val"][v].gold_parent is gold
        assert seen["by_val"][v].pred_parent is pred


# --------------------------------------------------------------------------
# Top-level bare-primitive leaf: no parent, empty path
# --------------------------------------------------------------------------
def test_top_level_primitive_has_no_parent():
    seen = {}

    @context_metric
    def cap(g, p, ctx):
        seen["ctx"] = ctx
        return 1.0

    aligner = ObjectAligner(
        {"type": "string", "score": "cap"},
        custom_metrics={"string": {"cap": cap}},
    )
    aligner.metric("hello", "hello")
    ctx = seen["ctx"]
    assert ctx.gold_parent is None
    assert ctx.pred_parent is None
    assert ctx.path == ()
    assert ctx.gold_root == "hello"
    assert ctx.pred_root == "hello"


# --------------------------------------------------------------------------
# List of scalars: parent is the list
# --------------------------------------------------------------------------
def test_list_of_scalars_parent_is_list():
    seen = {}

    @context_metric
    def cap(g, p, ctx):
        seen["ctx"] = ctx
        return 1.0

    aligner = ObjectAligner(
        {"type": "array", "order": "fixed", "items": {"type": "string", "score": "cap"}},
        custom_metrics={"string": {"cap": cap}},
    )
    gold = ["only"]
    pred = ["only"]
    aligner.metric(gold, pred)
    assert seen["ctx"].gold_parent is gold
    assert seen["ctx"].pred_parent is pred


# --------------------------------------------------------------------------
# Referential: context metric on a scope definer fires with real parents and
# leaves the id bijection unchanged (guards the swallowed-exception path).
# --------------------------------------------------------------------------
def _referential_schema(context_score):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "nodes": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer", "idScope": "n"},
                        "label": {"type": "string", "score": context_score},
                        "parent": {"type": "integer", "ref": "n"},
                    },
                },
            }
        },
    }


def test_referential_definer_metric_gets_non_none_parents_and_bijection_stable():
    fired = {"count": 0, "none_parent": 0}

    @context_metric
    def label_ctx(g, p, ctx):
        fired["count"] += 1
        if ctx.gold_parent is None or ctx.pred_parent is None:
            fired["none_parent"] += 1
        return 1.0 if g == p else 0.0

    gold = {"nodes": [
        {"id": 1, "label": "root", "parent": 1},
        {"id": 2, "label": "child", "parent": 1},
    ]}
    pred = {"nodes": [
        {"id": 7, "label": "child", "parent": 7},
        {"id": 9, "label": "root", "parent": 9},
    ]}

    with_ctx = ObjectAligner(
        _referential_schema("label_ctx"),
        custom_metrics={"string": {"label_ctx": label_ctx}},
    )
    plain = ObjectAligner(_referential_schema("exact"))

    s_ctx = with_ctx.metric(gold, pred)["score"]
    s_plain = plain.metric(gold, pred)["score"]

    # The context metric fired during derivation (n*m cost matrix) with a
    # non-None parent every time, and the score matches the plain-exact
    # baseline — the bijection was not corrupted.
    assert fired["count"] > 0
    assert fired["none_parent"] == 0
    assert s_ctx == s_plain


# --------------------------------------------------------------------------
# Determinism: reading parent/root must not change results across runs.
# --------------------------------------------------------------------------
def test_determinism_across_repeated_alignment():
    @context_metric
    def read_only(g, p, ctx):
        _ = ctx.gold_root, ctx.pred_parent, ctx.path
        return 1.0 if g == p else 0.0

    schema = {
        "type": "array",
        "order": "align",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"v": {"type": "string", "score": "read_only"}},
        },
    }
    aligner = ObjectAligner(schema, custom_metrics={"string": {"read_only": read_only}})
    gold = [{"v": "a"}, {"v": "b"}]
    pred = [{"v": "b"}, {"v": "a"}]
    first = aligner.metric(gold, pred)["score"]
    second = aligner.metric(gold, pred)["score"]
    assert first == second == 1.0


# --------------------------------------------------------------------------
# functools.partial can be decorated directly (it accepts attributes).
# --------------------------------------------------------------------------
def test_partial_can_be_decorated_directly():
    def base(g, p, ctx, *, bonus):
        return bonus if g == p else 0.0

    part = context_metric(functools.partial(base, bonus=1.0))
    assert part.wants_context is True

    aligner = ObjectAligner(
        {"type": "string", "score": "part"},
        custom_metrics={"string": {"part": part}},
    )
    assert aligner.metric("z", "z")["score"] == 1.0
    assert aligner.metric("z", "x")["score"] == 0.0


# --------------------------------------------------------------------------
# End-to-end acceptance: the feature-request fixture scores exactly 1.0,
# where a plain string comparator scores < 1.0.
# --------------------------------------------------------------------------
_TEXT = (
    "Před vlak EC 108 Comenius se ve Studénce na Novojičínsku z neznámých "
    "příčin zřítil most, který byl v opravě. Vlak, který jel z Krakova do "
    "Prahy tudy projížděl rychlostí 135 km/h.\nPodle prvních odhadů mohlo být "
    "10 mrtvých a nejméně 100 zraněných. Strojvedoucí, který si všiml vratkého "
    "mostu, před nárazem stačil použít rychlobrzdu. Na místě zasahují "
    "záchranáři z Nového Jičína, Opavy, Ostravy a Vítkova. Nemocnice v okolí "
    "mají pohotovost."
)


def _find_all(text, sub):
    out, i = [], text.find(sub)
    while i != -1:
        out.append(i)
        i = text.find(sub, i + 1)
    return out


def _resolve(text, span, context):
    ctx_positions = _find_all(text, context)
    if len(ctx_positions) >= 1:
        c = ctx_positions[0]
        off = context.find(span)
        if off != -1:
            start = c + off
            return (start, start + len(span))
    span_positions = _find_all(text, span)
    if len(span_positions) == 1:
        return (span_positions[0], span_positions[0] + len(span))
    return None


@context_metric
def _context_resolves_same(gold_ctx, pred_ctx, ctx):
    g = _resolve(ctx.gold_root["text"], ctx.gold_parent["span"], gold_ctx)
    p = _resolve(ctx.pred_root["text"], ctx.pred_parent["span"], pred_ctx)
    return 1.0 if (p is not None and p == g) else 0.0


def _fixture_schema(context_score):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "score": "exact", "valueWeight": 0.0},
            "locations": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer", "idScope": "mention"},
                        "coref_id": {"type": "integer"},
                        "span": {"type": "string", "score": "exact"},
                        "normalized": {"type": "string", "score": "exact"},
                        "context": {"type": "string", "score": context_score},
                        "coref_parent_id": {"type": "integer", "ref": "mention"},
                    },
                },
            },
        },
    }


_GOLD = {"text": _TEXT, "locations": [
    {"id": 1, "coref_id": 1, "span": "Studénce", "normalized": "Studénka", "context": "ve Studénce", "coref_parent_id": 2},
    {"id": 2, "coref_id": 2, "span": "Novojičínsku", "normalized": "Novojičínsko", "context": "na Novojičínsku"},
    {"id": 3, "coref_id": 3, "span": "Krakova", "normalized": "Krakov", "context": "z Krakova"},
    {"id": 4, "coref_id": 4, "span": "Prahy", "normalized": "Praha", "context": "do Prahy"},
    {"id": 5, "coref_id": 5, "span": "Nového Jičína", "normalized": "Nový Jičín", "context": "z Nového Jičína"},
    {"id": 6, "coref_id": 6, "span": "Opavy", "normalized": "Opava", "context": "Opavy"},
    {"id": 7, "coref_id": 7, "span": "Ostravy", "normalized": "Ostrava", "context": "Ostravy"},
    {"id": 8, "coref_id": 8, "span": "Vítkova", "normalized": "Vítkov", "context": "Vítkova"},
]}

_PRED = {"text": _TEXT, "locations": [
    {"id": 1, "coref_id": 1, "span": "Studénce", "normalized": "Studénka", "context": "ve Studénce na", "coref_parent_id": 2},
    {"id": 2, "coref_id": 2, "span": "Novojičínsku", "normalized": "Novojičínsko", "context": "na Novojičínsku"},
    {"id": 3, "coref_id": 3, "span": "Krakova", "normalized": "Krakov", "context": "z Krakova do"},
    {"id": 4, "coref_id": 4, "span": "Prahy", "normalized": "Praha", "context": "do Prahy tudy"},
    {"id": 5, "coref_id": 5, "span": "Nového Jičína", "normalized": "Nový Jičín", "context": "z Nového Jičína,"},
    {"id": 6, "coref_id": 6, "span": "Opavy", "normalized": "Opava", "context": "Opavy, Ostravy a"},
    {"id": 7, "coref_id": 7, "span": "Ostravy", "normalized": "Ostrava", "context": "Ostravy a Vítkova."},
    {"id": 8, "coref_id": 8, "span": "Vítkova", "normalized": "Vítkov", "context": "Vítkova."},
]}


def test_feature_request_fixture_scores_one():
    aligner = ObjectAligner(
        _fixture_schema("context_resolves_same"),
        custom_metrics={"string": {"context_resolves_same": _context_resolves_same}},
    )
    assert aligner.metric(_GOLD, _PRED)["score"] == 1.0


def test_feature_request_fixture_plain_metric_scores_below_one():
    aligner = ObjectAligner(_fixture_schema("jaro"))
    assert aligner.metric(_GOLD, _PRED)["score"] < 1.0

"""Tests for null-aware scoring with per-field ``nullScore``.

Covers the dispatch added in `_align_helper` (null branch before the type
dispatch) plus the downstream wiring in attribution / repair / feedback /
describe and the construction-time `nullScore` range validation.
"""

import pytest

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchDict, MatchItem


# -----------------------------------------------------------------------------
# Top-level nullable primitive
# -----------------------------------------------------------------------------

def test_both_null_scores_one():
    aligner = ObjectAligner({"type": ["string", "null"], "nullScore": 0.0})
    match = aligner.align(None, None)
    assert isinstance(match, MatchItem)
    assert match.score == 1.0
    assert match.kind == "null"


def test_default_null_score_is_zero():
    aligner = ObjectAligner({"type": ["string", "null"]})
    assert aligner.align(None, "hello").score == 0.0
    assert aligner.align("hello", None).score == 0.0


def test_custom_null_score_is_symmetric():
    aligner = ObjectAligner({"type": ["string", "null"], "nullScore": 0.8})
    assert aligner.align(None, "hello").score == 0.8
    assert aligner.align("hello", None).score == 0.8


def test_null_score_one_is_a_free_pass():
    aligner = ObjectAligner({"type": ["string", "null"], "nullScore": 1.0})
    assert aligner.align(None, "hello").score == 1.0
    assert aligner.align("hello", None).score == 1.0
    assert aligner.align(None, None).score == 1.0


def test_two_non_null_values_use_existing_comparator():
    aligner = ObjectAligner({"type": ["string", "null"], "nullScore": 0.8, "score": "jaro"})
    # No null involved → Jaro on the strings, not nullScore.
    assert aligner.align("hello", "hello").score == 1.0
    assert aligner.align("hello", "world").score < 1.0


def test_null_kind_marker_on_match_item():
    aligner = ObjectAligner({"type": ["string", "null"]})
    match = aligner.align(None, "hello")
    assert match.kind == "null"


def test_debug_tree_surfaces_null_marker():
    aligner = ObjectAligner({"type": ["string", "null"], "nullScore": 0.5})
    r = aligner.metric(None, "hello", debug=True)
    assert r["score"] == 0.5
    assert r["debug"]["marker"] == "null"


# -----------------------------------------------------------------------------
# Nullable values inside an object
# -----------------------------------------------------------------------------

@pytest.fixture
def dict_schema():
    return {
        "type": "object",
        "properties": {
            "diagnosis":   {"type": ["string", "null"], "nullScore": 0.0},
            "middle_name": {"type": ["string", "null"], "nullScore": 0.8},
        },
    }


def test_dict_property_per_field_null_score(dict_schema):
    aligner = ObjectAligner(dict_schema)
    # diagnosis is null-on-pred → 0.0, middle_name is null-on-both → 1.0.
    # Default keyImportance=0 → dict score is the mean of value pairs only:
    # values=mean(0.0, 1.0)=0.5.
    r = aligner.metric(
        {"diagnosis": "flu", "middle_name": None},
        {"diagnosis": None,  "middle_name": None},
    )
    assert r["score"] == pytest.approx(0.5)


def test_dict_property_mixed_value_to_null(dict_schema):
    aligner = ObjectAligner(dict_schema)
    # Only middle_name asymmetric → nullScore=0.8 on one value, 1.0 on the
    # other. values=mean(1.0, 0.8)=0.9 (default keyImportance=0).
    r = aligner.metric(
        {"diagnosis": "flu", "middle_name": "Q"},
        {"diagnosis": "flu", "middle_name": None},
    )
    assert r["score"] == pytest.approx(0.9)


def test_dict_pre_null_change_used_to_raise_typeerror(dict_schema):
    """Regression: under the old code, mixed (None, value) types in a dict
    property raised TypeError. Now they score via `nullScore`."""
    aligner = ObjectAligner(dict_schema)
    r = aligner.metric(
        {"diagnosis": None, "middle_name": "Q"},
        {"diagnosis": "x",  "middle_name": "Q"},
    )
    # diagnosis: nullScore=0.0; middle_name: both Q → 1.0.
    # values=mean(0.0, 1.0)=0.5 (default keyImportance=0).
    assert r["score"] == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# Nullable object and array values
# -----------------------------------------------------------------------------

def test_nullable_object_value():
    schema = {
        "type": "object",
        "properties": {
            "addr": {
                "type": ["object", "null"],
                "nullScore": 0.5,
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    aligner = ObjectAligner(schema)
    # gold has the object, pred has null → asymmetric, nullScore=0.5.
    # Default keyImportance=0 → total=0.5 (value only).
    r = aligner.metric({"addr": {"city": "Prague"}}, {"addr": None})
    assert r["score"] == pytest.approx(0.5)


def test_nullable_array_value():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": ["array", "null"],
                "nullScore": 0.3,
                "items": {"type": "string"},
            },
        },
    }
    aligner = ObjectAligner(schema)
    r = aligner.metric({"tags": ["a", "b"]}, {"tags": None})
    # Default keyImportance=0 → total=0.3 (value only, via nullScore).
    assert r["score"] == pytest.approx(0.3)


# -----------------------------------------------------------------------------
# Nullable list items
# -----------------------------------------------------------------------------

def test_list_item_nullability_reorder():
    schema = {
        "type": "array",
        "order": "align",
        "items": {"type": ["integer", "null"], "nullScore": 0.5},
    }
    aligner = ObjectAligner(schema)
    # [1, None] vs [1, 2]: Hungarian pairs 1↔1 (score 1.0) and
    # None↔2 (score 0.5 via nullScore). Mean = 0.75.
    r = aligner.metric([1, None], [1, 2])
    assert r["score"] == pytest.approx(0.75)


def test_list_item_nullability_fixed():
    schema = {
        "type": "array",
        "order": "fixed",
        "items": {"type": ["integer", "null"], "nullScore": 0.5},
    }
    aligner = ObjectAligner(schema)
    # Position-wise: 1 vs 1 → 1.0, None vs 2 → 0.5. Mean = 0.75.
    r = aligner.metric([1, None], [1, 2])
    assert r["score"] == pytest.approx(0.75)


# -----------------------------------------------------------------------------
# Construction-time validation
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, -1.0])
def test_construction_rejects_out_of_range_null_score(bad):
    with pytest.raises(ValueError, match="must be in"):
        ObjectAligner({"type": ["string", "null"], "nullScore": bad})


@pytest.mark.parametrize("bad", ["x", None, [0.5], {"v": 0.5}, True, False])
def test_construction_rejects_non_real_null_score(bad):
    with pytest.raises(ValueError, match="must be a real number"):
        ObjectAligner({"type": ["string", "null"], "nullScore": bad})


def test_construction_validates_nested_null_score():
    bad = {
        "type": "object",
        "properties": {
            "x": {"type": ["string", "null"], "nullScore": 1.5},
        },
    }
    with pytest.raises(ValueError, match="must be in"):
        ObjectAligner(bad)


# -----------------------------------------------------------------------------
# Validation interaction with non-nullable schemas
# -----------------------------------------------------------------------------

def test_non_nullable_schema_returns_zero_on_null_pred():
    """If the schema does not declare nullability, default validation
    rejects the null and `metric` returns score 0.0 — the existing
    invalid-pred behavior."""
    aligner = ObjectAligner({"type": "string"})
    r = aligner.metric("hello", None)
    assert r["score"] == 0.0


def test_skip_validation_lets_null_through_to_null_branch():
    """With skip_validation=True the null still routes through `_align_null`
    and scores `nullScore` (default 0.0). The point is no crash."""
    schema = {"type": "string", "nullScore": 0.5}  # nullScore on non-nullable
    aligner = ObjectAligner(schema)
    match = aligner.align(None, "x", skip_validation=True)
    assert match.score == 0.5
    assert match.kind == "null"


# -----------------------------------------------------------------------------
# Downstream consumer integration
# -----------------------------------------------------------------------------

def test_attribute_leaf_kind_is_null():
    schema = {"type": ["string", "null"], "nullScore": 0.2}
    aligner = ObjectAligner(schema)
    res = aligner.attribute("hello", None)
    assert len(res.entries) == 1
    e = res.entries[0]
    assert e.leaf_kind == "null"
    assert e.score == 0.2
    assert e.contribution == pytest.approx(0.8)


def test_repair_emits_null_value_replace():
    schema = {
        "type": "object",
        "properties": {
            "diagnosis": {"type": ["string", "null"], "nullScore": 0.0},
        },
    }
    aligner = ObjectAligner(schema)
    res = aligner.repair({"diagnosis": "flu"}, {"diagnosis": None})
    kinds = [op.kind for op in res.ops]
    assert "null_value_replace" in kinds
    null_op = next(op for op in res.ops if op.kind == "null_value_replace")
    assert null_op.path == "/diagnosis"
    assert null_op.gold == "flu"
    assert null_op.pred is None
    assert null_op.value == "flu"


def test_repair_apply_to_round_trips_a_null_fill():
    schema = {
        "type": "object",
        "properties": {
            "diagnosis": {"type": ["string", "null"], "nullScore": 0.0},
        },
    }
    aligner = ObjectAligner(schema)
    gold = {"diagnosis": "flu"}
    pred = {"diagnosis": None}
    patched = aligner.repair(gold, pred).apply_to(pred)
    assert patched == gold


def test_feedback_renders_null_line():
    schema = {
        "type": "object",
        "properties": {
            "diagnosis": {"type": ["string", "null"], "nullScore": 0.0},
        },
    }
    aligner = ObjectAligner(schema, generate_feedback=True)
    r = aligner.metric({"diagnosis": "flu"}, {"diagnosis": None})
    assert "null/value mismatch" in r["feedback"]
    assert "/diagnosis" in r["feedback"]


def test_describe_renders_null_match_and_mismatch():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": ["string", "null"], "nullScore": 0.0},
            "b": {"type": ["string", "null"], "nullScore": 0.0},
        },
    }
    aligner = ObjectAligner(schema, generate_description=True)
    r = aligner.metric(
        {"a": "x", "b": None},
        {"a": None, "b": None},
    )
    text = r["description"]
    assert "Both the predicted and gold values are null here" in text
    assert "null/value mismatch" in text


# -----------------------------------------------------------------------------
# Sanity: kind propagation through MatchDict
# -----------------------------------------------------------------------------

def test_match_tree_has_null_kind_leaf():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": ["string", "null"], "nullScore": 0.0},
        },
    }
    aligner = ObjectAligner(schema)
    match = aligner.align({"x": "hi"}, {"x": None})
    assert isinstance(match, MatchDict)
    leaf = next(iter(match.children.values()))
    assert isinstance(leaf, MatchItem)
    assert leaf.kind == "null"

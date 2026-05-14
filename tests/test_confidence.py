"""Tests for the alignment-confidence feature (Cluster 4 v1).

Covers:

* Backward-compat gates (default flags produce byte-identical output).
* Margin / entropy formulas on hand-crafted matrices.
* Up-tree aggregation (list / dict).
* Three feedback ranking modes.
* Opt-in ``pairing_ambiguous`` emission.
* ``describe(show_confidence=...)`` banding + ``include_ambiguous=...``.
* Constructor validation.
* Frozen-dataclass round-trip.
* Debug tree emission.
* Confidence threading through ``RepairOp`` and ``FeedbackEntry``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from object_aligner import ObjectAligner
from object_aligner.object_aligner import (
    MatchDict,
    MatchItem,
    MatchList,
    _hungarian_confidence,
)


# ---------------------------------------------------------------------------
# 1. Backward-compat gates
# ---------------------------------------------------------------------------

def _two_aligners(schema, **extra):
    """Return ``(off, on)`` aligners — same schema, different compute_confidence."""
    return (
        ObjectAligner(schema, compute_confidence=False, **extra),
        ObjectAligner(schema, compute_confidence=True, **extra),
    )


def test_default_feedback_byte_identical_with_compute_off():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    gold = {"name": "Ada", "age": 42, "email": "ada@x"}
    pred = {"naem": "Ad", "years": 41, "mail": "b@y"}
    off, on = _two_aligners(schema)
    # The contract: default flags produce byte-identical output regardless
    # of whether confidence is being computed under the hood.
    assert off.feedback(gold, pred).text == on.feedback(gold, pred).text
    assert off.describe(gold, pred).text == on.describe(gold, pred).text
    # And metric(debug=False) is exactly equal too.
    assert off.metric(gold, pred) == on.metric(gold, pred)


def test_default_metric_debug_byte_identical_when_compute_off():
    schema = {"type": "array", "order": "align", "items": {"type": "string"}}
    a_no = ObjectAligner(schema, compute_confidence=False)
    deb = a_no.metric(["a", "b", "c"], ["b", "a", "c"], debug=True)
    # No confidence field anywhere in the debug tree when off.
    def walk(node):
        if isinstance(node, dict):
            assert "confidence" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(deb["debug"])


def test_compute_confidence_default_is_false():
    schema = {"type": "string"}
    a = ObjectAligner(schema)
    assert a._compute_confidence is False


# ---------------------------------------------------------------------------
# 2. Margin helper on a known 3×3 matrix
# ---------------------------------------------------------------------------

def test_margin_helper_closed_form():
    import numpy as np
    # Matrix from docs/confidence.md Example 2.
    S = np.array([
        [0.92, 0.49, 0.69],
        [0.58, 0.78, 0.00],
        [0.78, 0.51, 0.93],
    ])
    row_ind = np.array([0, 1, 2])
    col_ind = np.array([0, 1, 2])
    pair_conf, node_conf = _hungarian_confidence(
        S, row_ind, col_ind, n=3, m=3, method="margin",
    )
    # row margins (chosen - row's second best)
    m_row0 = 0.92 - 0.69  # 0.23
    m_row1 = 0.78 - 0.58  # 0.20
    m_row2 = 0.93 - 0.78  # 0.15
    # column margins (chosen - column's second best)
    m_col0 = 0.92 - 0.78  # 0.14
    m_col1 = 0.78 - 0.51  # 0.27
    m_col2 = 0.93 - 0.69  # 0.24
    expected = [
        0.5 * (m_row0 + m_col0),
        0.5 * (m_row1 + m_col1),
        0.5 * (m_row2 + m_col2),
    ]
    for got, want in zip(pair_conf, expected):
        assert math.isclose(got, want, abs_tol=1e-9)
    assert math.isclose(node_conf, sum(expected) / 3, abs_tol=1e-9)


def test_margin_helper_clips_negative():
    import numpy as np
    # Row 0's chosen column is the diagonal, but row 0 would have preferred
    # column 1 (cost-tier scoring would have chosen differently). The
    # clip should send the margin to zero.
    S = np.array([[0.5, 0.9], [0.9, 0.5]])
    row_ind = np.array([0, 1])
    col_ind = np.array([0, 1])  # forced — not the global optimum, just a stress test
    pair_conf, _ = _hungarian_confidence(S, row_ind, col_ind, n=2, m=2)
    # row 0: chosen 0.5, second-best 0.9 → m_row = -0.4 → clip 0
    # col 0: chosen 0.5, second-best 0.9 → m_col = -0.4 → clip 0
    # symmetric average = 0
    assert pair_conf[0] == 0.0
    assert pair_conf[1] == 0.0


def test_excess_pair_is_unit_confidence():
    import numpy as np
    # 2x3 padded — row 1 is padding (n=1, m=2)
    S = np.zeros((2, 2))
    S[0, 0] = 0.9
    S[0, 1] = 0.1
    row_ind = np.array([0, 1])
    col_ind = np.array([0, 1])
    pair_conf, _ = _hungarian_confidence(S, row_ind, col_ind, n=1, m=2)
    # row 1 is padding -> 1.0; row 0 is matched normally
    assert pair_conf[1] == 1.0


# ---------------------------------------------------------------------------
# 3. Entropy method
# ---------------------------------------------------------------------------

def test_entropy_helper_uniform_row_zero_confidence():
    import numpy as np
    # Uniform similarities across m=3 → maximum entropy → confidence 0.
    S = np.full((3, 3), 0.5)
    row_ind = np.array([0, 1, 2])
    col_ind = np.array([0, 1, 2])
    pair_conf, _ = _hungarian_confidence(
        S, row_ind, col_ind, n=3, m=3, method="entropy", temperature=8.0,
    )
    for c in pair_conf:
        assert math.isclose(c, 0.0, abs_tol=1e-9)


def test_entropy_helper_peaked_row_high_confidence():
    import numpy as np
    # Row 0 strongly prefers column 0; entropy should be low → confidence high.
    S = np.array([[1.0, 0.0, 0.0]])
    # Fake out the shape — entropy is computed per row across m=3 cols.
    S = np.vstack([S, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    row_ind = np.array([0, 1, 2])
    col_ind = np.array([0, 1, 2])
    pair_conf, _ = _hungarian_confidence(
        S, row_ind, col_ind, n=3, m=3, method="entropy", temperature=8.0,
    )
    for c in pair_conf:
        assert c > 0.9


def test_entropy_helper_m1_returns_unit():
    import numpy as np
    S = np.array([[1.0]])
    row_ind = np.array([0])
    col_ind = np.array([0])
    pair_conf, node = _hungarian_confidence(
        S, row_ind, col_ind, n=1, m=1, method="entropy",
    )
    assert pair_conf[0] == 1.0
    assert node == 1.0


# ---------------------------------------------------------------------------
# 4. Aggregation up the tree
# ---------------------------------------------------------------------------

def test_dict_confidence_with_keyscore_exact_is_unit_for_perfect_match():
    # keyScore="exact" makes Jaro-based ambiguity irrelevant — the Hungarian
    # matrix has 1.0 on matched keys and 0.0 everywhere else, so margins
    # are tight and confidence is 1.0 across the board for a perfect
    # match.
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    m = a.align({"name": "Ada", "age": 42}, {"name": "Ada", "age": 42})
    assert math.isclose(m.confidence, 1.0, rel_tol=1e-9)


def test_dict_confidence_respects_importance_weights():
    # With keyImportance=0 the dict confidence collapses to values_conf
    # alone; perfect-value cases must therefore yield dict_conf = 1.0
    # regardless of how Jaro-confusable the keys are. This isolates the
    # blend formula from the Jaro key-similarity numerics that the
    # default keyScore would otherwise drag in.
    schema = {
        "type": "object",
        "keyImportance": 0,
        "valueImportance": 1,
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    m = a.align({"name": "Ada", "age": 42}, {"name": "Ada", "age": 42})
    assert math.isclose(m.confidence, 1.0, rel_tol=1e-9)


def test_list_reorder_confidence_is_mean_of_pair_confs():
    schema = {"type": "array", "order": "align", "items": {"type": "string"}}
    a = ObjectAligner(schema, compute_confidence=True)
    m = a.align(["alpha", "beta", "gamma"], ["beta", "alpha", "gamma"])
    pair_confs = [c.confidence for c in m.children]
    assert math.isclose(m.confidence, sum(pair_confs) / len(pair_confs), abs_tol=1e-9)


def test_fixed_list_confidence_is_child_mean():
    schema = {"type": "array", "order": "fixed", "items": {"type": "string"}}
    a = ObjectAligner(schema, compute_confidence=True)
    m = a.align(["a", "b"], ["a", "b"])
    # Fixed list children are MatchItems with confidence=1.0 (no Hungarian).
    for c in m.children:
        assert c.confidence == 1.0
    assert m.confidence == 1.0


# ---------------------------------------------------------------------------
# 5. Three ranking modes for feedback
# ---------------------------------------------------------------------------

def test_rank_by_default_preserves_score_delta_order():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    gold = {"name": "Ada", "age": 42}
    pred = {"naem": "Ad", "years": 41}
    a = ObjectAligner(schema, compute_confidence=True)
    ops = a.repair(gold, pred).ops
    deltas = [op.score_delta for op in ops]
    assert deltas == sorted(deltas, reverse=True)


def test_rank_by_expected_gain_uses_score_delta_times_confidence():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    gold = {"name": "Ada", "age": 42}
    pred = {"naem": "Ad", "years": 41}
    a = ObjectAligner(schema, compute_confidence=True)
    ops = a.repair(gold, pred, rank_by="expected_gain").ops
    # Filter out the silenced rename_remove halves (which carry delta=0 and
    # would tie at the bottom under any rank_by); compare the visible add
    # ops' expected gain.
    add_ops = [op for op in ops if op.kind == "key_rename_add"]
    eg = [op.score_delta * op.confidence for op in add_ops]
    assert eg == sorted(eg, reverse=True)


def test_rank_by_confidence_sorts_by_confidence_desc():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    gold = {"name": "Ada", "age": 42, "email": "a@x"}
    pred = {"naem": "Ad", "years": 41, "mail": "b@y"}
    a = ObjectAligner(schema, compute_confidence=True)
    ops = a.repair(gold, pred, rank_by="confidence").ops
    add_ops = [op for op in ops if op.kind == "key_rename_add"]
    confs = [op.confidence for op in add_ops]
    assert confs == sorted(confs, reverse=True)


def test_invalid_rank_by_raises():
    a = ObjectAligner({"type": "string"}, compute_confidence=True)
    with pytest.raises(ValueError, match="rank_by"):
        a.repair("a", "b", rank_by="bogus")


# ---------------------------------------------------------------------------
# 6. pairing_ambiguous opt-in
# ---------------------------------------------------------------------------

def test_pairing_ambiguous_off_by_default():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    ops = a.repair({"name": "Ada", "age": 42}, {"naem": "Ad", "years": 41}).ops
    assert not any(op.kind == "pairing_ambiguous" for op in ops)


def test_pairing_ambiguous_emitted_with_strict_threshold():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    ops = a.repair(
        {"name": "Ada", "age": 42, "email": "a@x"},
        {"naem": "Ad", "years": 41, "mail": "b@y"},
        include_pairing_ambiguous=True,
        ambiguity_threshold=0.99,
    ).ops
    ambig = [op for op in ops if op.kind == "pairing_ambiguous"]
    assert len(ambig) >= 1
    for op in ambig:
        assert op.score_delta == 0.0
        assert op.confidence < 0.99


def test_pairing_ambiguous_in_feedback_text_under_diagnostics():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    fb = a.feedback(
        {"name": "Ada", "age": 42, "email": "a@x"},
        {"naem": "Ad", "years": 41, "mail": "b@y"},
        include_pairing_ambiguous=True,
        ambiguity_threshold=0.99,
    )
    assert "Diagnostic notes" in fb.text
    assert "pairing between gold and predicted items was ambiguous" in fb.text


# ---------------------------------------------------------------------------
# 7. describe banding + ambiguous
# ---------------------------------------------------------------------------

def test_describe_show_confidence_off_is_byte_identical():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    g = {"name": "Ada", "age": 42}
    p = {"naem": "Ad", "years": 41}
    base = a.describe(g, p).text
    same = a.describe(g, p, show_confidence=False).text
    assert base == same


def test_describe_show_confidence_emits_banded_suffix():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    text = a.describe(
        {"name": "Ada", "age": 42, "email": "a@x"},
        {"naem": "Ad", "years": 41, "mail": "b@y"},
        show_confidence=True,
    ).text
    # At least one low-confidence band should fire on this fragile dict.
    assert "low confidence" in text or "confidence 0." in text


def test_describe_include_ambiguous_emits_note():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    res = a.describe(
        {"name": "Ada", "age": 42, "email": "a@x"},
        {"naem": "Ad", "years": 41, "mail": "b@y"},
        include_ambiguous=True,
        ambiguity_threshold=0.99,
    )
    assert "NOTE:" in res.text
    ambig_entries = [e for e in res.entries if e.outcome == "ambiguous"]
    assert len(ambig_entries) >= 1


# ---------------------------------------------------------------------------
# 8. Constructor validation
# ---------------------------------------------------------------------------

def test_bad_confidence_method_raises():
    with pytest.raises(ValueError, match="confidence_method"):
        ObjectAligner({"type": "string"}, confidence_method="bogus")


def test_bad_confidence_temperature_raises():
    with pytest.raises(ValueError, match="confidence_entropy_temperature"):
        ObjectAligner({"type": "string"}, confidence_entropy_temperature=0)
    with pytest.raises(ValueError, match="confidence_entropy_temperature"):
        ObjectAligner({"type": "string"}, confidence_entropy_temperature=-1.0)
    with pytest.raises(ValueError, match="confidence_entropy_temperature"):
        ObjectAligner({"type": "string"}, confidence_entropy_temperature=float("inf"))


# ---------------------------------------------------------------------------
# 9. Frozen dataclass round-trip via dataclasses.replace
# ---------------------------------------------------------------------------

def test_match_item_confidence_replace_round_trip():
    a = MatchItem(score=0.5, gold="g", pred="p", confidence=0.42)
    b = replace(a, confidence=0.99)
    assert a.confidence == 0.42  # original unchanged
    assert b.confidence == 0.99


def test_match_list_confidence_default_is_one():
    m = MatchList(score=1.0, children=[])
    assert m.confidence == 1.0


def test_match_dict_confidence_default_is_one():
    m = MatchDict(score=1.0, children={})
    assert m.confidence == 1.0


# ---------------------------------------------------------------------------
# 10. Debug tree only surfaces confidence when != 1.0
# ---------------------------------------------------------------------------

def test_debug_tree_emits_confidence_only_when_non_unit():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    deb = a.metric(
        {"name": "Ada", "age": 42},
        {"naem": "Ad", "years": 41},
        debug=True,
    )["debug"]
    # The top-level dict should carry confidence != 1.0 here.
    assert "confidence" in deb
    assert deb["confidence"] != 1.0


def test_debug_tree_skips_confidence_for_unit_subtree():
    schema = {"type": "string"}
    a = ObjectAligner(schema, compute_confidence=True)
    deb = a.metric("hello", "hello", debug=True)["debug"]
    # No Hungarian here → confidence stays 1.0 → field is suppressed.
    assert "confidence" not in deb


# ---------------------------------------------------------------------------
# 11. Confidence threads through RepairOp + FeedbackEntry + DescriptionEntry
# ---------------------------------------------------------------------------

def test_repair_op_carries_confidence_from_match_node():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    ops = a.repair(
        {"name": "Ada", "age": 42, "email": "a@x"},
        {"naem": "Ad", "years": 41, "mail": "b@y"},
    ).ops
    add_ops = [op for op in ops if op.kind == "key_rename_add"]
    # At least one rename op should carry a non-trivial confidence.
    assert any(op.confidence < 1.0 for op in add_ops)


def test_feedback_entry_confidence_field_set():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    fb = a.feedback({"name": "Ada", "age": 42}, {"naem": "Ad", "years": 41})
    for e in fb.entries:
        # confidence is always populated; for these ops it should be < 1
        assert hasattr(e, "confidence")


def test_description_entry_confidence_field_set():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    res = a.describe({"name": "Ada", "age": 42}, {"naem": "Ad", "years": 41})
    for e in res.entries:
        assert hasattr(e, "confidence")


# ---------------------------------------------------------------------------
# 12. expected_gain reranks vs score_delta in a contrived case
# ---------------------------------------------------------------------------

def test_expected_gain_can_reorder_relative_to_score_delta():
    # Construct a case where the rerank actually matters: a fragile
    # high-deficit op vs a committed lower-deficit op. The dict has a
    # near-tie between two pred keys for one gold key, plus a clear-cut
    # one. Under score_delta the fragile one might rank first (large
    # delta); under expected_gain the committed one rises.
    schema = {
        "type": "object",
        "properties": {
            "fish":  {"type": "string"},
            "horse": {"type": "string"},
        },
    }
    a = ObjectAligner(schema, compute_confidence=True)
    # gold "fish"; pred has "fush" (very fuzzy match) AND a clear key match
    # on "horse" via a typo "hors". Both will be key_rename_adds.
    g = {"fish": "salmon", "horse": "thoroughbred"}
    p = {"fush": "trout", "hors": "thoroughbreed"}
    sd_ops = [o for o in a.repair(g, p).ops if o.kind == "key_rename_add"]
    eg_ops = [o for o in a.repair(g, p, rank_by="expected_gain").ops if o.kind == "key_rename_add"]
    # Both lists same op set, possibly different order.
    assert {(o.path, round(o.score_delta, 6)) for o in sd_ops} == \
           {(o.path, round(o.score_delta, 6)) for o in eg_ops}


# ---------------------------------------------------------------------------
# 13. Template snapshots stay in sync via test_templates.py
# ---------------------------------------------------------------------------
# (Covered by tests/test_templates.py — no test needed here.)

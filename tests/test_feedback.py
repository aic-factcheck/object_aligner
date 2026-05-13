"""Tests for prompt-optimizer feedback rendering."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from object_aligner import (
    DEFAULT_FEEDBACK_TEMPLATES,
    FeedbackEntry,
    FeedbackResult,
    ObjectAligner,
    render_feedback,
)
from object_aligner.repair import RepairOp, RepairResult


EPS_TIGHT = 1e-9
EPS_LOOSE = 1e-6


# -----------------------------------------------------------------------------
# Shared schemas
# -----------------------------------------------------------------------------

MOVIE_SCHEMA = {
    "type": "object",
    "keyScore": "exact",
    "keyImportance": 0,
    "valueImportance": 1,
    "properties": {
        "title": {"type": "string", "score": "jaro", "valueWeight": 2.0},
        "year": {"type": "integer", "score": "exact", "valueWeight": 1.0},
        "genres": {
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
            "order": "align",
            "valueWeight": 1.0,
        },
    },
}

FLAT_DICT_SCHEMA = {
    "type": "object",
    "keyScore": "exact",
    "properties": {
        "a": {"type": "string", "score": "exact"},
        "b": {"type": "integer", "score": "exact"},
    },
}

FUZZY_KEY_SCHEMA = {
    "type": "object",
    "keyScore": "jaro",
    "properties": {
        "phoneNumber": {"type": "string", "score": "exact"},
    },
}

FIXED_LIST_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "score": "jaro"},
}

REORDER_LIST_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "score": "jaro"},
    "order": "align",
}


def _movie_aligner(**kwargs):
    return ObjectAligner(MOVIE_SCHEMA, **kwargs)


def _movie_gold_pred():
    gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
    pred = {"title": "The Matrx", "year": 2000, "genres": ["Sci-Fi", "Adventure"]}
    return gold, pred


# -----------------------------------------------------------------------------
# Group A — template validation (construction-time errors)
# -----------------------------------------------------------------------------

def test_unknown_feedback_template_key_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown feedback template keys"):
        ObjectAligner(
            MOVIE_SCHEMA,
            feedback_templates={"feedback.op.unknown": "x"},
        )


def test_bad_placeholder_raises_valueerror():
    with pytest.raises(ValueError, match="unknown placeholder"):
        ObjectAligner(
            MOVIE_SCHEMA,
            feedback_templates={
                "feedback.op.primitive_replace": "{nonsense}",
            },
        )


def test_non_mapping_templates_raises_typeerror():
    with pytest.raises(TypeError, match="must be a mapping"):
        ObjectAligner(MOVIE_SCHEMA, feedback_templates="oops")


def test_non_string_template_value_raises_typeerror():
    with pytest.raises(TypeError, match="must be a string"):
        ObjectAligner(
            MOVIE_SCHEMA,
            feedback_templates={"feedback.intro.perfect": 42},
        )


def test_invalid_style_preset_raises_valueerror():
    with pytest.raises(ValueError, match="feedback_style"):
        ObjectAligner(MOVIE_SCHEMA, feedback_style="weird")


def test_invalid_dominant_fraction_threshold_raises():
    with pytest.raises(ValueError, match="dominant_fraction_threshold"):
        ObjectAligner(MOVIE_SCHEMA, dominant_fraction_threshold="nope")


def test_user_template_with_unused_placeholder_is_ok():
    # User template uses fewer placeholders than allowed — should be fine.
    aligner = ObjectAligner(
        MOVIE_SCHEMA,
        feedback_templates={
            "feedback.op.primitive_replace": "{rank}. fix me",
        },
    )
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred)
    # No format-time KeyError; the customized template renders.
    assert "fix me" in fb.text


# -----------------------------------------------------------------------------
# Group B — top-K and min_score_delta filtering
# -----------------------------------------------------------------------------

def test_top_k_limits_entries():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, top_k=2)
    assert len(fb.entries) == 2
    # Sorted descending by score_delta.
    assert fb.entries[0].score_delta >= fb.entries[1].score_delta


def test_top_k_zero_returns_empty_entries_uses_empty_template():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, top_k=0)
    assert fb.entries == ()
    assert fb.truncated is True
    assert "no individually significant" in fb.text


def test_top_k_none_returns_all():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, top_k=None)
    rep = aligner.repair(gold, pred)
    assert len(fb.entries) == len(rep.ops)


def test_min_score_delta_drops_below_threshold():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    # Title delta is ~0.017 — set threshold above that to drop it.
    fb = aligner.feedback(gold, pred, min_score_delta=0.05, top_k=None)
    paths = [e.path for e in fb.entries]
    assert "/title" not in paths


def test_truncated_flag_true_when_filter_drops_ops():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, min_score_delta=0.5, top_k=None)
    # Nothing has delta >= 0.5 in the Matrix example.
    assert fb.entries == ()
    assert fb.truncated is True


def test_truncated_flag_false_when_all_ops_shown():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, top_k=None)
    assert fb.truncated is False


def test_negative_top_k_raises():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    with pytest.raises(ValueError, match="top_k"):
        aligner.feedback(gold, pred, top_k=-1)


def test_negative_min_score_delta_raises():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    with pytest.raises(ValueError, match="min_score_delta"):
        aligner.feedback(gold, pred, min_score_delta=-0.1)


def test_key_rename_pairs_kept_atomically():
    aligner = ObjectAligner(FUZZY_KEY_SCHEMA)
    gold = {"phoneNumber": "555-1234"}
    pred = {"phone": "555-9999"}
    fb = aligner.feedback(gold, pred, top_k=None)
    kinds = [e.op_kind for e in fb.entries]
    # Both halves of the rename pair are either both present or both absent.
    assert ("key_rename_add" in kinds) == ("key_rename_remove" in kinds)

    # And the same atomicity must hold under min_score_delta filtering.
    fb2 = aligner.feedback(gold, pred, top_k=None, min_score_delta=10.0)
    kinds2 = [e.op_kind for e in fb2.entries]
    assert "key_rename_add" not in kinds2
    assert "key_rename_remove" not in kinds2


# -----------------------------------------------------------------------------
# Group C — style presets
# -----------------------------------------------------------------------------

def test_gepa_style_default_text_includes_synthesis():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred)
    assert fb.style == "gepa"
    # Either single-dominant or mixed synthesis substring appears.
    assert (
        "Focus on" in fb.text
        or "spread across multiple" in fb.text
    )


def test_compact_style_has_no_synthesis_when_disabled():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, style="compact",
                          include_synthesis_line=False)
    assert fb.style == "compact"
    assert "Focus on" not in fb.text
    assert "spread across" not in fb.text


def test_compact_style_each_entry_is_single_line():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, style="compact",
                          include_synthesis_line=False)
    for e in fb.entries:
        if not e.text:
            continue
        assert "\n" not in e.text, f"compact entry has newline: {e.text!r}"


def test_json_style_returns_empty_text_but_populated_entries():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, style="json")
    assert fb.text == ""
    assert fb.style == "json"
    assert len(fb.entries) > 0
    for e in fb.entries:
        assert e.text == ""
        assert e.path  # populated
        assert e.op_kind  # populated


def test_json_style_with_include_metadata_populates_error_breakdown():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, style="json", include_metadata=True)
    assert fb.error_breakdown  # non-empty
    assert all(isinstance(v, float) for v in fb.error_breakdown.values())


# -----------------------------------------------------------------------------
# Group D — each op kind renders correctly
# -----------------------------------------------------------------------------

def test_renders_primitive_replace():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    fb = aligner.feedback("hello", "hallo")
    assert "expected" in fb.text
    assert "got" in fb.text
    assert any(e.op_kind == "primitive_replace" for e in fb.entries)


def test_renders_primitive_replace_reorder():
    aligner = ObjectAligner(REORDER_LIST_SCHEMA)
    fb = aligner.feedback(["dog", "cat"], ["doog", "cat"])
    assert any(e.op_kind == "primitive_replace_reorder" for e in fb.entries)
    assert "inside list" in fb.text


def test_renders_key_add():
    aligner = ObjectAligner(FLAT_DICT_SCHEMA)
    fb = aligner.feedback({"a": "x", "b": 1}, {"a": "x"})
    assert any(e.op_kind == "key_add" for e in fb.entries)
    assert "missing key" in fb.text
    assert '"b"' in fb.text


def test_renders_key_remove():
    aligner = ObjectAligner(FLAT_DICT_SCHEMA)
    fb = aligner.feedback({"a": "x"}, {"a": "x", "b": 1})
    assert any(e.op_kind == "key_remove" for e in fb.entries)
    assert "extraneous key" in fb.text


def test_renders_key_rename_pair_as_one_line():
    aligner = ObjectAligner(FUZZY_KEY_SCHEMA)
    gold = {"phoneNumber": "555-1234"}
    pred = {"phone": "555-1234"}
    fb = aligner.feedback(gold, pred)
    # add half is visible (text non-empty); remove half is silent.
    add_entries = [e for e in fb.entries if e.op_kind == "key_rename_add"]
    rem_entries = [e for e in fb.entries if e.op_kind == "key_rename_remove"]
    assert len(add_entries) == 1
    assert len(rem_entries) == 1
    assert add_entries[0].text  # non-empty
    assert rem_entries[0].text == ""  # silenced by default empty template
    # Both inherit the same visible rank (contiguous numbering).
    assert add_entries[0].rank == rem_entries[0].rank
    assert "rename" in fb.text


def test_renders_list_item_add():
    aligner = ObjectAligner(FIXED_LIST_SCHEMA)
    fb = aligner.feedback(["a", "b", "c"], ["a", "b"])
    assert any(e.op_kind == "list_item_add" for e in fb.entries)
    assert "missing list item" in fb.text


def test_renders_list_item_remove():
    aligner = ObjectAligner(FIXED_LIST_SCHEMA)
    fb = aligner.feedback(["a", "b"], ["a", "b", "c"])
    assert any(e.op_kind == "list_item_remove" for e in fb.entries)
    assert "extraneous list item" in fb.text


def test_renders_list_item_missing():
    aligner = ObjectAligner(REORDER_LIST_SCHEMA)
    fb = aligner.feedback(["alpha", "beta", "gamma"], ["alpha", "beta"])
    assert any(e.op_kind == "list_item_missing" for e in fb.entries)
    assert "list is missing item" in fb.text


def test_renders_list_item_excess():
    aligner = ObjectAligner(REORDER_LIST_SCHEMA)
    fb = aligner.feedback(["alpha", "beta"], ["alpha", "beta", "gamma"])
    assert any(e.op_kind == "list_item_excess" for e in fb.entries)
    assert "extraneous item" in fb.text


def test_renders_ref_fix():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "entities": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "keyScore": "exact",
                    "properties": {
                        "id": {"type": "string", "idScope": "entity"},
                        "name": {"type": "string", "score": "exact"},
                    },
                },
            },
            "primary": {"type": "string", "ref": "entity"},
        },
    }
    gold = {
        "entities": [
            {"id": "g1", "name": "Alice"},
            {"id": "g2", "name": "Bob"},
        ],
        "primary": "g1",
    }
    pred = {
        "entities": [
            {"id": "p1", "name": "Alice"},
            {"id": "p2", "name": "Bob"},
        ],
        "primary": "p2",  # points at Bob, but gold says Alice
    }
    aligner = ObjectAligner(schema)
    fb = aligner.feedback(gold, pred)
    assert any(e.op_kind == "ref_fix" for e in fb.entries)
    assert "wrong reference" in fb.text


def test_renders_subtree_replace():
    aligner = ObjectAligner(FLAT_DICT_SCHEMA)
    fb = aligner.feedback(
        {"a": "x", "b": 1}, {"a": "y", "b": 2},
        granularity="subtree",
    )
    assert any(e.op_kind == "subtree_replace" for e in fb.entries)
    assert "subtree differs" in fb.text


# -----------------------------------------------------------------------------
# Group E — synthesis line
# -----------------------------------------------------------------------------

def test_single_dominant_fires_when_one_kind_above_threshold():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred)
    # All three Matrix ops are primitive (one is replace_reorder, two are
    # plain replace) — collapses to one human kind "primitive-value" via
    # _OP_KIND_HUMAN. But note that primitive_replace and
    # primitive_replace_reorder map to DIFFERENT human kinds, so this fixture
    # actually hits "mixed". Use a flat-dict fixture to force single-dominant.
    aligner_flat = ObjectAligner(FLAT_DICT_SCHEMA)
    gold_flat = {"a": "x", "b": 1}
    pred_flat = {"a": "y", "b": 2}
    fb_flat = aligner_flat.feedback(gold_flat, pred_flat)
    # Both ops are primitive_replace -> one bucket -> 100% -> single dominant.
    assert "Focus on" in fb_flat.text
    assert "primitive-value" in fb_flat.text


def test_mixed_fires_when_no_kind_above_threshold():
    aligner = ObjectAligner(FLAT_DICT_SCHEMA)
    # Mix of key_add and key_remove and primitive_replace = three buckets.
    gold = {"a": "x", "b": 1}
    pred = {"a": "y"}  # b missing -> key_add; no extra; a wrong -> primitive
    fb = aligner.feedback(gold, pred,
                          dominant_fraction_threshold=0.99)
    # Force the high threshold to ensure even a 50/50 split is "mixed".
    assert "spread across" in fb.text


def test_dominant_fraction_threshold_configurable_per_call():
    aligner = ObjectAligner(FLAT_DICT_SCHEMA)
    gold = {"a": "x", "b": 1}
    pred = {"a": "y", "b": 2}
    # Even at very high threshold, all are primitive_replace -> 100% -> single.
    fb = aligner.feedback(gold, pred, dominant_fraction_threshold=0.99)
    assert "Focus on" in fb.text
    # At threshold above 1.0, nothing can ever cross it -> always mixed.
    fb2 = aligner.feedback(gold, pred, dominant_fraction_threshold=1.01)
    assert "spread across" in fb2.text


def test_dominant_fraction_threshold_configurable_via_constructor():
    aligner_hi = ObjectAligner(FLAT_DICT_SCHEMA,
                                dominant_fraction_threshold=1.01)
    gold = {"a": "x", "b": 1}
    pred = {"a": "y", "b": 2}
    fb = aligner_hi.feedback(gold, pred)
    assert "spread across" in fb.text


def test_include_synthesis_line_false_omits_synthesis():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, include_synthesis_line=False)
    assert "Focus on" not in fb.text
    assert "spread across" not in fb.text


def test_perfect_score_uses_intro_perfect():
    aligner = _movie_aligner()
    gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
    fb = aligner.feedback(gold, gold)
    assert fb.score == pytest.approx(1.0, abs=EPS_TIGHT)
    assert "perfectly matches" in fb.text
    assert fb.entries == ()


# -----------------------------------------------------------------------------
# Group F — public API + metric() integration
# -----------------------------------------------------------------------------

def test_aligner_feedback_returns_feedbackresult():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred)
    assert isinstance(fb, FeedbackResult)


def test_feedback_from_match_matches_aligner_feedback():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    # Use the public _align_with_ctx via repair() to get a match tree.
    match_tree = aligner.align(gold, pred)
    # We need mappings — but the Matrix schema has no refs, so {} works.
    fb1 = aligner.feedback(gold, pred)
    fb2 = aligner.feedback_from_match(match_tree, gold, pred, {})
    assert fb1.text == fb2.text
    assert fb1.score == fb2.score
    assert len(fb1.entries) == len(fb2.entries)


def test_metric_with_generate_feedback_true_adds_string():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    r = aligner.metric(gold, pred, generate_feedback=True)
    assert "feedback" in r
    assert isinstance(r["feedback"], str)
    assert r["feedback"]


def test_metric_with_generate_feedback_full_adds_dict():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    r = aligner.metric(gold, pred, generate_feedback="full")
    assert "feedback" in r
    assert isinstance(r["feedback"], dict)
    assert set(r["feedback"]) == {
        "score", "text", "entries", "style", "truncated",
        "n_total_ops", "error_breakdown",
    }
    assert r["feedback"]["error_breakdown"]  # full mode populates breakdown


def test_metric_with_generate_feedback_false_omits_key():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    r = aligner.metric(gold, pred, generate_feedback=False)
    assert "feedback" not in r


def test_metric_constructor_default_generate_feedback_flows_through():
    aligner = ObjectAligner(MOVIE_SCHEMA, generate_feedback=True)
    gold, pred = _movie_gold_pred()
    r = aligner.metric(gold, pred)
    assert "feedback" in r
    assert isinstance(r["feedback"], str)


def test_metric_invalid_generate_feedback_value_raises():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    with pytest.raises(ValueError, match="generate_feedback"):
        aligner.metric(gold, pred, generate_feedback="garbage")


def test_metric_reasoning_and_feedback_both_present():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    r = aligner.metric(
        gold, pred, generate_reasoning=True, generate_feedback=True,
    )
    assert isinstance(r["reasoning"], str)
    assert isinstance(r["feedback"], str)
    # Distinct outputs.
    assert r["reasoning"] != r["feedback"]


def test_metric_validation_failure_with_feedback_renders_error_text():
    aligner = _movie_aligner()
    r = aligner.metric(
        {"title": "x", "year": 1, "genres": []},
        "not a dict",
        generate_feedback=True,
    )
    assert r["score"] == 0.0
    assert "failed schema validation" in r["feedback"]


def test_metric_validation_failure_with_feedback_full_returns_dict():
    aligner = _movie_aligner()
    r = aligner.metric(
        {"title": "x", "year": 1, "genres": []},
        "not a dict",
        generate_feedback="full",
    )
    assert r["score"] == 0.0
    assert isinstance(r["feedback"], dict)
    assert r["feedback"]["score"] == 0.0
    assert r["feedback"]["entries"] == []
    assert "failed schema validation" in r["feedback"]["text"]


def test_feedback_skip_validation():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    # skip_validation=True bypasses the schema check (but inputs are valid
    # here anyway). Ensure it doesn't blow up.
    fb = aligner.feedback(gold, pred, skip_validation=True)
    assert isinstance(fb, FeedbackResult)


def test_feedback_validation_failure_path_returns_degenerate_result():
    aligner = _movie_aligner()
    gold = {"title": "x", "year": 1, "genres": []}
    fb = aligner.feedback(gold, "not a dict")
    assert fb.score == 0.0
    assert fb.entries == ()
    assert "failed schema validation" in fb.text


# -----------------------------------------------------------------------------
# Group G — edge cases
# -----------------------------------------------------------------------------

def test_perfect_match_no_entries_perfect_intro():
    aligner = _movie_aligner()
    gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi"]}
    fb = aligner.feedback(gold, gold)
    assert fb.entries == ()
    assert "perfectly matches" in fb.text
    assert fb.n_total_ops == 0


def test_empty_object_against_filled():
    aligner = ObjectAligner(FLAT_DICT_SCHEMA)
    fb = aligner.feedback({"a": "x", "b": 1}, {})
    # Two missing keys -> two key_add ops.
    assert len(fb.entries) == 2
    assert all(e.op_kind == "key_add" for e in fb.entries)


def test_thread_safety_concurrent_feedback_calls():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()

    def one():
        return aligner.feedback(gold, pred).text

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: one(), range(32)))

    # All outputs identical (alignment is deterministic).
    assert len(set(results)) == 1


# -----------------------------------------------------------------------------
# Group H — configurability / overrides
# -----------------------------------------------------------------------------

def test_user_template_override_renders_user_text():
    aligner = ObjectAligner(
        MOVIE_SCHEMA,
        feedback_templates={
            "feedback.op.primitive_replace":
                "FIX-{rank} {path}: {gold}!={pred}",
        },
    )
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred)
    assert "FIX-1 " in fb.text or "FIX-2 " in fb.text or "FIX-3 " in fb.text


def test_value_formatter_truncates_long_values():
    long_value = "x" * 500
    aligner = ObjectAligner({"type": "string", "score": "exact"})
    fb = render_feedback(
        aligner.repair("short", long_value),
        value_formatter=lambda v: repr(v)[:20],
    )
    # No rendered value blob exceeds the cap.
    assert all(len(s) <= 30 for s in fb.text.split())


def test_default_value_formatter_caps_at_80_chars():
    long_value = "x" * 500
    aligner = ObjectAligner({"type": "string", "score": "exact"})
    fb = aligner.feedback("short", long_value)
    # The repr of the long string should be truncated with an ellipsis.
    assert "…" in fb.text


def test_constructor_feedback_templates_flow_through_metric():
    aligner = ObjectAligner(
        MOVIE_SCHEMA,
        generate_feedback=True,
        feedback_templates={
            "feedback.intro.imperfect":
                "CUSTOM-INTRO score={score:.2f}\n",
        },
    )
    gold, pred = _movie_gold_pred()
    r = aligner.metric(gold, pred)
    assert r["feedback"].startswith("CUSTOM-INTRO score=")


def test_style_kwarg_overrides_constructor_default():
    aligner = ObjectAligner(MOVIE_SCHEMA, feedback_style="gepa")
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, style="json")
    assert fb.style == "json"
    assert fb.text == ""


def test_compact_overrides_apply_when_style_compact():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, style="compact",
                          include_synthesis_line=False)
    # Compact intro pattern.
    assert fb.text.startswith("Score ")
    assert "fixes" in fb.text


def test_user_override_wins_over_compact_preset():
    aligner = ObjectAligner(
        MOVIE_SCHEMA,
        feedback_templates={
            "feedback.intro.imperfect": "MY-INTRO\n",
        },
    )
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, style="compact",
                          include_synthesis_line=False)
    assert fb.text.startswith("MY-INTRO\n")


# -----------------------------------------------------------------------------
# Group I — sum-and-rank invariants
# -----------------------------------------------------------------------------

INVARIANT_FIXTURES = [
    (
        {"type": "string", "score": "jaro"},
        "hello", "hallo",
        "primitive string",
    ),
    (
        FLAT_DICT_SCHEMA,
        {"a": "alice", "b": 1},
        {"a": "alicia", "b": 2},
        "flat dict mismatches",
    ),
    (
        FLAT_DICT_SCHEMA,
        {"a": "x", "b": 1},
        {"a": "x"},
        "flat dict missing key",
    ),
    (
        FIXED_LIST_SCHEMA,
        ["a", "b", "c"],
        ["a", "c"],
        "fixed list missing item",
    ),
    (
        REORDER_LIST_SCHEMA,
        ["sci-fi", "drama"],
        ["drama", "sci-fy"],
        "reorder list with mismatch",
    ),
    (
        MOVIE_SCHEMA,
        {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]},
        {"title": "The Matrx", "year": 2000, "genres": ["Sci-Fi", "Adventure"]},
        "movie schema",
    ),
]


@pytest.mark.parametrize("schema,gold,pred,desc", INVARIANT_FIXTURES)
def test_entries_sorted_by_score_delta_desc(schema, gold, pred, desc):
    aligner = ObjectAligner(schema)
    fb = aligner.feedback(gold, pred, top_k=None)
    deltas = [e.score_delta for e in fb.entries]
    assert deltas == sorted(deltas, reverse=True), (
        f"entries not sorted desc for fixture: {desc}"
    )


@pytest.mark.parametrize("schema,gold,pred,desc", INVARIANT_FIXTURES)
def test_n_total_ops_matches_repair_result_len(schema, gold, pred, desc):
    aligner = ObjectAligner(schema)
    fb = aligner.feedback(gold, pred, top_k=None, include_metadata=True)
    rep = aligner.repair(gold, pred)
    assert fb.n_total_ops == len(rep.ops), f"fixture: {desc}"


@pytest.mark.parametrize("schema,gold,pred,desc", INVARIANT_FIXTURES)
def test_error_breakdown_matches_repair_total_delta(schema, gold, pred, desc):
    aligner = ObjectAligner(schema)
    fb = aligner.feedback(gold, pred, top_k=None, include_metadata=True)
    rep = aligner.repair(gold, pred)
    breakdown_sum = sum(fb.error_breakdown.values())
    assert breakdown_sum == pytest.approx(rep.total_delta, abs=EPS_LOOSE), (
        f"fixture: {desc}"
    )


# -----------------------------------------------------------------------------
# Group J — render_feedback functional API directly on a synthesized
# RepairResult (covers paths that aligner.feedback doesn't easily exercise)
# -----------------------------------------------------------------------------

def test_render_feedback_on_empty_repair_result():
    # No ops at all (perfect match).
    rep = RepairResult(
        score=1.0, ops=(), granularity="leaf",
        total_delta=0.0, residual=0.0, notes=(),
    )
    fb = render_feedback(rep)
    assert fb.score == 1.0
    assert fb.entries == ()
    assert "perfectly matches" in fb.text


def test_render_feedback_unknown_op_kind_raises():
    bogus_op = RepairOp(
        op="replace",
        path="/x",
        score_delta=0.5,
        kind="this_kind_does_not_exist",
        gold="a",
        pred="b",
    )
    rep = RepairResult(
        score=0.5, ops=(bogus_op,), granularity="leaf",
        total_delta=0.5, residual=0.0, notes=(),
    )
    with pytest.raises((KeyError, ValueError)):
        render_feedback(rep)


def test_render_feedback_invalid_style_raises():
    rep = RepairResult(
        score=1.0, ops=(), granularity="leaf",
        total_delta=0.0, residual=0.0, notes=(),
    )
    with pytest.raises(ValueError, match="style"):
        render_feedback(rep, style="invalid")


def test_feedback_result_str_returns_text():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred)
    assert str(fb) == fb.text


def test_feedback_result_indexable_and_iterable():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred)
    # Indexable, iterable, len-able.
    assert fb[0] is fb.entries[0]
    assert list(fb) == list(fb.entries)
    assert len(fb) == len(fb.entries)


def test_to_dict_round_trips_basic_types():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    fb = aligner.feedback(gold, pred, include_metadata=True)
    d = fb.to_dict()
    # All entries are dicts of basic types.
    for entry_dict in d["entries"]:
        assert isinstance(entry_dict, dict)
        assert isinstance(entry_dict["rank"], int)
        assert isinstance(entry_dict["score_delta"], float)
    # Top-level shape.
    assert d["score"] == fb.score
    assert d["text"] == fb.text


def test_default_feedback_templates_has_18_keys():
    assert len(DEFAULT_FEEDBACK_TEMPLATES) == 18


def test_feedback_entry_is_frozen():
    e = FeedbackEntry(
        rank=1, op_kind="x", op="add", path="/", score_delta=0.0,
        score_delta_pct=0.0, gold=None, pred=None, text="",
    )
    with pytest.raises(Exception):  # dataclass FrozenInstanceError
        e.rank = 99  # type: ignore[misc]


def test_feedback_result_is_frozen():
    fb = FeedbackResult(score=1.0, text="", entries=(), style="gepa",
                        truncated=False, n_total_ops=0)
    with pytest.raises(Exception):
        fb.score = 0.5  # type: ignore[misc]

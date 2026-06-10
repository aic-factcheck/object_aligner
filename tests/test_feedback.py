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
    "keyImportance": 1,
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


def test_ref_fix_text_is_pred_space_only():
    """The ref_fix line surfaces the pred-space replacement value, not the
    gold-space id, and never includes the gold id in the rendered text.
    Guards against gold leaking into prompt-optimizer feedback (GEPA hides
    gold from the optimizer).
    """
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
                        "id":   {"type": "string", "idScope": "entity"},
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
        "primary": "p2",  # points at Bob; gold expects Alice
    }
    aligner = ObjectAligner(schema)
    fb = aligner.feedback(gold, pred)
    ref_entry = next(e for e in fb.entries if e.op_kind == "ref_fix")
    # The pred-space replacement value (`p1`, the bijection image of gold
    # `g1`) appears in the rendered text; the gold-space id (`g1`) does not.
    assert "p1" in ref_entry.text
    assert "g1" not in ref_entry.text
    assert "g2" not in ref_entry.text
    # The synthesis-line bucket is "reference", not "unresolved-reference".
    assert "reference errors" in fb.text


def test_ref_fix_no_target_emitted_when_gold_referent_missing():
    """When gold references a definer that has no counterpart in pred under
    the derived bijection, the op kind is `ref_fix_no_target` and the text
    explains that the reference cannot be resolved — without leaking any
    gold-space id.
    """
    schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":    {"type": "integer", "idScope": "node"},
                        "color": {"type": "string", "score": "exact"},
                    },
                },
            },
            "edges": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "integer", "ref": "node"},
                        "target": {"type": "integer", "ref": "node"},
                    },
                },
            },
        },
    }
    aligner = ObjectAligner(schema)
    gold = {
        "nodes": [
            {"id": 0, "color": "blue"},
            {"id": 1, "color": "green"},
            {"id": 9, "color": "red"},
        ],
        "edges": [{"source": 0, "target": 9}],
    }
    pred = {
        "nodes": [
            {"id": 0, "color": "blue"},
            {"id": 1, "color": "green"},
        ],
        "edges": [{"source": 0, "target": 1}],
    }
    fb = aligner.feedback(gold, pred)
    no_target = [e for e in fb.entries if e.op_kind == "ref_fix_no_target"]
    assert len(no_target) == 1
    entry = no_target[0]
    assert entry.path == "/edges/0/target"
    assert "cannot be resolved" in entry.text
    # No gold-space id ("9") leaks into the user-visible text.
    assert "9" not in entry.text
    # Synthesis line buckets ref_fix_no_target under "unresolved-reference".
    assert "unresolved-reference" in fb.text


def test_ref_fix_no_target_apply_to_still_chains_to_one():
    """apply_to() carries the gold-side id as a best-effort replacement on
    ref_fix_no_target ops so that, combined with the sibling
    list_item_missing op, the joint apply still reaches score 1.0.
    """
    schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":    {"type": "integer", "idScope": "node"},
                        "color": {"type": "string", "score": "exact"},
                    },
                },
            },
            "edges": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "integer", "ref": "node"},
                        "target": {"type": "integer", "ref": "node"},
                    },
                },
            },
        },
    }
    aligner = ObjectAligner(schema)
    gold = {
        "nodes": [
            {"id": 0, "color": "blue"},
            {"id": 1, "color": "green"},
            {"id": 9, "color": "red"},
        ],
        "edges": [{"source": 0, "target": 9}],
    }
    pred = {
        "nodes": [
            {"id": 0, "color": "blue"},
            {"id": 1, "color": "green"},
        ],
        "edges": [{"source": 0, "target": 1}],
    }
    r = aligner.repair(gold, pred)
    patched = r.apply_to(pred)
    assert aligner.metric(gold, patched)["score"] == pytest.approx(1.0)


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


def test_metric_description_and_feedback_both_present():
    aligner = _movie_aligner()
    gold, pred = _movie_gold_pred()
    r = aligner.metric(
        gold, pred, generate_description=True, generate_feedback=True,
    )
    assert isinstance(r["description"], str)
    assert isinstance(r["feedback"], str)
    # Distinct outputs.
    assert r["description"] != r["feedback"]


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


def test_default_feedback_templates_has_30_keys():
    # 19 op/intro/synthesis/empty/validation keys + 2 confidence keys
    # (feedback.op.pairing_ambiguous and feedback.diagnostics.intro)
    # added in the confidence (cluster 4) release + 1 ref_fix_no_target
    # added in the pred-space-ref-feedback change + 8 semantic referential
    # feedback keys (6 refsem fragments + 2 .semantic op skeletons).
    assert len(DEFAULT_FEEDBACK_TEMPLATES) == 30


def test_feedback_invalid_pred_renders_validation_error_text():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner(schema)
    result = aligner.feedback({"name": "A", "age": 1}, {"name": "A"})
    assert result.score == 0.0
    assert result.entries == ()
    assert "failed schema validation" in result.text


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


# -----------------------------------------------------------------------------
# Semantic referential feedback (referential_feedback="semantic")
# -----------------------------------------------------------------------------

def _amr_schema():
    return {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array", "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "idScope": "node"},
                        "concept": {"type": "string", "score": "exact"},
                    },
                },
            },
            "relations": {
                "type": "array", "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "score": "exact"},
                        "source": {"type": "string", "ref": "node"},
                        "target": {"type": "string", "ref": "node"},
                    },
                },
            },
        },
    }


def _amr_gold_pred():
    gold = {
        "nodes": [{"id": "g1", "concept": "confirm-01"},
                  {"id": "g2", "concept": "protein"}],
        "relations": [{"label": ":ARG0", "source": "g1", "target": "g2"}],
    }
    pred = {
        "nodes": [{"id": "p1", "concept": "confirm-01"},
                  {"id": "p2", "concept": "protein"}],
        # source points at the protein node; gold says the confirm-01 node
        "relations": [{"label": ":ARG0", "source": "p2", "target": "p2"}],
    }
    return gold, pred


def test_referential_feedback_invalid_value_raises():
    with pytest.raises(ValueError, match="referential_feedback"):
        ObjectAligner(_amr_schema(), referential_feedback="bogus")


def test_semantic_renders_gold_endpoint_props_and_label():
    gold, pred = _amr_gold_pred()
    fb = ObjectAligner(_amr_schema(), referential_feedback="semantic").feedback(
        gold, pred
    )
    line = next(e.text for e in fb.entries if e.op_kind == "ref_fix")
    assert "concept 'confirm-01'" in line          # gold endpoint property
    assert "':ARG0'" in line                        # relation label
    assert "concept 'protein'" in line              # wrong node it used
    assert "should point to" in line
    # No gold-space id leaks.
    assert "g1" not in line and "g2" not in line


def test_semantic_score_identical_to_literal():
    gold, pred = _amr_gold_pred()
    lit = ObjectAligner(_amr_schema()).feedback(gold, pred)
    sem = ObjectAligner(_amr_schema(), referential_feedback="semantic").feedback(
        gold, pred
    )
    assert lit.score == sem.score
    # Only the ref line text differs; structured fields are unchanged.
    assert [e.op_kind for e in lit.entries] == [e.op_kind for e in sem.entries]
    assert [e.score_delta for e in lit.entries] == [e.score_delta for e in sem.entries]


def test_default_is_literal_and_byte_identical():
    gold, pred = _amr_gold_pred()
    aligner = ObjectAligner(_amr_schema())  # default referential_feedback
    assert (
        aligner.feedback(gold, pred).text
        == aligner.feedback(gold, pred, referential_feedback="literal").text
    )
    # The default literal line uses the opaque-id phrasing.
    assert "wrong reference" in aligner.feedback(gold, pred).text


def test_per_call_overrides_constructor_default():
    gold, pred = _amr_gold_pred()
    # Constructor literal, per-call semantic.
    a = ObjectAligner(_amr_schema())
    assert "should point to" in a.feedback(gold, pred, referential_feedback="semantic").text
    # Constructor semantic, per-call literal.
    b = ObjectAligner(_amr_schema(), referential_feedback="semantic")
    assert "wrong reference" in b.feedback(gold, pred, referential_feedback="literal").text


def test_semantic_multi_prop_endpoint():
    schema = {
        "type": "object",
        "properties": {
            "atoms": {
                "type": "array", "order": "align",
                "items": {"type": "object", "properties": {
                    "idx": {"type": "integer", "idScope": "atom"},
                    "element": {"type": "string", "score": "exact"},
                    "charge": {"type": "integer"},
                    "num_h": {"type": "integer"},
                }},
            },
            "bonds": {
                "type": "array", "order": "align",
                "items": {"type": "object", "properties": {
                    "order": {"type": "string", "score": "exact"},
                    "source": {"type": "integer", "ref": "atom"},
                    "target": {"type": "integer", "ref": "atom"},
                }},
            },
        },
    }
    gold = {
        "atoms": [{"idx": 1, "element": "C", "charge": 0, "num_h": 0},
                  {"idx": 2, "element": "N", "charge": 0, "num_h": 1}],
        "bonds": [{"order": "double", "source": 1, "target": 2}],
    }
    # pred is missing the carbon entirely -> no_target for source.
    pred = {
        "atoms": [{"idx": 9, "element": "N", "charge": 0, "num_h": 1}],
        "bonds": [{"order": "double", "source": 9, "target": 9}],
    }
    fb = ObjectAligner(schema, referential_feedback="semantic").feedback(gold, pred)
    line = next(e.text for e in fb.entries if e.op_kind == "ref_fix_no_target")
    assert "element 'C', charge 0, num_h 0" in line
    assert "'double'" in line
    assert "no such atom" in line


def test_semantic_falls_back_to_literal_when_no_discriminator():
    """A definer whose only field is the id has no discriminating property,
    so the semantic line falls back to the literal rendering for that op."""
    schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array", "order": "align",
                "items": {"type": "object", "properties": {
                    "id": {"type": "string", "idScope": "node"},
                }},
            },
            "relations": {
                "type": "array", "order": "align",
                "items": {"type": "object", "properties": {
                    "source": {"type": "string", "ref": "node"},
                    "target": {"type": "string", "ref": "node"},
                }},
            },
        },
    }
    gold = {"nodes": [{"id": "g1"}, {"id": "g2"}],
            "relations": [{"source": "g1", "target": "g2"}]}
    pred = {"nodes": [{"id": "p1"}, {"id": "p2"}],
            "relations": [{"source": "p2", "target": "p1"}]}
    lit = ObjectAligner(schema).feedback(gold, pred)
    sem = ObjectAligner(schema, referential_feedback="semantic").feedback(gold, pred)
    assert lit.text == sem.text


def test_semantic_no_op_on_non_referential_schema():
    schema = {"type": "object", "properties": {"a": {"type": "string", "score": "exact"}}}
    lit = ObjectAligner(schema).feedback({"a": "x"}, {"a": "y"})
    sem = ObjectAligner(schema, referential_feedback="semantic").feedback(
        {"a": "x"}, {"a": "y"}
    )
    assert lit.text == sem.text


def test_semantic_metric_honors_constructor_default():
    gold, pred = _amr_gold_pred()
    a = ObjectAligner(_amr_schema(), generate_feedback=True,
                      referential_feedback="semantic")
    out = a.metric(gold, pred)
    assert "should point to" in out["feedback"]
    # Score path unaffected.
    assert out["score"] == ObjectAligner(_amr_schema()).metric(gold, pred)["score"]


def test_semantic_endpoint_certain_flag_detects_property_twin():
    """When the gold endpoint shares its property signature with another gold
    definer, the descriptor flags it uncertain and the renderer hedges."""
    aligner = ObjectAligner(_amr_schema(), referential_feedback="semantic")
    gold = {
        "nodes": [{"id": "g1", "concept": "thing"},
                  {"id": "g2", "concept": "thing"},
                  {"id": "g3", "concept": "other"}],
        "relations": [],
    }
    pred = gold  # contents irrelevant; we test the descriptor helper directly
    op_for_g1 = RepairOp(op="replace", path="/relations/0/source",
                         score_delta=0.1, kind="ref_fix", gold="g1", pred="g3")
    descs = aligner._build_ref_endpoint_descriptors(gold, pred, [op_for_g1])
    d = descs["/relations/0/source"]
    assert d.usable is True
    assert d.endpoint_certain is False          # g1 has a twin (g2)
    assert ("concept", "thing") in d.gold_props


def test_semantic_never_raises_on_dangling_pred_ref():
    gold, pred = _amr_gold_pred()
    # Make pred reference an id that exists in neither node list.
    pred = {
        "nodes": [{"id": "p1", "concept": "confirm-01"},
                  {"id": "p2", "concept": "protein"}],
        "relations": [{"label": ":ARG0", "source": "zzz", "target": "p2"}],
    }
    fb = ObjectAligner(_amr_schema(), referential_feedback="semantic").feedback(
        gold, pred, skip_validation=True
    )
    assert isinstance(fb.text, str)

"""Tests for the describe() surface that goes beyond metric()'s wiring.

The metric()-level happy-path tests live in ``tests/test_api.py`` —
``test_metric_returns_description_when_enabled_in_constructor`` and
friends. Tests here exercise the *direct* ``aligner.describe(...)``
method, the structured ``DescriptionResult`` / ``DescriptionEntry``
surface, the ``style="json"`` preset, and the ``generate_description="full"``
mode.
"""

import pytest

from object_aligner import (
    DescriptionEntry,
    DescriptionResult,
    ObjectAligner,
    render_description,
)


# -----------------------------------------------------------------------------
# Direct aligner.describe() method
# -----------------------------------------------------------------------------

def test_describe_returns_description_result_for_imperfect_primitive():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    dr = aligner.describe("hello", "helo")

    assert isinstance(dr, DescriptionResult)
    assert dr.score == pytest.approx(0.9333, abs=1e-3)
    assert dr.text.startswith("The predicted output scores overall")
    # Default style still populates entries for the leaf walk.
    assert len(dr.entries) == 1
    assert dr.entries[0].match_kind == "item"
    assert dr.entries[0].outcome == "mismatch"


def test_describe_perfect_match_short_circuits_intro_text_with_no_entries():
    aligner = ObjectAligner({"type": "string", "score": "exact"})
    dr = aligner.describe("x", "x")
    assert dr.text == "The predicted output perfectly matches the gold."
    assert dr.entries == ()


def test_describe_str_returns_text():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    dr = aligner.describe("hello", "helo")
    assert str(dr) == dr.text


def test_describe_iterates_entries_in_traversal_order():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "name": {"type": "string", "score": "jaro"},
            "age":  {"type": "integer", "score": "exact"},
        },
    }
    aligner = ObjectAligner(schema)
    dr = aligner.describe(
        {"name": "Alice", "age": 30},
        {"name": "Alic",  "age": 31},
    )
    kinds = [e.match_kind for e in dr]
    # Dict root, then KEY/VALUE pair for each child.
    assert kinds[0] == "dict"
    assert "key" in kinds
    assert "item" in kinds


def test_describe_from_match_works_against_precomputed_tree():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    tree = aligner.align("hello", "helo")
    dr = aligner.describe_from_match(tree)
    assert dr.score == pytest.approx(0.9333, abs=1e-3)
    assert "helo" in dr.text


# -----------------------------------------------------------------------------
# style="json"
# -----------------------------------------------------------------------------

def test_json_style_empties_text_but_populates_entries():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    dr = aligner.describe("hello", "helo", style="json")
    assert dr.text == ""
    assert dr.style == "json"
    assert len(dr.entries) == 1
    assert dr.entries[0].text  # individual entry texts are rendered


def test_json_style_round_trips_through_to_dict():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    dr = aligner.describe("hello", "helo", style="json")
    d = dr.to_dict()
    assert d["text"] == ""
    assert d["style"] == "json"
    assert d["score"] == pytest.approx(0.9333, abs=1e-3)
    assert isinstance(d["entries"], list)
    assert d["entries"][0]["match_kind"] == "item"


def test_unknown_style_raises_at_render_time():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    with pytest.raises(ValueError, match="style must be one of"):
        aligner.describe("hello", "helo", style="bogus")


def test_unknown_style_in_constructor_raises():
    with pytest.raises(ValueError, match="description_style must be one of"):
        ObjectAligner({"type": "string"}, description_style="bogus")


# -----------------------------------------------------------------------------
# generate_description="full"
# -----------------------------------------------------------------------------

def test_metric_generate_description_full_returns_structured_dict():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    result = aligner.metric("hello", "helo", generate_description="full")
    assert isinstance(result["description"], dict)
    assert set(result["description"]) == {"score", "text", "entries", "style"}
    assert result["description"]["style"] == "default"


def test_metric_invalid_generate_description_value_raises():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    with pytest.raises(ValueError, match="generate_description"):
        aligner.metric("hello", "helo", generate_description="garbage")


def test_metric_invalid_pred_with_generate_description_full_returns_dict():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner(schema)
    result = aligner.metric(
        {"a": "x", "b": 1}, {"a": "x"}, generate_description="full",
    )
    assert result["score"] == 0.0
    assert isinstance(result["description"], dict)
    assert "JSON Schema validation failed" in result["description"]["text"]
    assert result["description"]["entries"] == []


# -----------------------------------------------------------------------------
# Functional render_description() — module-level entry point
# -----------------------------------------------------------------------------

def test_module_level_render_description_uses_match_tree_directly():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    tree = aligner.align("hello", "helo")
    dr = render_description(tree)
    assert dr.score == pytest.approx(0.9333, abs=1e-3)
    assert isinstance(dr, DescriptionResult)


def test_description_entry_path_for_list_reorder_is_list_path():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "items": {
                "type": "array",
                "order": "align",
                "items": {"type": "string", "score": "jaro"},
            },
        },
    }
    aligner = ObjectAligner(schema)
    dr = aligner.describe(
        {"items": ["alpha", "beta"]},
        {"items": ["alpha", "betz"]},
    )
    list_item_entries = [
        e for e in dr.entries if e.match_kind == "item"
    ]
    # All reorder-list children share the list path /items.
    assert all(e.path == "/items" for e in list_item_entries)


def test_description_entry_path_for_list_fixed_includes_index():
    aligner = ObjectAligner({
        "type": "array",
        "items": {"type": "integer", "score": "exact"},
    })
    dr = aligner.describe([1, 2, 3], [1, 9, 3])
    item_entries = [e for e in dr.entries if e.match_kind == "item"]
    paths = [e.path for e in item_entries]
    assert "/0" in paths
    assert "/1" in paths
    assert "/2" in paths


def test_describe_invalid_pred_renders_validation_error_text():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner(schema)
    result = aligner.describe({"name": "A", "age": 1}, {"name": "A"})
    assert result.score == 0.0
    assert result.entries == ()
    assert "JSON Schema validation failed" in result.text


# -----------------------------------------------------------------------------
# Ref leaves: pred-space output, no gold-space id leakage
# -----------------------------------------------------------------------------

_REF_SCHEMA = {
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


def test_describe_ref_match_omits_gold_id():
    """When pred and gold refs align under the bijection, the describe row
    for the ref names only the pred-space id. Uses two edges so we can
    introduce a mismatch elsewhere to force describe to walk the full tree
    (perfect overall match short-circuits the walker).
    """
    aligner = ObjectAligner(_REF_SCHEMA)
    gold = {
        "nodes": [{"id": 0, "color": "blue"}, {"id": 1, "color": "green"}],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 0},
        ],
    }
    pred = {
        "nodes": [{"id": 10, "color": "blue"}, {"id": 11, "color": "green"}],
        "edges": [
            {"source": 10, "target": 11},  # matches gold edge 0
            {"source": 10, "target": 10},  # mismatch — forces full walk
        ],
    }
    dr = aligner.describe(gold, pred)
    matching_refs = [e for e in dr.entries
                     if e.match_kind == "ref" and e.outcome == "match"]
    assert matching_refs  # at least one ref matched
    for e in matching_refs:
        # The match-row template uses only `{pred}`; gold-space ids ("0" /
        # "1") must not leak in. (Pred-space ids "10" / "11" may appear.)
        assert "\"0\"" not in e.text
        assert "\"1\"" not in e.text


def test_describe_ref_mismatch_uses_pred_space_value_not_gold():
    aligner = ObjectAligner(_REF_SCHEMA)
    gold = {
        "nodes": [{"id": 0, "color": "blue"}, {"id": 1, "color": "green"}],
        "edges": [{"source": 0, "target": 1}],
    }
    # Pred has the right structure but the edge target points to the wrong
    # node (pred-10 = blue ≡ gold-0, pred-11 = green ≡ gold-1; edge target=10
    # is the blue one, but the bijection-mapped gold target ≡ pred-11).
    pred = {
        "nodes": [{"id": 10, "color": "blue"}, {"id": 11, "color": "green"}],
        "edges": [{"source": 10, "target": 10}],
    }
    dr = aligner.describe(gold, pred)
    mismatch = next(e for e in dr.entries
                    if e.match_kind == "ref" and e.outcome == "mismatch")
    assert "11" in mismatch.text  # pred-space replacement value
    assert "should be" in mismatch.text
    # Gold-space ids ("0", "1") must not leak into describe.
    assert "\"0\"" not in mismatch.text
    assert "\"1\"" not in mismatch.text


def test_describe_ref_no_target_outcome_for_dangling_gold_ref():
    aligner = ObjectAligner(_REF_SCHEMA)
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
    dr = aligner.describe(gold, pred)
    no_target = [e for e in dr.entries
                 if e.match_kind == "ref" and e.outcome == "no_target"]
    assert len(no_target) == 1
    assert "cannot be resolved" in no_target[0].text
    # Gold-space id ("9") never appears in user-facing describe text.
    assert "9" not in no_target[0].text


# -----------------------------------------------------------------------------
# Sanity: dataclasses are frozen
# -----------------------------------------------------------------------------

def test_description_entry_is_frozen():
    e = DescriptionEntry(path="/x", depth=0, match_kind="item", outcome="match", score=1.0, text="x")
    with pytest.raises(Exception):  # FrozenInstanceError
        e.score = 0.0


def test_description_result_is_frozen():
    dr = DescriptionResult(score=1.0, text="x", entries=(), style="default")
    with pytest.raises(Exception):
        dr.score = 0.0

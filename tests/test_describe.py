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

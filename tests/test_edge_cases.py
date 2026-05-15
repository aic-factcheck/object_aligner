import pytest

from object_aligner import ObjectAligner


def test_identical_primitives_lists_and_dicts_score_one():
    assert ObjectAligner({"type": "string", "score": "exact"}).metric("x", "x")["score"] == 1.0
    assert ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "fixed"}).metric(["x"], ["x"])["score"] == 1.0
    assert ObjectAligner({"type": "object", "properties": {"x": {"type": "string", "score": "exact"}}, "keyScore": "exact"},
    ).metric({"x": "y"}, {"x": "y"})["score"] == 1.0


def test_completely_disjoint_primitives_lists_and_dicts_score_zero():
    assert ObjectAligner({"type": "string", "score": "exact"}).metric("x", "y")["score"] == 0.0
    assert ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "align"}).metric(["x"], ["y"])["score"] == 0.0
    assert ObjectAligner({"type": "object", "properties": {"a": {"type": "integer", "score": "exact"}}, "keyScore": "exact"},
    ).align({"a": 1}, {"b": 2}, skip_validation=True).score == 0.0


def test_single_element_lists_and_single_key_dicts():
    fixed = ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "fixed"})
    reorder = ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "align"})
    exact_dict = ObjectAligner({"type": "object", "properties": {"x": {"type": "integer"}}, "keyScore": "exact"})
    fuzzy_dict = ObjectAligner({"type": "object", "properties": {"x": {"type": "integer"}}, "keyScore": "jaro", "keyThreshold": 0.0})

    assert fixed.metric(["a"], ["b"])["score"] == 0.0
    assert reorder.metric(["a"], ["b"])["score"] == 0.0
    assert exact_dict.metric({"x": 1}, {"x": 1})["score"] == 1.0
    assert fuzzy_dict.align({"x": 1}, {"xx": 1}, skip_validation=True).score > 0.0


def test_empty_unicode_and_emoji_strings():
    exact = ObjectAligner({"type": "string", "score": "exact"})
    jaro = ObjectAligner({"type": "string", "score": "jaro"})
    emoji = ObjectAligner({"type": "object", "properties": {"name": {"type": "string", "score": "exact"}}, "keyScore": "exact"})

    assert exact.metric("", "")["score"] == 1.0
    assert jaro.metric("", "abc")["score"] == 0.0
    assert emoji.metric({"name": "🔥"}, {"name": "🔥"})["score"] == 1.0
    # Default keyImportance=0: score reflects only the (mismatched) value.
    assert emoji.metric({"name": "🔥"}, {"name": "❄️"})["score"] == 0.0


def test_float_precision_mixed_number_types_and_integer_validation():
    number = ObjectAligner({"type": "number", "score": "invdiff"})
    integer = ObjectAligner({"type": "integer", "score": "invdiff"})

    assert number.metric(1.0000001, 1.0)["score"] > 0.999999
    with pytest.raises(TypeError):
        number.metric(1, 1.0)
    assert integer.metric(1, 1.5)["score"] == 0.0


def test_large_homogeneous_lists_under_fixed_and_reorder():
    fixed = ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "fixed"})
    reorder = ObjectAligner({"type": "array", "items": {"type": "string", "score": "exact"}, "order": "align"})
    values = ["x"] * 50

    assert fixed.metric(values, values)["score"] == 1.0
    assert reorder.metric(values, values)["score"] == 1.0


def test_reorder_list_of_objects_respects_nested_value_weights():
    aligner = ObjectAligner({
            "type": "array",
            "order": "align",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "score": "exact", "valueWeight": 1.0},
                    "score": {"type": "integer", "score": "exact", "valueWeight": 10.0},
                },
                "keyScore": "exact",
                "keyImportance": 0.0,
                "valueImportance": 1.0,
            },
        },
    )

    result = aligner.metric(
        [{"name": "A", "score": 10}],
        [{"name": "A", "score": 9}],
    )
    assert result["score"] == pytest.approx(1 / 11)


def test_large_fuzzy_dict_alignment_handles_many_keys():
    properties = {f"field_{i:02d}": {"type": "integer", "score": "exact"} for i in range(20)}
    gold = {key: i for i, key in enumerate(properties)}
    pred = {key.replace("field", "filed"): value for key, value in gold.items()}
    aligner = ObjectAligner({
            "type": "object",
            "properties": properties,
            "keyScore": "jaro",
            "keyThreshold": 0.5,
            "keyImportance": 1.0,
            "valueImportance": 1.0,
        },
    )

    assert aligner.align(gold, pred, skip_validation=True).score > 0.9


def test_empty_dict_alignment_scores_one():
    aligner = ObjectAligner({"type": "object", "properties": {}, "keyScore": "exact"})
    assert aligner.metric({}, {})["score"] == 1.0

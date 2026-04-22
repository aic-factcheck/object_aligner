import pytest

from object_aligner import ObjectAligner


def test_list_of_dicts_students_example():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "score": "jaro", "valueWeight": 1.0},
                "score": {"type": "integer", "score": "invdiff", "valueWeight": 1.0},
            },
            "keyScore": "exact",
            "keyImportance": 0.0,
            "valueImportance": 1.0,
        },
        "order": "align",
    }
    aligner = ObjectAligner(schema)

    gold = [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 82}]
    pred = [{"name": "Alice", "score": 93}, {"name": "Bobby", "score": 82}]

    assert aligner.metric(gold, pred)["score"] == pytest.approx(0.8)


def test_dict_of_lists_scores_each_property_independently():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string", "score": "exact"},
                "order": "align",
                "valueWeight": 1.0,
            },
            "numbers": {
                "type": "array",
                "items": {"type": "integer", "score": "exact"},
                "order": "fixed",
                "valueWeight": 1.0,
            },
        },
        "keyScore": "exact",
        "keyImportance": 0.0,
        "valueImportance": 1.0,
    }
    aligner = ObjectAligner(schema)

    score = aligner.metric(
        {"tags": ["a", "b"], "numbers": [1, 2]},
        {"tags": ["b", "a"], "numbers": [1, 3]},
    )["score"]
    assert score == pytest.approx(2 / 3)


def test_mixed_depth_structure_drops_only_at_deep_leaf():
    schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "leaf": {"type": "string", "score": "exact"}
                                },
                                "keyScore": "exact",
                                "keyImportance": 0.0,
                                "valueImportance": 1.0,
                            },
                            "order": "fixed",
                        }
                    },
                    "keyScore": "exact",
                    "keyImportance": 0.0,
                    "valueImportance": 1.0,
                },
            }
        },
        "keyScore": "exact",
        "keyImportance": 0.0,
        "valueImportance": 1.0,
    }
    aligner = ObjectAligner(schema)

    gold = {"level1": [{"level2": [{"leaf": "x"}, {"leaf": "y"}]}]}
    pred_same = {"level1": [{"level2": [{"leaf": "x"}, {"leaf": "y"}]}]}
    pred_typo = {"level1": [{"level2": [{"leaf": "x"}, {"leaf": "z"}]}]}

    assert aligner.metric(gold, pred_same)["score"] == 1.0
    assert aligner.metric(gold, pred_typo)["score"] == pytest.approx(1 / 3)


def test_prefix_items_inside_nested_reordered_list():
    schema = {
        "type": "array",
        "order": "align",
        "items": {
            "type": "array",
            "prefixItems": [
                {"type": "string", "score": "exact"},
                {"type": "integer", "score": "exact"},
            ],
            "prefixWeights": [1.0, 1.0],
            "items": {"type": "string", "score": "exact"},
            "prefixImportance": 2.0,
            "restImportance": 1.0,
        },
    }
    aligner = ObjectAligner(schema)

    gold = [["bus", 3, "ship"], ["car", 5, "airplane"]]
    pred = [["car", 5, "plane"], ["bus", 3]]

    score = aligner.metric(gold, pred)["score"]
    assert 0.55 < score < 0.85


def test_completely_off_nested_structure_scores_zero_with_exact_matching():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "labels": {
                            "type": "array",
                            "items": {"type": "string", "score": "exact"},
                            "order": "fixed",
                        }
                    },
                    "keyScore": "exact",
                    "keyImportance": 0.0,
                    "valueImportance": 1.0,
                },
                "order": "align",
            }
        },
        "keyScore": "exact",
        "keyImportance": 0.0,
        "valueImportance": 1.0,
    }
    aligner = ObjectAligner(schema)

    gold = {"items": [{"labels": ["x", "y"]}]}
    pred = {"items": [{"labels": ["a", "b"]}]}
    assert aligner.metric(gold, pred)["score"] == 0.0

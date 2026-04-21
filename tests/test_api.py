import pytest
from jsonschema import ValidationError

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchDict, MatchItem, MatchList


def test_get_name_returns_id():
    aligner = ObjectAligner("demo-id", {"type": "string"})
    assert aligner.get_name() == "demo-id"


@pytest.mark.parametrize(
    ("schema", "gold", "pred", "expected_type"),
    [
        ({"type": "string"}, "a", "a", MatchItem),
        ({"type": "array", "items": {"type": "string"}}, ["a"], ["a"], MatchList),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "keyScore": "exact"}, {"a": "x"}, {"a": "x"}, MatchDict),
    ],
)
def test_align_returns_match_type_for_schema(schema, gold, pred, expected_type):
    aligner = ObjectAligner("typed", schema)
    assert isinstance(aligner.align(gold, pred), expected_type)


def test_align_validates_by_default_and_skip_validation_bypasses_it():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner("person", schema)

    assert aligner.align({"name": "Alice", "age": 30}, {"name": "Alice", "age": 30}).score == 1.0
    with pytest.raises(ValidationError):
        aligner.align({"name": "Alice", "age": 30}, {"name": "Alice"})

    skipped = aligner.align({"name": "Alice", "age": 30}, {"name": "Alice"}, skip_validation=True)
    assert isinstance(skipped, MatchDict)
    assert skipped.score == pytest.approx(0.5)


def test_metric_returns_score_and_reasoning():
    aligner = ObjectAligner("metric", {"type": "string", "score": "jaro"})
    result = aligner.metric("hello", "hallo")

    assert set(result) == {"score", "reasoning"}
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["reasoning"], str)


def test_metric_invalid_gold_raises_and_invalid_pred_returns_zero_score():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner("person", schema)

    with pytest.raises(ValidationError):
        aligner.metric({"name": "Alice"}, {"name": "Alice", "age": 30})

    result = aligner.metric({"name": "Alice", "age": 30}, {"name": "Alice"})
    assert result["score"] == 0.0
    assert 'JSON Schema validation failed' in result["reasoning"]


def test_metric_reasoning_for_perfect_and_imperfect_matches():
    perfect = ObjectAligner("perfect", {"type": "string", "score": "exact"})
    imperfect = ObjectAligner("imperfect", {"type": "string", "score": "exact"})

    assert perfect.metric("hello", "hello")["reasoning"] == "The predicted output perfectly matches the gold."
    assert imperfect.metric("hello", "world")["reasoning"].startswith(
        "The predicted output scores overall"
    )


def test_list_reasoning_mentions_missing_and_excessive_items():
    aligner = ObjectAligner(
        "list-reasoning",
        {
            "type": "array",
            "items": {"type": "integer", "score": "exact"},
            "order": "fixed",
        },
    )
    reasoning = aligner.metric([1, 2, 4], [2, 3])["reasoning"]

    assert 'misses the "1" list item' in reasoning
    assert 'list item "3" is excessive' in reasoning


def test_dict_reasoning_contains_key_and_value_lines():
    aligner = ObjectAligner(
        "dict-reasoning",
        {
            "type": "object",
            "properties": {
                "weight": {"type": "integer", "valueWeight": 1.0},
                "name": {"type": "string", "score": "jaro", "valueWeight": 1.0},
                "age": {"type": "integer", "valueWeight": 1.0},
            },
            "keyScore": "jaro",
            "keyThreshold": 0.5,
            "keyImportance": 1.0,
            "valueImportance": 1.0,
        },
    )
    reasoning = aligner.metric(
        {"weight": 90, "name": "John", "age": 24},
        {"name": "Johny", "ages": 23, "title": "Mr."},
    )["reasoning"]

    assert 'KEY = ' in reasoning
    assert 'VALUE = ' in reasoning


def test_metric_validation_edge_cases_cover_nested_type_and_item_constraints():
    nested = ObjectAligner(
        "nested-validation",
        {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                    "required": ["age"],
                }
            },
            "required": ["person"],
            "keyScore": "exact",
        },
    )
    bounded_list = ObjectAligner(
        "bounded-list",
        {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 3,
        },
    )

    nested_result = nested.metric({"person": {"age": 1}}, {"person": {"age": "bad"}})
    assert nested_result["score"] == 0.0
    assert '/person/age' in nested_result["reasoning"]

    assert bounded_list.metric([1, 2], [1])["score"] == 0.0
    assert bounded_list.metric([1, 2], [1, 2, 3, 4])["score"] == 0.0

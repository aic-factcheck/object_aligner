import pytest
from jsonschema import ValidationError

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchDict, MatchItem, MatchList


@pytest.mark.parametrize(
    ("schema", "gold", "pred", "expected_type"),
    [
        ({"type": "string"}, "a", "a", MatchItem),
        ({"type": "array", "items": {"type": "string"}}, ["a"], ["a"], MatchList),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "keyScore": "exact"}, {"a": "x"}, {"a": "x"}, MatchDict),
    ],
)
def test_align_returns_match_type_for_schema(schema, gold, pred, expected_type):
    aligner = ObjectAligner(schema)
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
    aligner = ObjectAligner(schema)

    assert aligner.align({"name": "Alice", "age": 30}, {"name": "Alice", "age": 30}).score == 1.0
    with pytest.raises(ValidationError):
        aligner.align({"name": "Alice", "age": 30}, {"name": "Alice"})

    skipped = aligner.align({"name": "Alice", "age": 30}, {"name": "Alice"}, skip_validation=True)
    assert isinstance(skipped, MatchDict)
    assert skipped.score == pytest.approx(0.5)



def test_metric_returns_score_only_by_default():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})
    result = aligner.metric("hello", "hallo")

    assert set(result) == {"score"}
    assert 0.0 <= result["score"] <= 1.0



def test_metric_returns_reasoning_when_enabled_in_constructor():
    aligner = ObjectAligner({"type": "string", "score": "jaro"}, generate_reasoning=True)
    result = aligner.metric("hello", "hallo")

    assert set(result) == {"score", "reasoning"}
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["reasoning"], str)



def test_metric_generate_reasoning_true_overrides_constructor_false():
    aligner = ObjectAligner({"type": "string", "score": "exact"})

    result = aligner.metric("hello", "world", generate_reasoning=True)

    assert set(result) == {"score", "reasoning"}
    assert result["reasoning"].startswith("The predicted output scores overall")



def test_metric_generate_reasoning_false_overrides_constructor_true():
    aligner = ObjectAligner({"type": "string", "score": "exact"}, generate_reasoning=True)

    result = aligner.metric("hello", "world", generate_reasoning=False)

    assert result == {"score": 0.0}



def test_metric_invalid_gold_raises():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner(schema)

    with pytest.raises(ValidationError):
        aligner.metric({"name": "Alice"}, {"name": "Alice", "age": 30})



def test_metric_invalid_pred_returns_score_only_when_reasoning_disabled():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner(schema)

    result = aligner.metric({"name": "Alice", "age": 30}, {"name": "Alice"})

    assert result == {"score": 0.0}



def test_metric_invalid_pred_returns_reasoning_when_enabled():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner(schema, generate_reasoning=True)

    result = aligner.metric({"name": "Alice", "age": 30}, {"name": "Alice"})

    assert result["score"] == 0.0
    assert 'JSON Schema validation failed' in result["reasoning"]



def test_metric_reasoning_for_perfect_and_imperfect_matches():
    perfect = ObjectAligner({"type": "string", "score": "exact"}, generate_reasoning=True)
    imperfect = ObjectAligner({"type": "string", "score": "exact"}, generate_reasoning=True)

    assert perfect.metric("hello", "hello")["reasoning"] == "The predicted output perfectly matches the gold."
    assert imperfect.metric("hello", "world")["reasoning"].startswith(
        "The predicted output scores overall"
    )



def test_list_reasoning_mentions_missing_and_excessive_items():
    aligner = ObjectAligner(
        {
            "type": "array",
            "items": {"type": "integer", "score": "exact"},
            "order": "fixed",
        },
        generate_reasoning=True,
    )
    reasoning = aligner.metric([1, 2, 4], [2, 3])["reasoning"]

    assert 'misses the "1" list item' in reasoning
    assert 'list item "3" is excessive' in reasoning



def test_dict_reasoning_accumulates_multiple_key_and_value_lines():
    aligner = ObjectAligner(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "score": "jaro", "valueWeight": 1.0},
                "age": {"type": "integer", "valueWeight": 1.0},
            },
            "keyScore": "jaro",
            "keyThreshold": 0.5,
            "keyImportance": 1.0,
            "valueImportance": 1.0,
        },
        generate_reasoning=True,
    )
    reasoning = aligner.metric(
        {"name": "John", "age": 24},
        {"nmae": "Johny", "ages": 23},
    )["reasoning"]

    assert reasoning.count('KEY = ') >= 2
    assert reasoning.count('VALUE = ') >= 2
    assert 'predicted key "nmae"' in reasoning
    assert 'predicted key "ages"' in reasoning



def test_reasoning_template_override_changes_output():
    aligner = ObjectAligner(
        {"type": "string", "score": "exact"},
        generate_reasoning=True,
        reasoning_templates={"metric.perfect": "Perfect match."},
    )

    result = aligner.metric("hello", "hello")

    assert result == {"score": 1.0, "reasoning": "Perfect match."}



def test_reasoning_template_partial_override_preserves_other_defaults():
    aligner = ObjectAligner(
        {"type": "string", "score": "exact"},
        generate_reasoning=True,
        reasoning_templates={"metric.perfect": "Perfect match."},
    )

    result = aligner.metric("hello", "world")

    assert result["reasoning"].startswith("The predicted output scores overall")



def test_reasoning_template_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown reasoning template keys"):
        ObjectAligner(
            {"type": "string", "score": "exact"},
            reasoning_templates={"metric.typo": "nope"},
        )



def test_nested_reasoning_still_renders_recursively():
    aligner = ObjectAligner(
        {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "items": {"type": "string", "score": "exact"},
                    "order": "fixed",
                }
            },
            "required": ["skills"],
            "keyScore": "exact",
        },
        generate_reasoning=True,
    )

    reasoning = aligner.metric({"skills": ["python", "sql"]}, {"skills": ["python", "rust"]})["reasoning"]

    assert 'KEY = The predicted key "skills" exactly matches the gold.' in reasoning
    assert 'The predicted list scores' in reasoning
    assert 'misses the "sql" list item' in reasoning
    assert 'list item "rust" is excessive' in reasoning



def test_metric_validation_edge_cases_cover_nested_type_and_item_constraints():
    nested = ObjectAligner(
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
        generate_reasoning=True,
    )
    bounded_list = ObjectAligner(
        {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 3,
        }
    )

    nested_result = nested.metric({"person": {"age": 1}}, {"person": {"age": "bad"}})
    assert nested_result["score"] == 0.0
    assert '/person/age' in nested_result["reasoning"]

    assert bounded_list.metric([1, 2], [1]) == {"score": 0.0}
    assert bounded_list.metric([1, 2], [1, 2, 3, 4]) == {"score": 0.0}

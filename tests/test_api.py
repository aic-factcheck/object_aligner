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



def test_custom_metrics_must_be_a_mapping():
    with pytest.raises(TypeError, match="custom_metrics must be a mapping"):
        ObjectAligner({"type": "string"}, custom_metrics=[("string", {})])



def test_custom_metrics_reject_unsupported_type_keys():
    with pytest.raises(ValueError, match="Unsupported custom metric types"):
        ObjectAligner({"type": "string"}, custom_metrics={"boolean": {"x": lambda gold, pred: 1.0}})



def test_custom_metrics_per_type_value_must_be_a_mapping():
    with pytest.raises(TypeError, match=r'custom_metrics\["string"\] must be a mapping'):
        ObjectAligner({"type": "string"}, custom_metrics={"string": [lambda gold, pred: 1.0]})



def test_custom_metrics_metric_names_must_be_strings():
    with pytest.raises(TypeError, match='metric names must be strings'):
        ObjectAligner({"type": "string"}, custom_metrics={"string": {1: lambda gold, pred: 1.0}})



def test_custom_metrics_values_must_be_callable():
    with pytest.raises(TypeError, match='must be callable'):
        ObjectAligner({"type": "number"}, custom_metrics={"number": {"x": 123}})



def test_custom_metrics_cannot_shadow_builtin_names():
    with pytest.raises(ValueError, match='collide with built-in metrics'):
        ObjectAligner({"type": "string"}, custom_metrics={"string": {"jaro": lambda gold, pred: 1.0}})


def test_match_and_metric_scores_use_builtin_python_float_types():
    aligner = ObjectAligner(
        {
            "type": "array",
            "items": {"type": "integer", "score": "exact"},
            "order": "fixed",
        },
        generate_reasoning=True,
    )

    match = aligner.align([42, 7, 13], [99, 7, 13])
    result = aligner.metric([42, 7, 13], [99, 7, 13])

    assert isinstance(match.score, float)
    assert isinstance(result["score"], float)
    assert match.score == 0.5
    assert result["score"] == 0.5


def test_metric_debug_returns_python_native_nested_alignment_data():
    aligner = ObjectAligner(
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "score": "jaro"},
                    "age": {"type": "integer", "score": "invdiff"},
                },
                "keyScore": "exact",
            },
            "order": "fixed",
        }
    )

    result = aligner.metric(
        [{"name": "Alice", "age": 30}],
        [{"name": "Alicia", "age": 29}],
        debug=True,
    )

    assert set(result) == {"score", "debug"}
    assert isinstance(result["score"], float)
    assert result["debug"]["kind"] == "list"
    assert isinstance(result["debug"]["score"], float)

    child = result["debug"]["children"][0]
    assert child["kind"] == "dict"
    assert isinstance(child["score"], float)

    key_debug = child["children"][0]["key"]
    value_debug = child["children"][0]["value"]
    assert key_debug["kind"] == "item"
    assert value_debug["score"] == pytest.approx(value_debug["score"])
    assert isinstance(key_debug["score"], float)
    assert isinstance(value_debug["score"], float)



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

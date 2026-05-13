import pytest

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchDict, MatchItem


def test_exact_key_matching_basic_cases():
    aligner = ObjectAligner({
            "type": "object",
            "properties": {
                "a": {"type": "integer", "score": "exact"},
                "b": {"type": "integer", "score": "exact"},
            },
            "keyScore": "exact",
        },
    )

    assert aligner.align({"a": 1, "b": 2}, {"a": 1, "b": 2}).score == 1.0
    assert aligner.align({"a": 1, "b": 2}, {"a": 1, "b": 99}).score == pytest.approx(0.75)
    assert aligner.align({"a": 1}, {"a": 1, "b": 2}, skip_validation=True).score == pytest.approx(0.5)
    assert aligner.align({"a": 1, "b": 2}, {"b": 2, "a": 1}).score == 1.0


def test_fuzzy_key_matching_handles_typos_and_documented_example():
    fuzzy = ObjectAligner({
            "type": "object",
            "properties": {
                "colour": {"type": "string", "score": "exact"},
            },
            "keyScore": "jaro",
            "keyThreshold": 0.5,
        },
    )
    doc = ObjectAligner({
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

    typo_match = fuzzy.align({"colour": "red"}, {"color": "red"}, skip_validation=True)
    assert typo_match.score > 0.9

    doc_match = doc.align(
        {"weight": 90, "name": "John", "age": 24},
        {"name": "Johny", "ages": 23, "title": "Mr."},
        skip_validation=True,
    )
    assert doc_match.score == pytest.approx(0.41874999999999996)


def test_key_threshold_can_prevent_or_allow_fuzzy_matches():
    high = ObjectAligner({
            "type": "object",
            "properties": {"colour": {"type": "string", "score": "exact"}},
            "keyScore": "jaro",
            "keyThreshold": 0.7,
        },
    )
    low = ObjectAligner({
            "type": "object",
            "properties": {"colour": {"type": "string", "score": "exact"}},
            "keyScore": "jaro",
            "keyThreshold": 0.2,
        },
    )

    high_match = high.align({"colour": "red"}, {"clr": "red"}, skip_validation=True)
    low_match = low.align({"colour": "red"}, {"clr": "red"}, skip_validation=True)

    assert high_match.score == 0.0
    assert low_match.score > 0.0


def test_key_and_value_importance_follow_documented_formula():
    base_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "score": "exact", "valueWeight": 3.0},
            "name": {"type": "string", "score": "exact", "valueWeight": 1.0},
        },
        "keyScore": "exact",
    }
    gold = {"id": 1, "name": "Alice"}
    pred = {"id": 2, "name": "Alice"}

    values_only = ObjectAligner({**base_schema, "keyImportance": 0.0, "valueImportance": 1.0})
    keys_only = ObjectAligner({**base_schema, "keyImportance": 1.0, "valueImportance": 0.0})
    balanced = ObjectAligner({**base_schema, "keyImportance": 1.0, "valueImportance": 1.0})
    unbalanced = ObjectAligner({**base_schema, "keyImportance": 1.0, "valueImportance": 3.0})

    assert values_only.align(gold, pred).score == pytest.approx(0.25)
    assert keys_only.align(gold, pred).score == pytest.approx(1.0)
    assert balanced.align(gold, pred).score == pytest.approx(0.625)
    assert unbalanced.align(gold, pred).score == pytest.approx(0.4375)


def test_empty_string_dict_key_is_not_dropped_exact():
    aligner = ObjectAligner({
        "type": "object",
        "properties": {"": {"type": "integer", "score": "exact"},
                       "a": {"type": "integer", "score": "exact"}},
        "keyScore": "exact",
    })
    match = aligner.align({"": 1, "a": 2}, {"a": 2, "b": 3}, skip_validation=True)
    seen_gold_keys = [k.gold for k in match.children]
    seen_pred_keys = [k.pred for k in match.children]
    assert "" in seen_gold_keys  # falsy gold key not silently dropped
    assert "a" in seen_gold_keys
    assert "b" in seen_pred_keys


def test_empty_string_dict_key_is_not_dropped_jaro():
    aligner = ObjectAligner({
        "type": "object",
        "properties": {"": {"type": "integer", "score": "exact"},
                       "a": {"type": "integer", "score": "exact"}},
        "keyScore": "jaro",
        "keyThreshold": 0.99,  # force "" vs "b" to fall below threshold → score 0
    })
    match = aligner.align({"": 1, "a": 2}, {"a": 2, "b": 3}, skip_validation=True)
    seen_gold_keys = [k.gold for k in match.children]
    assert "" in seen_gold_keys


def test_key_value_importance_sum_zero_raises_at_construction():
    with pytest.raises(ValueError, match="keyImportance and valueImportance"):
        ObjectAligner({
            "type": "object",
            "properties": {"a": {"type": "integer", "score": "exact"}},
            "keyScore": "exact",
            "keyImportance": 0.0,
            "valueImportance": 0.0,
        })


def test_value_weights_sum_zero_raises_at_construction():
    with pytest.raises(ValueError, match="valueWeights"):
        ObjectAligner({
            "type": "object",
            "properties": {
                "a": {"type": "string", "valueWeight": 0.0},
                "b": {"type": "string", "valueWeight": 0.0},
            },
            "keyScore": "exact",
        })


def test_value_weights_affect_value_score():
    aligner = ObjectAligner({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "score": "exact", "valueWeight": 3.0},
                "name": {"type": "string", "score": "exact", "valueWeight": 1.0},
            },
            "keyScore": "exact",
            "keyImportance": 0.0,
            "valueImportance": 1.0,
        },
    )

    assert aligner.align({"id": 1, "name": "Alice"}, {"id": 2, "name": "Alice"}).score == pytest.approx(0.25)


def test_missing_and_extra_keys_produce_zero_scored_value_matches():
    aligner = ObjectAligner({
            "type": "object",
            "properties": {
                "a": {"type": "integer", "score": "exact"},
                "b": {"type": "integer", "score": "exact"},
            },
            "keyScore": "exact",
        },
    )

    match = aligner.align({"a": 1}, {"b": 2}, skip_validation=True)
    assert isinstance(match, MatchDict)
    assert match.score == 0.0

    items = list(match.children.items())
    assert items[0][0] == MatchItem(score=0.0, gold=None, pred="b")
    assert items[0][1] == MatchItem(score=0.0, gold=None, pred=2)
    assert items[1][0] == MatchItem(score=0.0, gold="a", pred=None)
    assert items[1][1] == MatchItem(score=0.0, gold=1, pred=None)


@pytest.mark.parametrize(
    ("gold", "pred"),
    [
        ({"a": 1}, {"a": "one"}),
        ({"colour": "red"}, {"color": 1}),
    ],
)
def test_type_mismatch_softens_to_zero_under_skip_validation(gold, pred):
    properties = {next(iter(gold.keys())): {"type": "integer" if isinstance(next(iter(gold.values())), int) else "string"}}
    if "colour" in gold:
        properties = {"colour": {"type": "string"}}
        schema = {
            "type": "object",
            "properties": properties,
            "keyScore": "jaro",
            "keyThreshold": 0.5,
        }
    else:
        schema = {
            "type": "object",
            "properties": properties,
            "keyScore": "exact",
        }

    aligner = ObjectAligner(schema)
    # With skip_validation=True the caller opted into looser semantics;
    # type-mismatched values score 0 rather than raising. The dict's
    # overall score may still be non-zero if the key match contributes.
    match = aligner.align(gold, pred, skip_validation=True)
    value_matches = list(match.children.values())
    assert len(value_matches) == 1
    assert value_matches[0].score == 0.0

    # Without skip_validation the schema-side validation catches the bad
    # pred before alignment reaches the type-mismatch branch.
    from jsonschema import ValidationError
    with pytest.raises((ValidationError, TypeError)):
        aligner.align(gold, pred)

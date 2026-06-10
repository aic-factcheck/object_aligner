import pytest

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchItem, similarity_string_jaro


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (True, True, 1.0),
        (False, False, 1.0),
        (True, False, 0.0),
        (False, True, 0.0),
    ],
)
def test_boolean_alignment_is_exact(gold, pred, expected):
    aligner = ObjectAligner({"type": "boolean", "score": "ignored", "threshold": 0.99})
    match = aligner.align(gold, pred)

    assert isinstance(match, MatchItem)
    assert match.score == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (42, 42, 1.0),
        (42, 43, 0.0),
        (3.14, 3.14, 1.0),
        (3.14, 2.71, 0.0),
    ],
)
def test_number_exact_mode(gold, pred, expected):
    aligner = ObjectAligner({"type": "number", "score": "exact"})
    assert aligner.align(gold, pred).score == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (50, 50, 1.0),
        (50, 51, 0.5),
        (50, 52, 1 / 3),
        (0, 5, 1 / 6),
        (-5, -3, 1 / 3),
        (1.0, 1.5, 1 / 1.5),
    ],
)
def test_number_invdiff_mode(gold, pred, expected):
    aligner = ObjectAligner({"type": "number", "score": "invdiff"})
    assert aligner.align(gold, pred).score == pytest.approx(expected)


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (50, 50, 1.0),
        (0, 0, 1.0),          # equal values short-circuit before the division
        (0.0, 0.0, 1.0),
        (2020, 2021, 1.0 - 1 / 2021),
        (100, 110, 1.0 - 10 / 110),
        (1, -1, 0.0),         # |a-b| >= max magnitude clamps to 0
        (0, 5, 0.0),
        (-50, -55, 1.0 - 5 / 55),
        (1.0, 1.5, 1.0 - 0.5 / 1.5),
    ],
)
def test_number_relative_mode(gold, pred, expected):
    aligner = ObjectAligner({"type": "number", "score": "relative"})
    assert aligner.align(gold, pred).score == pytest.approx(expected)


def test_number_relative_is_scale_invariant():
    aligner = ObjectAligner({"type": "number", "score": "relative"})
    base = aligner.align(100.0, 110.0).score
    assert aligner.align(100_000.0, 110_000.0).score == pytest.approx(base)
    assert aligner.align(0.001, 0.0011).score == pytest.approx(base)


def test_integer_relative_mode_supported():
    aligner = ObjectAligner({"type": "integer", "score": "relative"})
    assert aligner.align(1_000_000, 1_000_010).score == pytest.approx(1.0 - 10 / 1_000_010)


def test_custom_number_metric_named_relative_collides_with_builtin():
    with pytest.raises(ValueError, match="collide with built-in metrics"):
        ObjectAligner(
            {"type": "number", "score": "relative"},
            custom_metrics={"number": {"relative": lambda g, p: 1.0}},
        )


def test_number_threshold_applies_after_scoring():
    invdiff = ObjectAligner({"type": "integer", "score": "invdiff", "threshold": 0.5})
    exact = ObjectAligner({"type": "integer", "score": "exact", "threshold": 0.5})

    assert invdiff.align(50, 51).score == 0.5
    assert invdiff.align(50, 52).score == 0.0
    assert exact.align(42, 42).score == 1.0
    assert exact.align(42, 43).score == 0.0


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("hello", "hello", 1.0),
        ("hello", "hallo", 0.0),
        ("", "", 1.0),
        ("a", "", 0.0),
    ],
)
def test_string_exact_mode(gold, pred, expected):
    aligner = ObjectAligner({"type": "string", "score": "exact"})
    assert aligner.align(gold, pred).score == expected


def test_string_jaro_mode_covers_classic_cases():
    aligner = ObjectAligner({"type": "string", "score": "jaro"})

    assert aligner.align("hello", "hello").score == 1.0
    assert aligner.align("hello", "hallo").score > 0.8
    assert aligner.align("MARTHA", "MARHTA").score == pytest.approx(0.9444444444444445)
    assert aligner.align("DWAYNE", "DUANE").score == pytest.approx(0.8222222222222223)
    assert aligner.align("café", "cafè").score > 0.8
    assert aligner.align("", "abc").score == 0.0
    assert aligner.align("abc", "").score == 0.0


@pytest.mark.parametrize(
    "metric_name",
    [
        "jaro_winkler",
        "levenshtein",
        "damerau_levenshtein",
        "osa",
        "indel",
        "lcsseq",
    ],
)
def test_new_builtin_string_metrics_accept_representative_inputs(metric_name):
    aligner = ObjectAligner({"type": "string", "score": metric_name})

    assert aligner.align("hello", "hello").score == 1.0
    assert aligner.align("abc", "").score == 0.0
    assert 0.0 <= aligner.align("kitten", "sitting").score <= 1.0


def test_transposition_metrics_score_transpositions_better_than_plain_levenshtein():
    levenshtein = ObjectAligner({"type": "string", "score": "levenshtein"})
    damerau = ObjectAligner({"type": "string", "score": "damerau_levenshtein"})
    osa = ObjectAligner({"type": "string", "score": "osa"})

    lev_score = levenshtein.align("abcd", "abdc").score
    assert damerau.align("abcd", "abdc").score > lev_score
    assert osa.align("abcd", "abdc").score > lev_score


def test_string_threshold_zeroes_out_below_cutoff_and_keeps_boundary():
    aligner = ObjectAligner({"type": "string", "score": "jaro", "threshold": 0.7})
    boundary = similarity_string_jaro("cat", "car")

    assert boundary > 0.7
    assert aligner.align("cat", "car").score == pytest.approx(boundary)
    assert aligner.align("cat", "dog").score == 0.0


def test_threshold_applies_to_nondefault_builtin_string_metrics():
    aligner = ObjectAligner({"type": "string", "score": "levenshtein", "threshold": 0.7})

    assert aligner.align("abcd", "abce").score == pytest.approx(0.75)
    assert aligner.align("abcd", "abdc").score == 0.0


def test_custom_string_metric_is_resolved_by_name_and_thresholded():
    def first_letter_bonus(gold, pred):
        return 0.9 if gold[:1] == pred[:1] else 0.4

    aligner = ObjectAligner(
        {"type": "string", "score": "first_letter_bonus", "threshold": 0.5},
        custom_metrics={"string": {"first_letter_bonus": first_letter_bonus}},
    )

    assert aligner.align("cat", "car").score == 0.9
    assert aligner.align("cat", "dog").score == 0.0


def test_custom_number_metric_is_resolved_by_name():
    def closish(gold, pred):
        return 0.8 if abs(gold - pred) <= 2 else 0.2

    aligner = ObjectAligner(
        {"type": "number", "score": "closish"},
        custom_metrics={"number": {"closish": closish}},
    )

    assert aligner.align(10, 12).score == 0.8
    assert aligner.align(10, 20).score == 0.2


def test_integer_schema_falls_back_to_number_custom_metric_and_can_be_overridden():
    def number_metric(gold, pred):
        return 0.6

    def integer_metric(gold, pred):
        return 0.9

    fallback = ObjectAligner(
        {"type": "integer", "score": "parityish"},
        custom_metrics={"number": {"parityish": number_metric}},
    )
    override = ObjectAligner(
        {"type": "integer", "score": "parityish"},
        custom_metrics={
            "number": {"parityish": number_metric},
            "integer": {"parityish": integer_metric},
        },
    )

    assert fallback.align(1, 2).score == 0.6
    assert override.align(1, 2).score == 0.9


@pytest.mark.parametrize(
    ("schema", "gold", "pred", "schema_type"),
    [
        ({"type": "string", "score": "semantic"}, "a", "a", "string"),
        ({"type": "number", "score": "gaussian"}, 1, 1, "number"),
    ],
)
def test_unknown_primitive_score_name_raises_clear_error(schema, gold, pred, schema_type):
    aligner = ObjectAligner(schema)

    with pytest.raises(ValueError, match=fr'Unsupported score ".*" for schema type "{schema_type}"'):
        aligner.align(gold, pred)


def test_invalid_custom_metric_return_value_outside_unit_interval_raises():
    aligner = ObjectAligner(
        {"type": "string", "score": "bad"},
        custom_metrics={"string": {"bad": lambda gold, pred: 1.5}},
    )

    with pytest.raises(ValueError, match=r'must return a score in \[0, 1\]'):
        aligner.align("a", "b")


def test_invalid_custom_metric_return_value_type_raises():
    aligner = ObjectAligner(
        {"type": "number", "score": "bad"},
        custom_metrics={"number": {"bad": lambda gold, pred: "nope"}},
    )

    with pytest.raises(TypeError, match=r'must return a real number in \[0, 1\]'):
        aligner.align(1, 2)


def test_boolean_true_is_treated_as_boolean_not_number():
    aligner = ObjectAligner({"type": "boolean"})
    match = aligner.align(True, True)

    assert isinstance(match, MatchItem)
    assert match.gold is True
    assert match.pred is True
    assert match.score == 1.0

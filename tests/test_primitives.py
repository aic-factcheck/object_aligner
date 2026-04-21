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
    aligner = ObjectAligner("bool-test", {"type": "boolean", "score": "ignored", "threshold": 0.99})
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
    aligner = ObjectAligner("num-exact", {"type": "number", "score": "exact"})
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
    aligner = ObjectAligner("num-invdiff", {"type": "number", "score": "invdiff"})
    assert aligner.align(gold, pred).score == pytest.approx(expected)


def test_number_threshold_applies_after_scoring():
    invdiff = ObjectAligner("num-threshold", {"type": "integer", "score": "invdiff", "threshold": 0.5})
    exact = ObjectAligner("num-threshold-exact", {"type": "integer", "score": "exact", "threshold": 0.5})

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
    aligner = ObjectAligner("str-exact", {"type": "string", "score": "exact"})
    assert aligner.align(gold, pred).score == expected


def test_string_jaro_mode_covers_classic_cases():
    aligner = ObjectAligner("str-jaro", {"type": "string", "score": "jaro"})

    assert aligner.align("hello", "hello").score == 1.0
    assert aligner.align("hello", "hallo").score > 0.8
    assert aligner.align("MARTHA", "MARHTA").score == pytest.approx(0.9444444444444445)
    assert aligner.align("DWAYNE", "DUANE").score == pytest.approx(0.8222222222222223)
    assert aligner.align("café", "cafè").score > 0.8
    assert aligner.align("", "abc").score == 0.0
    assert aligner.align("abc", "").score == 0.0


def test_string_threshold_zeroes_out_below_cutoff_and_keeps_boundary():
    aligner = ObjectAligner("str-threshold", {"type": "string", "score": "jaro", "threshold": 0.7})
    boundary = similarity_string_jaro("cat", "car")

    assert boundary > 0.7
    assert aligner.align("cat", "car").score == pytest.approx(boundary)
    assert aligner.align("cat", "dog").score == 0.0


def test_boolean_true_is_treated_as_boolean_not_number():
    aligner = ObjectAligner("bool", {"type": "boolean"})
    match = aligner.align(True, True)

    assert isinstance(match, MatchItem)
    assert match.gold is True
    assert match.pred is True
    assert match.score == 1.0

import pytest

from object_aligner import ObjectAligner
from object_aligner.object_aligner import MatchItem, MatchList


FIXED_INT_EXACT = {
    "type": "array",
    "items": {"type": "integer", "score": "exact"},
    "order": "fixed",
}

REORDER_STR_EXACT = {
    "type": "array",
    "items": {"type": "string", "score": "exact"},
    "order": "align",
}


def test_fixed_order_basic_cases():
    aligner = ObjectAligner("fixed", FIXED_INT_EXACT)

    assert aligner.align([1, 2, 3], [1, 2, 3]).score == 1.0
    assert aligner.align([1, 2, 3], [1, 3, 2]).score == pytest.approx(0.5)
    assert aligner.align([1, 2], [1, 2, 99]).score == pytest.approx(2 / 3)
    assert aligner.align([1, 2, 99], [1, 2]).score == pytest.approx(2 / 3)
    assert aligner.align([], []).score == 1.0
    assert aligner.align([], [1]).score == 0.0


def test_fixed_order_gap_behavior_matches_current_dp_alignment():
    aligner = ObjectAligner("fixed", FIXED_INT_EXACT)
    match = aligner.align([1, 2, 4], [2, 3])

    assert isinstance(match, MatchList)
    assert match.score == pytest.approx(0.25)
    assert match.children == [
        MatchItem(score=0.0, gold=1, pred=None),
        MatchItem(score=1.0, gold=2, pred=2),
        MatchItem(score=0.0, gold=4, pred=None),
        MatchItem(score=0.0, gold=None, pred=3),
    ]


def test_fixed_order_with_string_and_number_scoring():
    string_aligner = ObjectAligner(
        "fixed-jaro",
        {
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
            "order": "fixed",
        },
    )
    number_aligner = ObjectAligner(
        "fixed-invdiff",
        {
            "type": "array",
            "items": {"type": "number", "score": "invdiff"},
            "order": "fixed",
        },
    )

    assert string_aligner.align(["hello"], ["hallo"]).score > 0.8
    assert number_aligner.align([1, 2], [1, 3]).score == pytest.approx(0.75)


def test_reorder_alignment_basic_cases():
    aligner = ObjectAligner(
        "reorder",
        {
            "type": "array",
            "items": {"type": "string", "score": "jaro", "threshold": 0.5},
            "order": "align",
        },
    )

    assert aligner.align(["a", "b", "c"], ["c", "a", "b"]).score == 1.0
    assert aligner.align(["Python", "JavaScript", "SQL"], ["Pythn", "SQL", "JavaScrypt"]).score == pytest.approx(0.9592592592592593)
    assert aligner.align(["weight", "name", "age"], ["name", "ages", "title"]).score == pytest.approx(0.47916666666666663)


def test_reorder_alignment_size_cases_and_mismatch_structure():
    aligner = ObjectAligner("reorder", REORDER_STR_EXACT)

    assert aligner.align([], []).score == 1.0
    assert aligner.align([], ["a"]).score == 0.0
    assert aligner.align(["a"], []).score == 0.0
    assert aligner.align(["a"], ["a"]).score == 1.0

    mismatch = aligner.align(["a"], ["b"])
    assert mismatch.score == 0.0
    assert mismatch.children == [
        MatchItem(score=0.0, gold=None, pred="b"),
        MatchItem(score=0.0, gold="a", pred=None),
    ]


def test_reorder_threshold_turns_low_similarity_pairs_into_unaligned_items():
    aligner = ObjectAligner(
        "reorder-threshold",
        {
            "type": "array",
            "items": {"type": "string", "score": "jaro", "threshold": 0.9},
            "order": "align",
        },
    )
    match = aligner.align(["hello"], ["hallo"])

    assert match.score == 0.0
    assert match.children == [
        MatchItem(score=0.0, gold=None, pred="hallo"),
        MatchItem(score=0.0, gold="hello", pred=None),
    ]


def test_prefix_items_support_weighted_prefix_scoring():
    aligner = ObjectAligner(
        "prefix-only",
        {
            "type": "array",
            "prefixItems": [
                {"type": "integer", "score": "exact"},
                {"type": "string", "score": "exact"},
            ],
            "prefixWeights": [3.0, 1.0],
        },
    )

    assert aligner.align([42, "hello"], [42, "hello"]).score == 1.0
    assert aligner.align([42, "hello"], [99, "hello"]).score == pytest.approx(0.25)


def test_prefix_and_items_combined_use_importance_weighting():
    aligner = ObjectAligner(
        "prefix-combined",
        {
            "type": "array",
            "prefixItems": [
                {"type": "integer", "score": "exact"},
                {"type": "string", "score": "exact"},
            ],
            "prefixWeights": [1.0, 1.0],
            "items": {"type": "string", "score": "exact"},
            "prefixImportance": 3.0,
            "restImportance": 1.0,
        },
    )

    match = aligner.align([1, "ok", "tail1", "tail2"], [1, "ok", "bad1"])
    assert match.score == pytest.approx(0.75)


def test_ignore_excess_and_ignore_missing_change_list_normalization():
    ignore_excess = ObjectAligner(
        "ignore-excess",
        {
            "type": "array",
            "items": {"type": "string", "score": "exact"},
            "order": "align",
            "ignoreExcess": True,
        },
    )
    ignore_missing = ObjectAligner(
        "ignore-missing",
        {
            "type": "array",
            "items": {"type": "string", "score": "exact"},
            "order": "align",
            "ignoreMissing": True,
        },
    )
    ignore_both = ObjectAligner(
        "ignore-both",
        {
            "type": "array",
            "items": {"type": "string", "score": "exact"},
            "order": "align",
            "ignoreExcess": True,
            "ignoreMissing": True,
        },
    )

    assert ignore_excess.align(["a"], ["a", "b"]).score == 1.0
    assert ignore_missing.align(["a", "b"], ["a"]).score == 1.0
    assert ignore_both.align(["a"], ["a", "b"]).score == 1.0


@pytest.mark.xfail(reason="current implementation divides by zero when both ignore flags are true and no items overlap")
def test_ignore_excess_and_ignore_missing_with_total_mismatch_is_known_issue():
    aligner = ObjectAligner(
        "ignore-both-total-mismatch",
        {
            "type": "array",
            "items": {"type": "string", "score": "exact"},
            "order": "align",
            "ignoreExcess": True,
            "ignoreMissing": True,
        },
    )

    assert aligner.align(["a"], ["b"]).score == 0.0

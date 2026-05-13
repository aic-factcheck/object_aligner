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
    aligner = ObjectAligner(FIXED_INT_EXACT)

    assert aligner.align([1, 2, 3], [1, 2, 3]).score == 1.0
    assert aligner.align([1, 2, 3], [1, 3, 2]).score == pytest.approx(0.5)
    assert aligner.align([1, 2], [1, 2, 99]).score == pytest.approx(2 / 3)
    assert aligner.align([1, 2, 99], [1, 2]).score == pytest.approx(2 / 3)
    assert aligner.align([], []).score == 1.0
    assert aligner.align([], [1]).score == 0.0


def test_fixed_order_gap_behavior_matches_current_dp_alignment():
    aligner = ObjectAligner(FIXED_INT_EXACT)
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
    string_aligner = ObjectAligner({
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
            "order": "fixed",
        },
    )
    number_aligner = ObjectAligner({
            "type": "array",
            "items": {"type": "number", "score": "invdiff"},
            "order": "fixed",
        },
    )

    assert string_aligner.align(["hello"], ["hallo"]).score > 0.8
    assert number_aligner.align([1, 2], [1, 3]).score == pytest.approx(0.75)


def test_reorder_alignment_basic_cases():
    aligner = ObjectAligner({
            "type": "array",
            "items": {"type": "string", "score": "jaro", "threshold": 0.5},
            "order": "align",
        },
    )

    assert aligner.align(["a", "b", "c"], ["c", "a", "b"]).score == 1.0
    assert aligner.align(["Python", "JavaScript", "SQL"], ["Pythn", "SQL", "JavaScrypt"]).score == pytest.approx(0.9592592592592593)
    assert aligner.align(["weight", "name", "age"], ["name", "ages", "title"]).score == pytest.approx(0.47916666666666663)


def test_reorder_alignment_size_cases_and_mismatch_structure():
    aligner = ObjectAligner(REORDER_STR_EXACT)

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
    aligner = ObjectAligner({
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
    aligner = ObjectAligner({
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
    aligner = ObjectAligner({
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
    ignore_excess = ObjectAligner({
            "type": "array",
            "items": {"type": "string", "score": "exact"},
            "order": "align",
            "ignoreExcess": True,
        },
    )
    ignore_missing = ObjectAligner({
            "type": "array",
            "items": {"type": "string", "score": "exact"},
            "order": "align",
            "ignoreMissing": True,
        },
    )
    ignore_both = ObjectAligner({
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


def test_ignore_excess_and_ignore_missing_with_total_mismatch_is_known_issue():
    aligner = ObjectAligner({
            "type": "array",
            "items": {"type": "string", "score": "exact"},
            "order": "align",
            "ignoreExcess": True,
            "ignoreMissing": True,
        },
    )

    assert aligner.align(["a"], ["b"]).score == 0.0


@pytest.mark.parametrize(
    ("order", "item_type", "score", "gold", "pred", "expected_children", "expected_score"),
    [
        # Reorder: empty string in gold must be reported as missing.
        ("align", "string", "exact", ["", "a"], ["b"], 3, 0.0),
        # Fixed: empty-string mismatched against a real value — three rows,
        # normalization denominator must be 3 (so score = 1/3, not 1/2).
        ("fixed", "string", "exact", ["", "x"], ["y", "x"], 3, 1 / 3),
        # Reorder with falsy integer 0 — four "missing/excess" rows.
        ("align", "integer", "exact", [0, 1], [9, 2], 4, 0.0),
        # Reorder with False (still a bool — boolean items in lists).
        ("align", "boolean", "exact", [False, True], [True], 2, 0.5),
    ],
)
def test_falsy_primitive_items_emit_missing_and_excess_rows(
    order, item_type, score, gold, pred, expected_children, expected_score
):
    aligner = ObjectAligner({
        "type": "array",
        "items": {"type": item_type, "score": score},
        "order": order,
    })
    match = aligner.align(gold, pred)
    assert len(match.children) == expected_children, match
    assert match.score == pytest.approx(expected_score)


def test_empty_string_item_present_in_children():
    aligner = ObjectAligner({
        "type": "array",
        "items": {"type": "string", "score": "exact"},
        "order": "align",
    })
    match = aligner.align(["", "a"], ["b"])
    golds = [c.gold for c in match.children]
    preds = [c.pred for c in match.children]
    assert "" in golds  # the falsy item is not silently dropped
    assert "a" in golds
    assert "b" in preds


def test_prefix_items_pads_when_gold_shorter_than_schema():
    aligner = ObjectAligner({
        "type": "array",
        "prefixItems": [
            {"type": "string", "score": "exact"},
            {"type": "integer", "score": "exact"},
        ],
        "prefixWeights": [1.0, 1.0],
    })
    match = aligner.align(["car"], ["car", 5], skip_validation=True)
    assert len(match.children) == 2
    assert match.children[0].score == 1.0
    assert match.children[1].gold is None
    assert match.children[1].pred == 5
    assert match.score == pytest.approx(0.5)


def test_prefix_items_pads_when_pred_shorter_than_schema():
    aligner = ObjectAligner({
        "type": "array",
        "prefixItems": [
            {"type": "string", "score": "exact"},
            {"type": "integer", "score": "exact"},
        ],
        "prefixWeights": [1.0, 1.0],
    })
    match = aligner.align(["car", 5], ["car"], skip_validation=True)
    assert len(match.children) == 2
    assert match.children[0].score == 1.0
    assert match.children[1].gold == 5
    assert match.children[1].pred is None
    assert match.score == pytest.approx(0.5)


def test_prefix_items_pads_when_both_shorter_than_schema():
    aligner = ObjectAligner({
        "type": "array",
        "prefixItems": [
            {"type": "string", "score": "exact"},
            {"type": "integer", "score": "exact"},
            {"type": "string", "score": "exact"},
        ],
        "prefixWeights": [1.0, 1.0, 1.0],
    })
    match = aligner.align(["car"], ["car"], skip_validation=True)
    assert len(match.children) == 3
    # Position 0: exact match. Positions 1+2: both sides missing (sentinel).
    assert match.children[0].score == 1.0
    assert match.children[1].gold is None and match.children[1].pred is None
    assert match.children[2].gold is None and match.children[2].pred is None
    assert match.score == pytest.approx(1 / 3)


def test_prefix_weights_sum_zero_raises_at_construction():
    with pytest.raises(ValueError, match="prefixWeights"):
        ObjectAligner({
            "type": "array",
            "prefixItems": [
                {"type": "integer", "score": "exact"},
                {"type": "string", "score": "exact"},
            ],
            "prefixWeights": [0.0, 0.0],
        })


def test_prefix_importance_sum_zero_raises_at_construction():
    with pytest.raises(ValueError, match="prefixImportance and restImportance"):
        ObjectAligner({
            "type": "array",
            "prefixItems": [{"type": "integer", "score": "exact"}],
            "items": {"type": "integer", "score": "exact"},
            "prefixImportance": 0.0,
            "restImportance": 0.0,
        })

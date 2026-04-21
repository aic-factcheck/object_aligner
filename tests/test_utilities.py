from dataclasses import FrozenInstanceError

import pytest

from object_aligner.object_aligner import (
    MatchDict,
    MatchItem,
    MatchList,
    path2str,
    similarity_exact,
    similarity_num_inv_diff,
    similarity_string_jaro,
    to_pct_str,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (5, 5, 1.0),
        (5, 6, 0.0),
        ("a", "a", 1.0),
        ("a", "b", 0.0),
    ],
)
def test_similarity_exact(left, right, expected):
    assert similarity_exact(left, right) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (50, 50, 1.0),
        (50, 51, 0.5),
        (50, 52, 1 / 3),
        (50, 100, 1 / 51),
    ],
)
def test_similarity_num_inv_diff(left, right, expected):
    assert similarity_num_inv_diff(left, right) == pytest.approx(expected)


def test_similarity_string_jaro_characteristic_cases():
    assert similarity_string_jaro("hello", "hello") == 1.0
    assert similarity_string_jaro("hello", "hallo") > 0.8
    assert similarity_string_jaro("hello", "world") > 0.0
    assert similarity_string_jaro("hello", "world") < similarity_string_jaro("hello", "hallo")
    assert similarity_string_jaro("a", "b") == 0.0


def test_misc_utility_formatters():
    assert path2str([]) == "/"
    assert path2str(["person", "age"]) == "/person/age"
    assert to_pct_str(0.125) == "12%"
    assert to_pct_str(1.0) == "100%"


def test_match_dataclasses_instantiate():
    item = MatchItem(score=0.5, gold="a", pred="b")
    match_list = MatchList(score=0.5, children=[item])
    match_dict = MatchDict(score=0.5, children={item: item})

    assert item.score == 0.5
    assert match_list.children == [item]
    assert match_dict.children[item] is item


@pytest.mark.parametrize(
    "match",
    [
        MatchItem(score=0.5, gold="a", pred="b"),
        MatchList(score=0.5, children=[]),
        MatchDict(score=0.5, children={}),
    ],
)
def test_match_dataclasses_are_frozen(match):
    with pytest.raises(FrozenInstanceError):
        match.score = 1.0

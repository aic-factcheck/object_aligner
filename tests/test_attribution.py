"""Tests for tree-walk per-property score attribution."""

import pytest

from object_aligner import (
    AttributionEntry,
    AttributionResult,
    ObjectAligner,
    tree_walk_attribution,
)
from object_aligner.object_aligner import MatchDict, MatchItem, MatchList


# Tight tolerance used for invariant checks; loose tolerance for fuzzy
# primitive scores (Jaro etc.) where the upstream library defines exact
# values we don't want to hard-code.
EPS_TIGHT = 1e-9
EPS_LOOSE = 1e-6


# -----------------------------------------------------------------------------
# Group A — sum-invariant across many fixtures
# -----------------------------------------------------------------------------

INVARIANT_FIXTURES = [
    # (schema, gold, pred, description)
    (
        {"type": "string"},
        "hello",
        "hallo",
        "primitive string",
    ),
    (
        {"type": "integer", "score": "invdiff"},
        10,
        12,
        "primitive integer",
    ),
    (
        {
            "type": "object",
            "keyScore": "exact",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer", "score": "exact"},
            },
        },
        {"a": "alice", "b": 1},
        {"a": "alicia", "b": 2},
        "flat dict",
    ),
    (
        {
            "type": "object",
            "keyScore": "exact",
            "keyImportance": 0,
            "valueImportance": 1,
            "properties": {
                "x": {"type": "string", "valueWeight": 2.0},
                "y": {"type": "string", "valueWeight": 1.0},
            },
        },
        {"x": "foo", "y": "bar"},
        {"x": "fob", "y": "baz"},
        "weighted dict",
    ),
    (
        {
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
        },
        ["a", "b", "c"],
        ["a", "c"],
        "fixed list, gold longer",
    ),
    (
        {
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
            "order": "align",
        },
        ["sci-fi", "drama"],
        ["drama", "sci-fy"],
        "reorder list",
    ),
    (
        {
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
            "order": "align",
            "ignoreExcess": True,
        },
        ["alpha", "beta"],
        ["alpha", "beta", "extra"],
        "reorder list with ignoreExcess",
    ),
    (
        {
            "type": "array",
            "prefixItems": [
                {"type": "string"},
                {"type": "integer", "score": "exact"},
            ],
            "prefixWeights": [2.0, 1.0],
        },
        ["foo", 1],
        ["foe", 2],
        "prefix list",
    ),
    (
        {
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer", "score": "exact"}],
            "items": {"type": "string", "score": "jaro"},
            "prefixImportance": 1.0,
            "restImportance": 2.0,
        },
        ["foo", 1, "a", "b"],
        ["foe", 2, "x", "b"],
        "combined prefix+items",
    ),
    (
        {
            "type": "object",
            "keyScore": "exact",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "keyScore": "exact",
                        "properties": {
                            "name": {"type": "string"},
                            "qty":  {"type": "integer", "score": "exact"},
                        },
                    },
                    "order": "align",
                },
            },
        },
        {"items": [{"name": "apple", "qty": 3}, {"name": "pear", "qty": 1}]},
        {"items": [{"name": "pear", "qty": 1}, {"name": "appel", "qty": 4}]},
        "nested dict of reorder list of dicts",
    ),
]


@pytest.mark.parametrize("schema, gold, pred, description", INVARIANT_FIXTURES)
def test_invariant_sum_of_leaf_contributions_equals_deficit(schema, gold, pred, description):
    aligner = ObjectAligner(schema)
    result = aligner.attribute(gold, pred, include_empty_positions=True)

    total = sum(e.contribution for e in result.entries)
    assert abs(total - (1.0 - result.score)) < EPS_TIGHT, (
        f"[{description}] sum-invariant violated: "
        f"sum={total}, deficit={1 - result.score}"
    )


@pytest.mark.parametrize("schema, gold, pred, description", INVARIANT_FIXTURES)
def test_invariant_partition_weights_sum_to_one(schema, gold, pred, description):
    """Σ c_L over value-leaves should equal 1 modulo dict keys."""
    aligner = ObjectAligner(schema)
    result = aligner.attribute(gold, pred, include_empty_positions=True)

    # Sum every leaf weight regardless of part ("key" or "value"): they together
    # partition the root c=1 by construction.
    total_weight = sum(e.weight for e in result.entries if e.is_leaf)
    assert abs(total_weight - 1.0) < EPS_TIGHT, (
        f"[{description}] partition violated: Σ c_L = {total_weight}"
    )


@pytest.mark.parametrize("schema, gold, pred, description", INVARIANT_FIXTURES)
def test_invariant_all_weights_nonnegative(schema, gold, pred, description):
    aligner = ObjectAligner(schema)
    result = aligner.attribute(gold, pred, include_empty_positions=True)
    for e in result.entries:
        assert e.weight >= 0.0, f"[{description}] negative weight at {e.path}: {e.weight}"
        assert 0.0 <= e.score <= 1.0, f"[{description}] score out of range at {e.path}: {e.score}"
        assert e.contribution >= -EPS_TIGHT


@pytest.mark.parametrize("schema, gold, pred, description", INVARIANT_FIXTURES)
def test_entries_sorted_by_contribution_descending(schema, gold, pred, description):
    aligner = ObjectAligner(schema)
    result = aligner.attribute(gold, pred)
    contribs = [e.contribution for e in result.entries]
    assert contribs == sorted(contribs, reverse=True), (
        f"[{description}] entries not sorted descending: {contribs}"
    )


# -----------------------------------------------------------------------------
# Group B — exact worked example from research doc §2.3
# -----------------------------------------------------------------------------

MOVIE_SCHEMA = {
    "type": "object",
    "keyScore": "exact",
    "keyImportance": 0,
    "valueImportance": 1,
    "properties": {
        "title":  {"type": "string",  "score": "jaro",   "valueWeight": 2.0},
        "year":   {"type": "integer", "score": "exact",  "valueWeight": 1.0},
        "genres": {
            "type": "array",
            "items": {"type": "string", "score": "jaro"},
            "order": "align",
            "valueWeight": 1.0,
        },
    },
}


def _entry_by_path(result, path, part="value"):
    matches = [e for e in result.entries if e.path == path and e.part == part]
    assert len(matches) == 1, f"expected 1 entry for path={path!r} part={part!r}, got {len(matches)}"
    return matches[0]


def test_movie_schema_effective_weights():
    gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
    pred = {"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]}
    aligner = ObjectAligner(MOVIE_SCHEMA)
    result = aligner.attribute(gold, pred)

    assert _entry_by_path(result, "/title").weight == pytest.approx(0.5, abs=EPS_TIGHT)
    assert _entry_by_path(result, "/year").weight == pytest.approx(0.25, abs=EPS_TIGHT)
    assert _entry_by_path(result, "/genres/0").weight == pytest.approx(0.125, abs=EPS_TIGHT)
    assert _entry_by_path(result, "/genres/1").weight == pytest.approx(0.125, abs=EPS_TIGHT)


def test_movie_schema_year_dominates_attribution():
    gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
    pred = {"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]}
    aligner = ObjectAligner(MOVIE_SCHEMA)
    result = aligner.attribute(gold, pred)

    top = result.entries[0]
    assert top.path == "/year"
    assert top.contribution == pytest.approx(0.25, abs=EPS_LOOSE)
    assert _entry_by_path(result, "/year").score == 0.0


def test_movie_schema_perfect_match_zero_contributions():
    gold = {"title": "X", "year": 1, "genres": ["a"]}
    aligner = ObjectAligner(MOVIE_SCHEMA)
    result = aligner.attribute(gold, gold)

    assert result.score == 1.0
    for e in result.entries:
        assert e.contribution == pytest.approx(0.0, abs=EPS_TIGHT)


# -----------------------------------------------------------------------------
# Group C — per-aggregator checks
# -----------------------------------------------------------------------------

def test_reorder_list_ignore_excess_drops_excess_from_D():
    schema = {
        "type": "array",
        "items": {"type": "string", "score": "exact"},
        "order": "align",
        "ignoreExcess": True,
    }
    # gold has 2, pred has 3 (one excess). D should be 2 (excess dropped).
    aligner = ObjectAligner(schema)
    result = aligner.attribute(["a", "b"], ["a", "b", "extra"])

    # Excess entry is dropped from attribution (alpha=0).
    paths = [e.path for e in result.entries if e.part == "value" and e.is_leaf]
    assert "/0" in paths
    assert "/1" in paths
    # Score should be 1.0 (the two matches plus the ignored excess).
    assert result.score == pytest.approx(1.0, abs=EPS_TIGHT)


def test_reorder_list_ignore_missing_drops_missing_from_D():
    schema = {
        "type": "array",
        "items": {"type": "string", "score": "exact"},
        "order": "align",
        "ignoreMissing": True,
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute(["a", "b", "c"], ["a", "b"])

    assert result.score == pytest.approx(1.0, abs=EPS_TIGHT)
    for e in result.entries:
        assert e.contribution == pytest.approx(0.0, abs=EPS_TIGHT)


def test_fixed_list_mismatched_lengths():
    schema = {
        "type": "array",
        "items": {"type": "string", "score": "exact"},
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute(["a", "b", "c"], ["a", "x"])

    total = sum(e.contribution for e in result.entries)
    assert abs(total - (1 - result.score)) < EPS_TIGHT


def test_prefix_only_list_weights_match_normalized_prefix_weights():
    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "string"},
            {"type": "string"},
        ],
        "prefixWeights": [3.0, 1.0],
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute(["foo", "bar"], ["zzz", "yyy"])

    # Normalized weights are 0.75, 0.25.
    assert _entry_by_path(result, "/0").weight == pytest.approx(0.75, abs=EPS_TIGHT)
    assert _entry_by_path(result, "/1").weight == pytest.approx(0.25, abs=EPS_TIGHT)


def test_combined_prefix_and_items_uses_importance_mixture():
    schema = {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "string"}],
        "prefixWeights": [1.0, 1.0],
        "items": {"type": "string", "score": "exact"},
        "prefixImportance": 1.0,
        "restImportance": 3.0,
    }
    aligner = ObjectAligner(schema)
    # 2 prefix children + 2 rest children = 4 total.
    result = aligner.attribute(["a", "b", "c", "d"], ["a", "b", "c", "d"])

    # bar_pi = 0.25, bar_rho = 0.75.
    # Per prefix child: 0.25 * 0.5 = 0.125.
    # Per rest child:   0.75 / 2 = 0.375.
    assert _entry_by_path(result, "/0").weight == pytest.approx(0.125, abs=EPS_TIGHT)
    assert _entry_by_path(result, "/1").weight == pytest.approx(0.125, abs=EPS_TIGHT)
    assert _entry_by_path(result, "/2").weight == pytest.approx(0.375, abs=EPS_TIGHT)
    assert _entry_by_path(result, "/3").weight == pytest.approx(0.375, abs=EPS_TIGHT)


def test_dict_value_weight_drives_relative_attribution():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "keyImportance": 0,
        "valueImportance": 1,
        "properties": {
            "tiny": {"type": "string", "valueWeight": 1.0},
            "huge": {"type": "string", "valueWeight": 9.0},
        },
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute({"tiny": "a", "huge": "a"}, {"tiny": "z", "huge": "z"})

    tiny = _entry_by_path(result, "/tiny")
    huge = _entry_by_path(result, "/huge")
    assert tiny.weight == pytest.approx(0.1, abs=EPS_TIGHT)
    assert huge.weight == pytest.approx(0.9, abs=EPS_TIGHT)


def test_dict_key_importance_emits_key_entries():
    schema = {
        "type": "object",
        "keyScore": "jaro",
        "keyImportance": 1,
        "valueImportance": 1,
        "properties": {
            "username": {"type": "string"},
        },
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute({"username": "alice"}, {"usernme": "alice"})

    # Key entry should appear with part="key".
    key_entries = [e for e in result.entries if e.part == "key"]
    assert len(key_entries) == 1
    assert key_entries[0].weight == pytest.approx(0.5, abs=EPS_TIGHT)


def test_dict_key_importance_zero_yields_no_key_contribution():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "keyImportance": 0,
        "valueImportance": 1,
        "properties": {"a": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute({"a": "x"}, {"a": "y"})

    for e in result.entries:
        if e.part == "key":
            assert e.weight == 0.0


# -----------------------------------------------------------------------------
# Group D — granularity & flags
# -----------------------------------------------------------------------------

def test_granularity_leaf_emits_only_leaves():
    aligner = ObjectAligner(MOVIE_SCHEMA)
    gold = {"title": "X", "year": 1, "genres": ["a", "b"]}
    pred = {"title": "Y", "year": 2, "genres": ["c", "d"]}
    result = aligner.attribute(gold, pred, granularity="leaf")
    for e in result.entries:
        assert e.is_leaf, f"unexpected internal entry in leaf granularity: {e.path}"


def test_granularity_subtree_emits_only_internals():
    aligner = ObjectAligner(MOVIE_SCHEMA)
    gold = {"title": "X", "year": 1, "genres": ["a", "b"]}
    pred = {"title": "Y", "year": 2, "genres": ["c", "d"]}
    result = aligner.attribute(gold, pred, granularity="subtree")
    for e in result.entries:
        assert not e.is_leaf, f"unexpected leaf in subtree granularity: {e.path}"


def test_granularity_subtree_root_equals_deficit():
    aligner = ObjectAligner(MOVIE_SCHEMA)
    gold = {"title": "X", "year": 1, "genres": ["a", "b"]}
    pred = {"title": "Y", "year": 2, "genres": ["c", "d"]}
    result = aligner.attribute(gold, pred, granularity="subtree")
    root = [e for e in result.entries if e.path == ""]
    assert len(root) == 1
    assert root[0].contribution == pytest.approx(1.0 - result.score, abs=EPS_TIGHT)


def test_granularity_all_combines_leaves_and_internals():
    aligner = ObjectAligner(MOVIE_SCHEMA)
    gold = {"title": "X", "year": 1, "genres": ["a", "b"]}
    pred = {"title": "Y", "year": 2, "genres": ["c", "d"]}
    leaf = aligner.attribute(gold, pred, granularity="leaf")
    subtree = aligner.attribute(gold, pred, granularity="subtree")
    all_ = aligner.attribute(gold, pred, granularity="all")
    assert len(all_.entries) == len(leaf.entries) + len(subtree.entries)


def test_invalid_granularity_raises():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    with pytest.raises(ValueError, match="granularity"):
        aligner.attribute("a", "b", granularity="bogus")


def test_include_empty_positions_filters_dual_none_sentinels():
    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "string"},
            {"type": "string"},
            {"type": "string"},
        ],
    }
    aligner = ObjectAligner(schema)
    # Both gold and pred shorter than prefix → one dual-None sentinel at idx 2.
    result_filtered = aligner.attribute(["a", "b"], ["a", "b"])
    result_unfiltered = aligner.attribute(["a", "b"], ["a", "b"], include_empty_positions=True)

    paths_filtered = {e.path for e in result_filtered.entries}
    paths_unfiltered = {e.path for e in result_unfiltered.entries}
    assert "/2" not in paths_filtered
    assert "/2" in paths_unfiltered
    # When unfiltered, sum is exactly deficit.
    total_unfilt = sum(e.contribution for e in result_unfiltered.entries)
    assert abs(total_unfilt - (1 - result_unfiltered.score)) < EPS_TIGHT


# -----------------------------------------------------------------------------
# Group E — referential (id / ref)
# -----------------------------------------------------------------------------

def test_id_leaf_contributes_zero():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "people": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "keyScore": "exact",
                    "properties": {
                        "id":   {"type": "integer", "idScope": "person"},
                        "name": {"type": "string"},
                    },
                },
            },
        },
    }
    aligner = ObjectAligner(schema)
    gold = {"people": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
    pred = {"people": [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}]}
    result = aligner.attribute(gold, pred)

    id_entries = [e for e in result.entries if e.leaf_kind == "id"]
    assert len(id_entries) >= 1
    for e in id_entries:
        assert e.contribution == 0.0


def test_ref_wrong_pair_contributes_its_weight():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "people": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "keyScore": "exact",
                    "properties": {
                        "id":   {"type": "integer", "idScope": "person"},
                        "name": {"type": "string"},
                    },
                },
            },
            "relations": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "keyScore": "exact",
                    "properties": {
                        "source": {"type": "integer", "ref": "person"},
                        "target": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    }
    aligner = ObjectAligner(schema)
    # Same graph topology — refs should resolve correctly.
    gold = {
        "people": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "relations": [{"source": 1, "target": 2}],
    }
    pred = {
        "people": [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}],
        "relations": [{"source": 53, "target": 124}],
    }
    result = aligner.attribute(gold, pred)
    ref_entries = [e for e in result.entries if e.leaf_kind == "ref"]
    # All refs should be correctly resolved → score 1, contribution 0.
    for e in ref_entries:
        assert e.score == 1.0
        assert e.contribution == 0.0


def test_ref_swapped_pair_contributes_full_weight():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "people": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "keyScore": "exact",
                    "properties": {
                        "id":   {"type": "integer", "idScope": "person"},
                        "name": {"type": "string"},
                    },
                },
            },
            "relations": {
                "type": "array",
                "order": "align",
                "items": {
                    "type": "object",
                    "keyScore": "exact",
                    "properties": {
                        "source": {"type": "integer", "ref": "person"},
                        "target": {"type": "integer", "ref": "person"},
                    },
                },
            },
        },
    }
    aligner = ObjectAligner(schema)
    gold = {
        "people": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "relations": [{"source": 1, "target": 2}],
    }
    # Swap source/target on pred — refs should resolve but to the wrong way.
    pred = {
        "people": [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}],
        "relations": [{"source": 124, "target": 53}],
    }
    result = aligner.attribute(gold, pred)
    wrong_refs = [e for e in result.entries if e.leaf_kind == "ref" and e.score == 0.0]
    assert len(wrong_refs) == 2  # both source and target wrong
    for e in wrong_refs:
        assert e.contribution == pytest.approx(e.weight, abs=EPS_TIGHT)


# -----------------------------------------------------------------------------
# Group F — edge cases
# -----------------------------------------------------------------------------

def test_perfect_match_yields_zero_residual_and_zero_contributions():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    gold = {"a": "alpha", "b": "beta"}
    result = aligner.attribute(gold, gold)
    assert result.score == 1.0
    for e in result.entries:
        assert e.contribution == pytest.approx(0.0, abs=EPS_TIGHT)


def test_all_zero_match_contributions_distribute_by_weight():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "keyImportance": 0,
        "valueImportance": 1,
        "properties": {
            "a": {"type": "string", "score": "exact", "valueWeight": 1.0},
            "b": {"type": "string", "score": "exact", "valueWeight": 3.0},
        },
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute({"a": "x", "b": "y"}, {"a": "X", "b": "Y"})

    assert result.score == 0.0
    a = _entry_by_path(result, "/a")
    b = _entry_by_path(result, "/b")
    assert a.contribution == pytest.approx(0.25, abs=EPS_TIGHT)
    assert b.contribution == pytest.approx(0.75, abs=EPS_TIGHT)


def test_empty_list_degenerates_to_leaf():
    schema = {"type": "array", "items": {"type": "string"}}
    aligner = ObjectAligner(schema)
    result = aligner.attribute([], [])
    assert result.score == 1.0
    assert len(result.entries) == 1
    assert result.entries[0].is_leaf
    assert result.entries[0].contribution == 0.0


def test_empty_dict_degenerates_to_leaf():
    schema = {"type": "object", "keyScore": "exact", "properties": {}}
    aligner = ObjectAligner(schema)
    result = aligner.attribute({}, {})
    assert result.score == 1.0
    assert len(result.entries) == 1
    assert result.entries[0].is_leaf


def test_threshold_clipped_leaf_uses_clipped_score():
    schema = {"type": "string", "score": "jaro", "threshold": 0.95}
    aligner = ObjectAligner(schema)
    result = aligner.attribute("hello", "world")
    # Jaro of "hello" vs "world" is well below 0.95; clipped to 0.
    e = result.entries[0]
    assert e.score == 0.0
    assert e.contribution == pytest.approx(1.0, abs=EPS_TIGHT)


def test_pred_validation_failure_returns_empty_result():
    schema = {"type": "object", "required": ["a"], "keyScore": "exact",
              "properties": {"a": {"type": "string"}}}
    aligner = ObjectAligner(schema)
    result = aligner.attribute({"a": "x"}, {})  # pred missing required "a"
    assert result.score == 0.0
    assert result.entries == ()


def test_skip_validation_skips_jsonschema_check():
    schema = {"type": "object", "required": ["a"], "keyScore": "exact",
              "properties": {"a": {"type": "string"}}}
    aligner = ObjectAligner(schema)
    # With skip_validation=True, no validation runs; bad data goes straight
    # to align. We pass valid data to keep the test focused.
    result = aligner.attribute({"a": "x"}, {"a": "y"}, skip_validation=True)
    assert result.score < 1.0


# -----------------------------------------------------------------------------
# Group G — JSON Pointer encoding
# -----------------------------------------------------------------------------

def test_root_path_is_empty_string():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    result = aligner.attribute("a", "b")
    assert result.entries[0].path == ""


def test_list_paths_use_indices():
    schema = {"type": "array", "items": {"type": "string", "score": "jaro"}}
    aligner = ObjectAligner(schema)
    # All three positions match well enough (jaro > 0), so the aggregator emits
    # one child per gold position.
    result = aligner.attribute(["alpha", "beta", "gamma"], ["alpha", "betta", "gamma"])
    paths = {e.path for e in result.entries}
    assert paths == {"/0", "/1", "/2"}


def test_dict_paths_use_keys():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {"alpha": {"type": "string"}, "beta": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute({"alpha": "x", "beta": "y"}, {"alpha": "x", "beta": "Y"})
    paths = {e.path for e in result.entries}
    assert paths >= {"/alpha", "/beta"}


def test_nested_paths_compose_with_slashes():
    schema = {
        "type": "object",
        "keyScore": "exact",
        "properties": {
            "outer": {
                "type": "object",
                "keyScore": "exact",
                "properties": {"inner": {"type": "string"}},
            },
        },
    }
    aligner = ObjectAligner(schema)
    result = aligner.attribute({"outer": {"inner": "a"}}, {"outer": {"inner": "b"}})
    inner_paths = [e.path for e in result.entries if e.is_leaf and e.part == "value"]
    assert "/outer/inner" in inner_paths


def test_json_pointer_escapes_tilde_and_slash():
    """RFC 6901: ~ → ~0, / → ~1."""
    from object_aligner.attribution import _encode_pointer_token
    assert _encode_pointer_token("a/b") == "a~1b"
    assert _encode_pointer_token("a~b") == "a~0b"
    assert _encode_pointer_token("a~/b") == "a~0~1b"
    # Order matters: ~ must be escaped before /, else "/" -> "~1" -> "~01".
    assert _encode_pointer_token("~1") == "~01"


def test_non_string_dict_keys_are_stringified():
    from object_aligner.attribution import _encode_pointer_token
    assert _encode_pointer_token(42) == "42"
    assert _encode_pointer_token(True) == "True"


# -----------------------------------------------------------------------------
# Group H — public API surface
# -----------------------------------------------------------------------------

def test_attribute_from_match_skips_realignment():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    match_tree = aligner.align("hello", "hallo")
    result = aligner.attribute_from_match(match_tree)
    assert result.score == pytest.approx(match_tree.score, abs=EPS_TIGHT)


def test_attribution_result_is_iterable():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    result = aligner.attribute("hello", "hallo")
    iterated = list(result)
    indexed = [result[i] for i in range(len(result))]
    assert iterated == indexed


def test_attribution_entry_is_hashable():
    e = AttributionEntry(
        path="/a", score=0.5, weight=0.5, contribution=0.25,
        gold="x", pred="y", is_leaf=True,
    )
    s = {e}  # would raise if not hashable
    assert e in s


def test_attribution_result_is_hashable():
    r = AttributionResult(score=0.5, entries=(), granularity="leaf",
                         total_contribution=0.5, residual=0.0)
    assert r in {r}


def test_attribution_entry_is_frozen():
    e = AttributionEntry(
        path="/a", score=0.5, weight=0.5, contribution=0.25,
        gold="x", pred="y", is_leaf=True,
    )
    with pytest.raises(Exception):
        e.path = "/changed"


def test_tree_walk_attribution_callable_directly():
    """The functional core is also importable for advanced use."""
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    tree = aligner.align("hello", "hallo")
    result = tree_walk_attribution(tree, schema)
    assert result.score == pytest.approx(tree.score, abs=EPS_TIGHT)
    assert len(result.entries) == 1

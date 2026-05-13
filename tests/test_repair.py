"""Tests for scored JSON-Patch repair generation."""

import pytest

from object_aligner import ObjectAligner, RepairOp, RepairResult, generate_repairs
from object_aligner.repair import _apply_op


EPS_TIGHT = 1e-9
EPS_LOOSE = 1e-6


# -----------------------------------------------------------------------------
# Group A — sum-invariant across many fixtures
# -----------------------------------------------------------------------------

INVARIANT_FIXTURES = [
    (
        {"type": "string"},
        "hello", "hallo",
        "primitive string",
    ),
    (
        {"type": "integer", "score": "invdiff"},
        10, 12,
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
            "keyImportance": 0, "valueImportance": 1,
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
        {"type": "array", "items": {"type": "string", "score": "jaro"}},
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
            "items": {"type": "string", "score": "exact"},
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
            "prefixItems": [{"type": "string"}, {"type": "integer", "score": "exact"}],
            "prefixWeights": [2.0, 1.0],
        },
        ["foo", 1],
        ["foe", 2],
        "prefix list",
    ),
    (
        {
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "string"}],
            "items": {"type": "string", "score": "exact"},
            "prefixImportance": 1.0,
            "restImportance": 2.0,
        },
        ["foo", "bar", "a", "b"],
        ["foe", "bar", "x", "b"],
        "combined prefix + items",
    ),
]


@pytest.mark.parametrize("schema, gold, pred, label", INVARIANT_FIXTURES)
def test_sum_of_score_deltas_equals_deficit_leaf(schema, gold, pred, label):
    aligner = ObjectAligner(schema)
    r = aligner.repair(gold, pred)
    total = sum(op.score_delta for op in r.ops)
    assert abs(total - (1 - r.score)) < EPS_TIGHT, (
        f"[{label}] sum invariant: total={total}, deficit={1 - r.score}"
    )


@pytest.mark.parametrize("schema, gold, pred, label", INVARIANT_FIXTURES)
def test_score_deltas_nonnegative(schema, gold, pred, label):
    aligner = ObjectAligner(schema)
    r = aligner.repair(gold, pred)
    for op in r.ops:
        assert op.score_delta >= -EPS_TIGHT, f"[{label}] negative delta at {op.path!r}"


@pytest.mark.parametrize("schema, gold, pred, label", INVARIANT_FIXTURES)
def test_ops_sorted_descending(schema, gold, pred, label):
    aligner = ObjectAligner(schema)
    r = aligner.repair(gold, pred)
    deltas = [op.score_delta for op in r.ops]
    assert deltas == sorted(deltas, reverse=True), f"[{label}] not sorted: {deltas}"


# -----------------------------------------------------------------------------
# Group B — round-trip: apply ops -> re-align -> score moves
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("schema, gold, pred, label", INVARIANT_FIXTURES)
def test_apply_all_reaches_perfect_score(schema, gold, pred, label):
    """For these fixtures, applying every emitted op should bring score to 1.0."""
    aligner = ObjectAligner(schema)
    r = aligner.repair(gold, pred)
    patched = r.apply_to(pred)
    new_score = aligner.metric(gold, patched)["score"]
    assert new_score == pytest.approx(1.0, abs=EPS_LOOSE), (
        f"[{label}] apply-all reached {new_score}, not 1.0; ops={list(r.ops)}"
    )


def test_apply_does_not_mutate_input():
    schema = {"type": "object", "keyScore": "exact",
              "properties": {"a": {"type": "string"}}}
    aligner = ObjectAligner(schema)
    pred = {"a": "x"}
    r = aligner.repair({"a": "y"}, pred)
    _ = r.apply_to(pred)
    assert pred == {"a": "x"}  # unchanged


def test_single_op_score_delta_matches_actual_gain_no_hungarian():
    """On a Hungarian-free schema, applying one op alone gives delta == score_delta."""
    schema = {
        "type": "object",
        "keyScore": "exact",
        "keyImportance": 0, "valueImportance": 1,
        "properties": {
            "a": {"type": "string", "score": "exact"},
            "b": {"type": "string", "score": "exact"},
        },
    }
    aligner = ObjectAligner(schema)
    gold = {"a": "x", "b": "y"}
    pred = {"a": "X", "b": "Y"}
    r = aligner.repair(gold, pred)
    baseline = r.score
    for op in r.ops:
        # Apply only this op.
        intermediate = _apply_op({"a": "X", "b": "Y"}, op)
        new_score = aligner.metric(gold, intermediate)["score"]
        assert (new_score - baseline) == pytest.approx(op.score_delta, abs=EPS_LOOSE), (
            f"single-op gain for {op.path} = {new_score - baseline}, expected {op.score_delta}"
        )


# -----------------------------------------------------------------------------
# Group C — per-op-kind
# -----------------------------------------------------------------------------

def test_primitive_replace_op_shape():
    schema = {"type": "object", "keyScore": "exact",
              "properties": {"a": {"type": "string"}}}
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "alice"}, {"a": "bob"})
    ops = [op for op in r.ops if op.kind == "primitive_replace"]
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "replace"
    assert op.path == "/a"
    assert op.value == "alice"
    assert op.gold == "alice"
    assert op.pred == "bob"


def test_key_add_emitted_for_missing_gold_key():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
        },
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "x", "b": "y"}, {"a": "x"})
    ops = [op for op in r.ops if op.kind == "key_add"]
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "add"
    assert op.path == "/b"
    assert op.value == "y"


def test_key_remove_emitted_for_excess_pred_key():
    schema = {
        "type": "object", "keyScore": "exact",
        "additionalProperties": True,
        "properties": {"a": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "x"}, {"a": "x", "extra": "z"})
    ops = [op for op in r.ops if op.kind == "key_remove"]
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "remove"
    assert op.path == "/extra"


def test_key_rename_emits_pair_with_shared_pair_id():
    schema = {
        "type": "object",
        "keyScore": "jaro",
        "keyImportance": 1, "valueImportance": 1,
        "additionalProperties": True,
        "properties": {"username": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"username": "alice"}, {"usrname": "alice"})

    remove_ops = [op for op in r.ops if op.kind == "key_rename_remove"]
    add_ops = [op for op in r.ops if op.kind == "key_rename_add"]
    assert len(remove_ops) == 1
    assert len(add_ops) == 1
    assert remove_ops[0].pair_id == add_ops[0].pair_id
    assert remove_ops[0].pair_id != ""
    # Remove carries zero delta; add carries the full key gain.
    assert remove_ops[0].score_delta == 0.0
    assert add_ops[0].score_delta > 0.0
    # Add carries gold value.
    assert add_ops[0].value == "alice"


def test_list_item_add_for_positional_missing():
    schema = {"type": "array", "items": {"type": "string", "score": "exact"}}
    aligner = ObjectAligner(schema)
    r = aligner.repair(["a", "b", "c"], ["a", "c"])
    adds = [op for op in r.ops if op.kind == "list_item_add"]
    assert any(op.value == "b" for op in adds)


def test_list_item_remove_for_positional_excess():
    schema = {"type": "array", "items": {"type": "string", "score": "exact"}}
    aligner = ObjectAligner(schema)
    r = aligner.repair(["a", "c"], ["a", "b", "c"])
    removes = [op for op in r.ops if op.kind == "list_item_remove"]
    assert len(removes) >= 1


def test_list_item_missing_for_reorder_missing():
    schema = {"type": "array", "items": {"type": "string", "score": "exact"}, "order": "align"}
    aligner = ObjectAligner(schema)
    r = aligner.repair(["a", "b", "c"], ["b", "a"])
    missing = [op for op in r.ops if op.kind == "list_item_missing"]
    assert len(missing) == 1
    assert missing[0].path == ""  # root (the list itself)
    assert missing[0].value == "c"


def test_list_item_excess_for_reorder_excess():
    schema = {"type": "array", "items": {"type": "string", "score": "exact"}, "order": "align"}
    aligner = ObjectAligner(schema)
    r = aligner.repair(["a", "b"], ["b", "a", "c"])
    excess = [op for op in r.ops if op.kind == "list_item_excess"]
    assert len(excess) == 1
    assert excess[0].path == ""  # the list itself
    assert excess[0].pred == "c"


def test_primitive_replace_reorder_path_is_list_level():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {
            "genres": {
                "type": "array",
                "items": {"type": "string", "score": "jaro"},
                "order": "align",
            },
        },
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair(
        {"genres": ["Action", "Drama"]},
        {"genres": ["Drama", "Adventure"]},
    )
    reorder_ops = [op for op in r.ops if op.kind == "primitive_replace_reorder"]
    assert len(reorder_ops) == 1
    assert reorder_ops[0].path == "/genres"  # NOT /genres/N
    assert reorder_ops[0].value == "Action"  # gold value
    assert reorder_ops[0].pred == "Adventure"  # paired pred value


def test_subtree_replace_in_subtree_mode():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
        },
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "x", "b": "y"}, {"a": "X", "b": "Y"}, granularity="subtree")
    subtree_ops = [op for op in r.ops if op.kind == "subtree_replace"]
    assert len(subtree_ops) == 1
    # The root subtree_replace's delta equals the deficit.
    assert subtree_ops[0].score_delta == pytest.approx(1.0 - r.score, abs=EPS_TIGHT)


# -----------------------------------------------------------------------------
# Group D — granularity modes
# -----------------------------------------------------------------------------

def test_granularity_leaf_emits_no_subtree_replace():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "x", "b": "y"}, {"a": "X", "b": "Y"}, granularity="leaf")
    assert all(op.kind != "subtree_replace" for op in r.ops)


def test_granularity_subtree_emits_only_subtree_replaces():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "x", "b": "y"}, {"a": "X", "b": "Y"}, granularity="subtree")
    assert all(op.kind == "subtree_replace" for op in r.ops)


def test_granularity_all_includes_both():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "x", "b": "y"}, {"a": "X", "b": "Y"}, granularity="all")
    kinds = {op.kind for op in r.ops}
    assert "subtree_replace" in kinds
    assert "primitive_replace" in kinds


def test_invalid_granularity_raises():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    with pytest.raises(ValueError, match="granularity"):
        aligner.repair("a", "b", granularity="bogus")


def test_granularity_subtree_apply_root_replace_reaches_perfect():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    gold = {"a": "x", "b": "y"}
    pred = {"a": "X", "b": "Y"}
    r = aligner.repair(gold, pred, granularity="subtree")
    # Apply just the root subtree_replace (the largest).
    patched = _apply_op(dict(pred), r.ops[0])
    assert aligner.metric(gold, patched)["score"] == pytest.approx(1.0, abs=EPS_TIGHT)


# -----------------------------------------------------------------------------
# Group E — min_contribution filtering
# -----------------------------------------------------------------------------

def test_min_contribution_filters_small_ops():
    schema = {
        "type": "object", "keyScore": "exact",
        "keyImportance": 0, "valueImportance": 1,
        "properties": {
            "big":   {"type": "string", "score": "exact", "valueWeight": 9.0},
            "small": {"type": "string", "score": "exact", "valueWeight": 1.0},
        },
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"big": "x", "small": "y"},
                       {"big": "X", "small": "Y"},
                       min_contribution=0.5)
    # 'big' is at weight 0.9 → delta = 0.9; 'small' at 0.1 → delta = 0.1.
    # Threshold 0.5 keeps only big.
    paths = {op.path for op in r.ops}
    assert "/big" in paths
    assert "/small" not in paths
    # Residual reflects the dropped contribution.
    assert r.residual == pytest.approx(-0.1, abs=EPS_TIGHT)


def test_min_contribution_keeps_key_rename_pair_atomically():
    schema = {
        "type": "object",
        "keyScore": "jaro",
        "keyImportance": 1, "valueImportance": 1,
        "additionalProperties": True,
        "properties": {"username": {"type": "string"}},
    }
    aligner = ObjectAligner(schema)
    r = aligner.repair({"username": "alice"}, {"usrname": "alice"})
    add_delta = next(op.score_delta for op in r.ops if op.kind == "key_rename_add")
    # Filter at half of add_delta — pair should stay.
    r2 = aligner.repair({"username": "alice"}, {"usrname": "alice"},
                        min_contribution=add_delta / 2)
    pair_ops = [op for op in r2.ops if op.pair_id]
    assert len(pair_ops) == 2

    # Filter above add_delta — pair dropped together.
    r3 = aligner.repair({"username": "alice"}, {"usrname": "alice"},
                        min_contribution=add_delta + 0.1)
    pair_ops = [op for op in r3.ops if op.pair_id]
    assert len(pair_ops) == 0


# -----------------------------------------------------------------------------
# Group F — reorder list semantics + apply round-trip
# -----------------------------------------------------------------------------

def test_reorder_apply_to_primitive_replace_works():
    schema = {"type": "array", "items": {"type": "string", "score": "jaro"},
              "order": "align"}
    aligner = ObjectAligner(schema)
    gold = ["alpha", "beta"]
    pred = ["alphz", "beta"]
    r = aligner.repair(gold, pred)
    patched = r.apply_to(pred)
    assert "alpha" in patched
    assert "alphz" not in patched


def test_reorder_apply_handles_missing_and_excess():
    schema = {"type": "array", "items": {"type": "string", "score": "exact"},
              "order": "align"}
    aligner = ObjectAligner(schema)
    gold = ["a", "b"]
    pred = ["b", "c"]  # missing 'a', excess 'c'
    r = aligner.repair(gold, pred)
    patched = r.apply_to(pred)
    # The patched list should contain 'a' and 'b'.
    assert "a" in patched
    assert "b" in patched
    assert "c" not in patched


def test_reorder_notes_emitted():
    schema = {"type": "array", "items": {"type": "string"}, "order": "align"}
    aligner = ObjectAligner(schema)
    r = aligner.repair(["a"], ["b"])
    assert any("align" in note for note in r.notes)


# -----------------------------------------------------------------------------
# Group G — referential (id / ref)
# -----------------------------------------------------------------------------

def test_id_leaf_emits_no_op():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {
            "people": {
                "type": "array", "order": "align",
                "items": {
                    "type": "object", "keyScore": "exact",
                    "properties": {
                        "id":   {"type": "integer", "idScope": "person"},
                        "name": {"type": "string"},
                    },
                },
            },
        },
    }
    aligner = ObjectAligner(schema)
    gold = {"people": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]}
    pred = {"people": [{"id": 53, "name": "A"}, {"id": 124, "name": "B"}]}
    r = aligner.repair(gold, pred)
    # No op should reference the id field (the ids are pred-side arbitrary).
    for op in r.ops:
        assert "/id" not in op.path or op.kind == "ref_fix"


def test_ref_fix_uses_pred_side_mapped_id():
    schema = {
        "type": "object", "keyScore": "exact",
        "properties": {
            "people": {
                "type": "array", "order": "align",
                "items": {
                    "type": "object", "keyScore": "exact",
                    "properties": {
                        "id":   {"type": "integer", "idScope": "person"},
                        "name": {"type": "string"},
                    },
                },
            },
            "relations": {
                "type": "array", "order": "align",
                "items": {
                    "type": "object", "keyScore": "exact",
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
        "people":    [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "relations": [{"source": 1, "target": 2}],
    }
    pred = {
        "people":    [{"id": 53, "name": "Alice"}, {"id": 124, "name": "Bob"}],
        "relations": [{"source": 124, "target": 53}],  # swapped
    }
    r = aligner.repair(gold, pred)
    ref_ops = [op for op in r.ops if op.kind == "ref_fix"]
    assert len(ref_ops) == 2
    # The replacement value should be a pred-side id from the bijection.
    for op in ref_ops:
        assert op.value in (53, 124)


# -----------------------------------------------------------------------------
# Group H — public API surface
# -----------------------------------------------------------------------------

def test_repair_from_match_skips_realignment():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    tree = aligner.align("hello", "hallo")
    r = aligner.repair_from_match(tree, "hello", "hallo")
    assert r.score == pytest.approx(tree.score, abs=EPS_TIGHT)


def test_pred_validation_failure_yields_empty_result():
    schema = {"type": "object", "required": ["a"], "keyScore": "exact",
              "properties": {"a": {"type": "string"}}}
    aligner = ObjectAligner(schema)
    r = aligner.repair({"a": "x"}, {})
    assert r.score == 0.0
    assert r.ops == ()


def test_repair_op_is_frozen():
    op = RepairOp(op="replace", path="/a", score_delta=0.5, kind="primitive_replace")
    with pytest.raises(Exception):
        op.score_delta = 0.6


def test_repair_op_is_hashable():
    op = RepairOp(op="replace", path="/a", score_delta=0.5, kind="primitive_replace")
    s = {op}
    assert op in s


def test_repair_result_iterable_and_indexable():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    r = aligner.repair("hello", "hallo")
    assert list(r) == [r[i] for i in range(len(r))]


def test_generate_repairs_directly_callable():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    tree = aligner.align("hello", "hallo")
    r = generate_repairs(tree, schema, "hello", "hallo", None)
    assert r.score == pytest.approx(tree.score, abs=EPS_TIGHT)


# -----------------------------------------------------------------------------
# Group I — edge cases
# -----------------------------------------------------------------------------

def test_perfect_match_yields_no_ops():
    schema = {"type": "string"}
    aligner = ObjectAligner(schema)
    r = aligner.repair("hello", "hello")
    assert r.score == 1.0
    assert r.ops == ()


def test_empty_list_no_ops():
    schema = {"type": "array", "items": {"type": "string"}}
    aligner = ObjectAligner(schema)
    r = aligner.repair([], [])
    assert r.score == 1.0
    assert r.ops == ()


def test_empty_dict_no_ops():
    schema = {"type": "object", "keyScore": "exact", "properties": {}}
    aligner = ObjectAligner(schema)
    r = aligner.repair({}, {})
    assert r.score == 1.0
    assert r.ops == ()


def test_threshold_clipped_leaf_uses_post_clip_score():
    schema = {"type": "string", "score": "jaro", "threshold": 0.95}
    aligner = ObjectAligner(schema)
    r = aligner.repair("hello", "hallo")  # jaro ≈ 0.87, clipped to 0
    assert len(r.ops) == 1
    op = r.ops[0]
    assert op.score_delta == pytest.approx(1.0, abs=EPS_TIGHT)
    assert op.value == "hello"


def test_skip_validation_with_repair():
    schema = {"type": "object", "required": ["a"], "keyScore": "exact",
              "properties": {"a": {"type": "string"}}}
    aligner = ObjectAligner(schema)
    # Inputs satisfy the schema; passing skip_validation just bypasses the check.
    r = aligner.repair({"a": "x"}, {"a": "y"}, skip_validation=True)
    assert r.score < 1.0


def test_nested_movie_schema_invariant_and_round_trip():
    """Canonical example: verify invariant and round-trip apply."""
    schema = {
        "type": "object",
        "keyScore": "exact",
        "keyImportance": 0, "valueImportance": 1,
        "properties": {
            "title":  {"type": "string",  "score": "jaro",  "valueWeight": 2.0},
            "year":   {"type": "integer", "score": "exact", "valueWeight": 1.0},
            "genres": {
                "type": "array",
                "items": {"type": "string", "score": "jaro"},
                "order": "align",
                "valueWeight": 1.0,
            },
        },
    }
    aligner = ObjectAligner(schema)
    gold = {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"]}
    pred = {"title": "The Matrx",  "year": 2000, "genres": ["Sci-Fi", "Adventure"]}
    r = aligner.repair(gold, pred)
    # Top op should be /year (largest contribution).
    assert r.ops[0].path == "/year"
    # Apply-all reaches 1.0.
    patched = r.apply_to(pred)
    assert aligner.metric(gold, patched)["score"] == pytest.approx(1.0, abs=EPS_LOOSE)
    # Original unmutated.
    assert pred["year"] == 2000


def test_repair_invalid_pred_returns_empty_with_sentinel_residual():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "keyScore": "exact",
    }
    aligner = ObjectAligner(schema)
    result = aligner.repair({"name": "A", "age": 1}, {"name": "A"})
    assert result.score == 0.0
    assert result.ops == ()
    assert result.residual == -1.0

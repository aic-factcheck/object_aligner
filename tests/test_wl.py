"""Unit tests for the pure Weisfeiler–Leman refinement core (object_aligner._wl).

These exercise ``wl_tokens`` directly on hand-built graphs, independent of any
schema or instance data, so the refinement's correctness and determinism can be
pinned down in isolation. The integration with the cost matrix lives in
``test_referential.py``.
"""

import random

from object_aligner._wl import RefGraph, _RefEdge, wl_tokens


def _digraph(vertex_ids, edges, *, role=("e",), label=()):
    """Build a RefGraph: attribute-less vertices + directed labeled edges."""
    graph = RefGraph()
    for vid in vertex_ids:
        graph.vertices[vid] = ()
    for src, dst in edges:
        graph.incidences.append(_RefEdge(src=src, dst=dst, role=role, label=label))
    return graph


def _partition(tokens):
    """The unordered partition (set of vertex-groups) induced by the tokens."""
    groups = {}
    for vid, tok in tokens.items():
        groups.setdefault(tok, set()).add(vid)
    return sorted((frozenset(g) for g in groups.values()), key=repr)


def test_directed_path_fully_refines_and_matches_cross_side():
    gold = _digraph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")])
    pred = _digraph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    gt, pt = wl_tokens(gold, pred)

    assert len(set(gt.values())) == 4  # four distinct colors per side
    assert len(set(pt.values())) == 4
    # Structural correspondence is recovered despite different id values.
    assert gt["a"] == pt[0]
    assert gt["b"] == pt[1]
    assert gt["c"] == pt[2]
    assert gt["d"] == pt[3]


def test_directed_3_cycle_is_single_color_automorphism():
    gold = _digraph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    pred = _digraph(["x", "y", "z"], [("x", "y"), ("y", "z"), ("z", "x")])
    gt, pt = wl_tokens(gold, pred)

    assert len(set(gt.values())) == 1  # every node interchangeable
    assert len(set(pt.values())) == 1
    assert set(gt.values()) == set(pt.values())  # same color across sides


def test_path_direction_is_respected():
    # gold: a->b->c ; pred: 2->1->0 (same shape, reversed id ordering).
    gold = _digraph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    pred = _digraph([0, 1, 2], [(1, 0), (2, 1)])
    gt, pt = wl_tokens(gold, pred)

    # Gold source 'a' (out-only) matches pred source 2 (out-only), never the
    # pred sink 0 (in-only).
    assert gt["a"] == pt[2]
    assert gt["a"] != pt[0]


def test_six_cycle_vs_two_triangles_is_a_blind_spot():
    six = _digraph(
        [0, 1, 2, 3, 4, 5],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    )
    two_triangles = _digraph(
        ["a", "b", "c", "d", "e", "f"],
        [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d")],
    )
    gt, pt = wl_tokens(six, two_triangles)

    # 1-WL collapses every 2-regular vertex to one color on both sides, so the
    # two non-isomorphic graphs are (correctly, for 1-WL) indistinguishable.
    assert len(set(gt.values())) == 1
    assert len(set(pt.values())) == 1
    assert set(gt.values()) == set(pt.values())


def test_edge_label_splits_otherwise_identical_structure():
    gold = _digraph(["a", "b"], [("a", "b")], label=(("type", "knows"),))
    pred_same = _digraph([0, 1], [(0, 1)], label=(("type", "knows"),))
    pred_diff = _digraph([0, 1], [(0, 1)], label=(("type", "hates"),))

    gt, pt_same = wl_tokens(gold, pred_same)
    assert gt["a"] == pt_same[0]

    gt2, pt_diff = wl_tokens(gold, pred_diff)
    assert gt2["a"] != pt_diff[0]  # different edge label -> not comparable


def test_parallel_edges_preserve_multiplicity():
    one = _digraph(["a", "b"], [("a", "b")])
    two = _digraph([0, 1], [(0, 1), (0, 1)])
    gt, pt = wl_tokens(one, two)

    # 'a' has one out-edge, '0' has two -> different multiset -> different token.
    assert gt["a"] != pt[0]


def test_blend_seed_uses_own_scalars():
    gold = RefGraph(vertices={"a": (("name", "x"),), "b": (("name", "y"),)})
    pred = RefGraph(vertices={0: (("name", "x"),), 1: (("name", "y"),)})
    gt, pt = wl_tokens(gold, pred, mode="blend")

    # With no edges, color comes purely from own scalars under blend.
    assert gt["a"] == pt[0]
    assert gt["b"] == pt[1]
    assert gt["a"] != gt["b"]

    # Under tie_break the same vertices are a single constant color.
    gt_tb, _ = wl_tokens(gold, pred, mode="tie_break")
    assert len(set(gt_tb.values())) == 1


def test_determinism_under_input_reordering():
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    canonical = _digraph([0, 1, 2, 3], edges)

    shuffled_edges = edges[:]
    random.Random(20240527).shuffle(shuffled_edges)
    reordered = RefGraph()
    for vid in [3, 1, 0, 2]:
        reordered.vertices[vid] = ()
    for src, dst in shuffled_edges:
        reordered.incidences.append(_RefEdge(src=src, dst=dst, role=("e",), label=()))

    t1, _ = wl_tokens(canonical, canonical)
    t2, _ = wl_tokens(reordered, reordered)
    assert _partition(t1) == _partition(t2)


def test_empty_graphs_return_empty():
    assert wl_tokens(RefGraph(), RefGraph()) == ({}, {})


def test_rounds_zero_yields_seed_partition():
    gold = _digraph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    pred = _digraph([0, 1, 2], [(0, 1), (1, 2)])
    gt, pt = wl_tokens(gold, pred, rounds=0)

    # No refinement: the tie_break seed is a single constant color.
    assert len(set(gt.values())) == 1
    assert len(set(pt.values())) == 1

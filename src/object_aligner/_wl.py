"""Weisfeiler–Leman (1-WL) color refinement for id-scope bijection derivation.

Internal module — not part of the public API and not re-exported from the
package root. The functions here are pure and side-effect-free: they operate on
abstract :class:`RefGraph` structures (vertex ids plus directed, labeled
incidences) and know nothing about JSON schemas, instance data, or alignment
context. Graph *construction* lives on ``ObjectAligner`` (it needs the schema
walkers and the per-call referential mappings); only the refinement itself
lives here so it can be unit-tested in isolation.

The refinement runs over the *disjoint union* of the gold and pred graphs with a
single signature→token dictionary shared across both sides each round. Identical
structural signatures therefore receive identical integer tokens on both sides,
which is what makes the per-side colors comparable — without ever consulting a
cross-side id mapping (so no bootstrapping problem). See ``docs/referential.md``.

In the paper's terms, 1-WL is the tractable approximation to the graph
isomorphism test used to break ties between *property-identical twins* —
records that agree on every non-id, non-ref attribute and so are
indistinguishable to the property cost matrix, yet differ in structure.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _RefEdge:
    """One directed, labeled incidence contributed by a carrier object.

    ``src`` and ``dst`` are vertex ids (referenced definer id values);
    ``src == dst`` is a self-loop. ``role`` encodes the ref-field identity
    (and, for k-ary / symmetric relations, the star-to-hub role) and ``label``
    carries the carrier's exactly-comparable scalars plus any resolved
    higher-scope targets. Both are hashable, ``repr``-sortable tuples.
    """

    src: Any
    dst: Any
    role: tuple
    label: tuple


@dataclass
class RefGraph:
    """A per-side directed, labeled multigraph over one scope's definer items.

    ``vertices`` maps every vertex id (definer ids plus any synthetic hub ids)
    to its own exactly-comparable scalar label tuple — used only by the
    ``"blend"`` seeding; ``()`` under ``"tie_break"``. ``incidences`` is the
    flat list of directed labeled edges; parallel edges and self-loops are
    kept, so multiplicity is preserved by the sorted multiset built each round.
    """

    vertices: dict = field(default_factory=dict)
    incidences: list = field(default_factory=list)


def _relabel(signatures, keys):
    """Map each key's signature to a small int, shared across the union.

    Distinct signatures are ordered by ``repr`` (a total, deterministic order
    independent of set-iteration order / hash seed) and numbered ``0..k-1``.
    Equal signatures — on either side — collapse to the same token, which is
    exactly what makes gold and pred tokens comparable.
    """
    distinct = sorted({signatures[k] for k in keys}, key=repr)
    token = {sig: i for i, sig in enumerate(distinct)}
    return {k: token[signatures[k]] for k in keys}


def _num_partitions(color, keys):
    return len({color[k] for k in keys})


def wl_tokens(gold_graph, pred_graph, *, mode="tie_break", rounds=None):
    """Refine both graphs jointly and return per-side ``{id: color_int}``.

    Args:
        gold_graph: the gold-side :class:`RefGraph`.
        pred_graph: the pred-side :class:`RefGraph`, built independently and
            identically.
        mode: ``"tie_break"`` seeds every vertex with one constant color
            (structure only); ``"blend"`` seeds with the vertex's own
            exactly-comparable scalar label so refinement builds on top of the
            hard attributes.
        rounds: cap on refinement rounds; ``None`` runs to a stable partition.
            Hard-capped at ``|V_union|`` regardless.

    Returns:
        ``(gold_tokens, pred_tokens)``, each a ``dict`` from vertex id to its
        stable integer color. Tokens are comparable *across* the two dicts:
        equal tokens mean the two vertices are structurally indistinguishable
        (same WL color); only such pairs are candidates for the structural
        tie-break / blend bonus.
    """
    graphs = {0: gold_graph, 1: pred_graph}

    # Adjacency over the disjoint union; each vertex is keyed by (side, id).
    adjacency = {}
    for side, graph in graphs.items():
        for vid in graph.vertices:
            adjacency[(side, vid)] = []
    for side, graph in graphs.items():
        for edge in graph.incidences:
            src_key = (side, edge.src)
            dst_key = (side, edge.dst)
            if src_key in adjacency:
                adjacency[src_key].append(("out", edge.role, edge.label, dst_key))
            if dst_key in adjacency:
                adjacency[dst_key].append(("in", edge.role, edge.label, src_key))

    keys = list(adjacency)
    if not keys:
        return {}, {}

    def seed(key):
        side, vid = key
        if mode == "blend":
            return graphs[side].vertices.get(vid, ())
        return ()

    color = _relabel({k: ("c0", seed(k)) for k in keys}, keys)
    parts = _num_partitions(color, keys)

    max_rounds = len(keys) if rounds is None else min(int(rounds), len(keys))
    for _ in range(max_rounds):
        signatures = {}
        for key in keys:
            neighborhood = sorted(
                (
                    (direction, role, label, color[neighbor])
                    for (direction, role, label, neighbor) in adjacency[key]
                ),
                key=repr,
            )
            signatures[key] = (color[key], tuple(neighborhood))
        new_color = _relabel(signatures, keys)
        new_parts = _num_partitions(new_color, keys)
        color = new_color
        if new_parts == parts:
            break
        parts = new_parts

    gold_tokens = {vid: color[(0, vid)] for vid in gold_graph.vertices}
    pred_tokens = {vid: color[(1, vid)] for vid in pred_graph.vertices}
    return gold_tokens, pred_tokens

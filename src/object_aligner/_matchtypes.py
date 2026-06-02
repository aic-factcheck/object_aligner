"""Match-tree dataclasses and the per-call alignment context.

Frozen result nodes (:class:`MatchItem`, :class:`MatchList`,
:class:`MatchDict`) plus the internal bookkeeping dataclasses
(:class:`_IdScope`, :class:`_AlignContext`) and the :func:`to_python_value`
numpy-to-builtin coercion helper. Split out of ``object_aligner.py`` so the rest
of the package (and the sibling subsystems) can import these leaf types without
pulling in the aligner itself.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatchItem:
    """Leaf node of the alignment tree, produced for a single primitive value.

    Returned by [`ObjectAligner.align`][object_aligner.ObjectAligner.align]
    whenever the schema type is `string`, `number`, `integer`, or `boolean`.
    Also produced for `idScope` and `ref` primitives.

    Attributes:
        score: Similarity in `[0, 1]` between `gold` and `pred`.
        gold: The gold (reference) primitive value.
        pred: The predicted primitive value.
        kind: `"id"` for `idScope` fields, `"ref"` for `ref` fields,
            `"null"` when one or both of `gold`/`pred` is `None`, and
            `""` otherwise. Surfaced as `"marker"` in the debug tree when
            non-empty.
        confidence: Per-pair stability score in `[0, 1]` from the
            enclosing Hungarian matching (key-pair confidence for keys
            of a `MatchDict`, item-pair confidence for items of a
            `MatchList` with `kind="reorder"`). `1.0` for leaves whose
            parent did not run a Hungarian assignment, and `1.0` for
            excess/missing pairs. Populated only when the owning
            `ObjectAligner` was constructed with `compute_confidence=True`.
        aux: Optional per-leaf metadata that downstream consumers
            (repair, feedback, describe) read without re-computing it.
            Currently set only for `kind="ref"` leaves: a mapping
            `{"mapped_pred": <pred-space id> | None}` carrying the
            bijection-resolved pred-space id corresponding to `gold`.
            `None` for all other leaves and for ref leaves that ran in
            a masked context. Surfaced as `"aux"` in the debug tree
            when non-`None`.
    """

    score: float
    gold: Any
    pred: Any
    kind: str = ""
    confidence: float = 1.0
    aux: Mapping[str, Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class MatchList:
    """Alignment node for a list-typed schema.

    Returned by [`ObjectAligner.align`][object_aligner.ObjectAligner.align]
    when the schema type is `array`. The `children` list is the per-element
    alignment in alignment order; for `order: "align"` lists this is the
    order produced by the Hungarian matching, not the original `pred` order.

    Attributes:
        score: Aggregate similarity in `[0, 1]`.
        children: Per-child match nodes (`MatchItem`, `MatchList`, or
            `MatchDict`) in alignment order.
        kind: `"reorder"` / `"fixed"` / `"prefix"` / `"combined"` based on
            the list aggregator selected by the schema; `""` if not set.
            Consumed by attribution/repair to pick the per-aggregator α
            schedule.
        confidence: Aggregate stability score in `[0, 1]`. For
            `kind="reorder"` lists, the mean per-pair confidence from the
            Hungarian matching over matched children; for
            `kind="fixed"`/`"prefix"`/`"combined"` lists, the mean of
            child confidences. `1.0` when `compute_confidence=False`.
    """

    score: float
    children: list = field(default_factory=list)
    kind: str = ""
    confidence: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class MatchDict:
    """Alignment node for an object-typed schema.

    Returned by [`ObjectAligner.align`][object_aligner.ObjectAligner.align]
    when the schema type is `object`. `children` maps a key match
    (`MatchItem` over the keys) to the corresponding value match.

    Attributes:
        score: Aggregate similarity in `[0, 1]`.
        children: Mapping from a key-level `MatchItem` to the value-level
            match (`MatchItem`, `MatchList`, or `MatchDict`).
        confidence: Aggregate stability score in `[0, 1]` blending the
            key-pair Hungarian confidence and the value-subtree confidence
            using the same `keyImportance` / `valueImportance` weights
            used for `score`. `1.0` when `compute_confidence=False`.
    """

    score: float
    children: dict = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass
class _IdScope:
    scope: str
    primitive_type: str
    definer_schema_path: tuple
    definer_array_path: tuple
    definer_node: Any
    ref_paths: list = field(default_factory=list)
    ref_nodes: list = field(default_factory=list)
    degraded: bool = False


@dataclass
class _AlignContext:
    """Per-call state for an ``align()`` invocation.

    Threaded through the recursive ``_align_*`` methods so a single
    ``ObjectAligner`` instance can be shared across threads — each call
    creates its own context and never touches the instance's state.
    """
    current_mappings: dict = field(default_factory=dict)
    pred_ids: dict = field(default_factory=dict)
    gold_ids: dict = field(default_factory=dict)
    pred_excess_ids: dict = field(default_factory=dict)
    mask_scope: Any = None
    mask_all_refs: bool = False
    skip_validation: bool = False
    compute_confidence: bool = False
    confidence_method: str = "margin"
    confidence_temperature: float = 8.0


def to_python_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [to_python_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_python_value(item) for item in value)
    if isinstance(value, dict):
        return {to_python_value(key): to_python_value(item) for key, item in value.items()}
    return value

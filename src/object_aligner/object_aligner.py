import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from jsonschema import ValidationError
from jsonschema.validators import validator_for
from rapidfuzz.distance import DamerauLevenshtein, Indel, Jaro, JaroWinkler, LCSseq, Levenshtein, OSA
from scipy.optimize import linear_sum_assignment


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


# Cross-module imports are placed here — *after* the MatchItem/MatchList/
# MatchDict dataclass definitions — so that the modules below (which all
# import those names at their own top level) can resolve them off the
# partially-loaded ``object_aligner.object_aligner`` module.
#
# The one genuinely-circular pair (``describe._walk`` needs the Match types
# at render time) is still broken by a lazy import inside that function.
from object_aligner.attribution import AttributionResult, tree_walk_attribution
from object_aligner.repair import RepairResult, generate_repairs
from object_aligner.describe import (
    DescriptionResult,
    _VALID_STYLES as _DESCRIBE_VALID_STYLES,
    merge_description_templates,
    render_description,
    render_validation_error as render_description_validation_error,
)
from object_aligner.feedback import (
    FeedbackResult,
    _VALID_STYLES as _FEEDBACK_VALID_STYLES,
    merge_feedback_templates,
    render_feedback,
)


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


def path2str(p):
    return "/" + "/".join([str(d) for d in p])


def _schema_allows_type(schema_type, name):
    """Return True if ``schema_type`` (a JSON Schema ``type`` value, which
    may be a string or a list of strings) permits ``name``. Used by the
    alignment dispatcher to accept union types such as
    ``type: ["string", "null"]`` alongside plain ``type: "string"``."""
    if schema_type == name:
        return True
    if isinstance(schema_type, list) and name in schema_type:
        return True
    return False


def similarity_exact(a, b):
    return float(a == b)


def similarity_num_inv_diff(a, b):
    diff = abs(a - b)
    score = 1 / (1 + diff)
    return score


def similarity_string_jaro(a, b):
    return Jaro.normalized_similarity(a, b)


def similarity_string_jaro_winkler(a, b):
    return JaroWinkler.normalized_similarity(a, b)


def similarity_string_levenshtein(a, b):
    return Levenshtein.normalized_similarity(a, b)


def similarity_string_damerau_levenshtein(a, b):
    return DamerauLevenshtein.normalized_similarity(a, b)


def similarity_string_osa(a, b):
    return OSA.normalized_similarity(a, b)


def similarity_string_indel(a, b):
    return Indel.normalized_similarity(a, b)


def similarity_string_lcsseq(a, b):
    return LCSseq.normalized_similarity(a, b)


BUILTIN_STRING_METRICS = {
    "exact": similarity_exact,
    "jaro": similarity_string_jaro,
    "jaro_winkler": similarity_string_jaro_winkler,
    "levenshtein": similarity_string_levenshtein,
    "damerau_levenshtein": similarity_string_damerau_levenshtein,
    "osa": similarity_string_osa,
    "indel": similarity_string_indel,
    "lcsseq": similarity_string_lcsseq,
}
BUILTIN_NUMBER_METRICS = {
    "exact": similarity_exact,
    "invdiff": similarity_num_inv_diff,
}
SUPPORTED_CUSTOM_METRIC_TYPES = frozenset({"string", "number", "integer"})


def _with_confidence(match, confidence):
    """Return a copy of ``match`` with its top-level ``confidence`` set.

    Used at Hungarian sites to attach a per-pair confidence to an already
    fully-built recursive match object without re-aligning. All three
    Match dataclasses are frozen, so a plain field write is impossible;
    we lean on ``dataclasses.replace`` which the frozen API officially
    supports.
    """
    from dataclasses import replace
    return replace(match, confidence=float(confidence))


def _hungarian_confidence(
    similarity_matrix,
    row_ind,
    col_ind,
    n,
    m,
    *,
    method="margin",
    temperature=8.0,
):
    """Per-pair and node-level confidence from a Hungarian assignment.

    Reads the similarity matrix that was passed to
    :func:`scipy.optimize.linear_sum_assignment` and returns a stability
    score in ``[0, 1]`` for each chosen pair plus an aggregate scalar.
    Excess/missing pairs (one side is zero-padding, i.e. ``row >= n`` or
    ``col >= m``) score ``1.0`` — there is no ambiguity, the item is
    simply unmatched.

    The ``"margin"`` method computes the symmetric clipped margin against
    the row's and column's second-best entries; the ``"entropy"`` method
    softmaxes each row over its first ``m`` columns and returns
    ``1 - H / log m``.

    Args:
        similarity_matrix: ``(d, d)`` matrix used by the Hungarian site,
            with ``d = max(n, m)``.
        row_ind, col_ind: Output of ``linear_sum_assignment(-similarity_matrix)``.
        n, m: Real (unpadded) gold and pred sizes.
        method: ``"margin"`` or ``"entropy"``.
        temperature: Softmax temperature ``β`` (entropy method only).

    Returns:
        A tuple ``(pair_confidences, node_confidence)`` where
        ``pair_confidences`` is a 1-D ``np.ndarray`` of length
        ``len(row_ind)`` aligned with ``zip(row_ind, col_ind)`` and
        ``node_confidence`` is a Python ``float`` — the mean of the
        confidences over genuinely matched pairs (both sides in range),
        or ``1.0`` if no such pair exists.
    """
    k = len(row_ind)
    pair_conf = np.ones(k, dtype=np.float64)
    matched_confs = []
    if method == "margin":
        for idx in range(k):
            ri = int(row_ind[idx])
            ci = int(col_ind[idx])
            if ri >= n or ci >= m:
                continue
            row = similarity_matrix[ri, :m]
            col = similarity_matrix[:n, ci]
            chosen = float(similarity_matrix[ri, ci])
            if m > 1:
                row_others = np.delete(row, ci)
                m_row = chosen - float(np.max(row_others))
            else:
                m_row = chosen
            if n > 1:
                col_others = np.delete(col, ri)
                m_col = chosen - float(np.max(col_others))
            else:
                m_col = chosen
            c = 0.5 * (min(1.0, max(0.0, m_row)) + min(1.0, max(0.0, m_col)))
            pair_conf[idx] = c
            matched_confs.append(c)
    elif method == "entropy":
        beta = float(temperature)
        log_m = np.log(m) if m > 1 else 1.0
        for idx in range(k):
            ri = int(row_ind[idx])
            ci = int(col_ind[idx])
            if ri >= n or ci >= m:
                continue
            if m == 1:
                pair_conf[idx] = 1.0
                matched_confs.append(1.0)
                continue
            row = similarity_matrix[ri, :m].astype(np.float64)
            shifted = beta * (row - np.max(row))
            exp_row = np.exp(shifted)
            denom = float(np.sum(exp_row))
            if denom <= 0.0:
                pair_conf[idx] = 1.0
                matched_confs.append(1.0)
                continue
            p = exp_row / denom
            with np.errstate(divide="ignore", invalid="ignore"):
                ent_terms = np.where(p > 0.0, -p * np.log(p), 0.0)
            H = float(np.sum(ent_terms))
            c = 1.0 - H / log_m
            c = min(1.0, max(0.0, c))
            pair_conf[idx] = c
            matched_confs.append(c)
    else:
        raise ValueError(f"unknown confidence method: {method!r}")
    node_conf = float(np.mean(matched_confs)) if matched_confs else 1.0
    return pair_conf, node_conf


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


class ObjectAligner:
    """Aligns a gold object against a predicted object under a schema.

    `ObjectAligner` is the entry point for every alignment operation in the
    library: scoring (`metric`), tree extraction (`align`), per-path deficit
    decomposition (`attribute`), structured repair operations (`repair`), and
    prompt-optimizer feedback (`feedback`). One instance is bound to one
    schema; it is safe to share across threads because per-call state lives
    in an internal `_AlignContext` that is created per call.

    The constructor validates the schema, builds the JSON Schema validator
    used by `metric()`, resolves the custom-metric registry, and discovers
    the dependency order between `idScope` declarations.

    See [`docs/concepts.md`](../concepts.md) for the architectural tour.
    """

    def __init__(
        self,
        schema,
        *,
        custom_metrics=None,
        generate_description=False,
        description_templates=None,
        description_style="default",
        generate_feedback=False,
        feedback_templates=None,
        feedback_style="gepa",
        dominant_fraction_threshold=0.60,
        warn_on_ambiguous_mapping=False,
        compute_confidence=False,
        confidence_method="margin",
        confidence_entropy_temperature=8.0,
    ):
        """Initialize an `ObjectAligner` for the given schema.

        Args:
            schema: JSON-Schema-inspired dict describing the structure and
                scoring of the objects to be aligned. See
                [`docs/schema_reference.md`](../schema_reference.md) for the
                full list of supported keywords.
            custom_metrics: Optional mapping from schema type
                (`"string"` / `"number"` / `"integer"`) to a mapping of
                `name -> callable(gold, pred) -> float in [0, 1]`. Integer
                schemas use built-in number metrics and fall back to custom
                `number` metrics unless overridden by a custom `integer`
                metric with the same name. Boolean scoring is exact-only
                and cannot be customized.
            generate_description: Default for the `generate_description`
                parameter of `metric()`. When truthy, `metric()` returns
                include a `"description"` key. Accepts `True` / `False` /
                `"full"`; see [`docs/describe.md`](../describe.md).
            description_templates: Optional partial override of the
                packaged description templates. Unknown keys or unknown
                placeholders raise `ValueError`.
            description_style: One of the registered description styles
                (default `"default"`). Controls whether the renderer
                produces prose (`"default"`) or empty `.text` plus
                populated `.entries` (`"json"`).
            generate_feedback: Default for the `generate_feedback`
                parameter of `metric()`. When truthy, `metric()` returns
                include a `"feedback"` key. Accepts `True` / `False` /
                `"full"`; see [`docs/feedback.md`](../feedback.md).
            feedback_templates: Optional partial override of the packaged
                feedback templates. Validated against the same allowlist
                machinery as `description_templates`.
            feedback_style: One of the registered feedback styles
                (default `"gepa"`). Controls phrasing and synthesis-line
                shape.
            dominant_fraction_threshold: Fraction of the deficit that one
                op kind must own for the feedback synthesis line to switch
                between the "single dominant" and "mixed" phrasings.
                Defaults to `0.60`.
            warn_on_ambiguous_mapping: If `True`, emit a `UserWarning`
                whenever the Hungarian-derived id mapping for an `idScope`
                is non-unique because of tied costs. Off by default.
            compute_confidence: If `True`, populate the `confidence` field
                on `MatchItem` / `MatchList` / `MatchDict` from the
                similarity matrix used at each Hungarian site
                (`order: "align"` lists and dict-key matching). Default
                `False` keeps `confidence == 1.0` everywhere, which
                preserves byte-identical output for `feedback()` and
                `describe()` under default flags. See
                [`docs/confidence.md`](../confidence.md).
            confidence_method: `"margin"` (default) or `"entropy"`. Selects
                the per-pair confidence formula. Margin is a fast linear
                pass over the similarity matrix; entropy softmaxes each
                row and reports `1 - H / log m`.
            confidence_entropy_temperature: Softmax temperature `β` used
                only when `confidence_method="entropy"`. Defaults to `8.0`,
                which puts a Jaro 0.95 vs 0.80 at roughly a 3:1
                probability ratio on `[0, 1]`-bounded similarities.

        Raises:
            ValueError: If `custom_metrics` contains an unsupported schema
                type, collides with a built-in metric name,
                `feedback_style` is not a registered style, or
                `description_style` is not a registered style.
            jsonschema.SchemaError: If `schema` itself is not a valid JSON
                Schema.
        """
        self.schema = schema

        self.generate_description_default = generate_description
        if description_style not in _DESCRIBE_VALID_STYLES:
            raise ValueError(
                f"description_style must be one of {_DESCRIBE_VALID_STYLES!r}, "
                f"got {description_style!r}"
            )
        self.description_style_default = description_style
        self.description_templates = merge_description_templates(description_templates)

        self.generate_feedback_default = generate_feedback
        if feedback_style not in _FEEDBACK_VALID_STYLES:
            raise ValueError(
                f"feedback_style must be one of {_FEEDBACK_VALID_STYLES!r}, "
                f"got {feedback_style!r}"
            )
        self.feedback_style_default = feedback_style
        try:
            self.dominant_fraction_threshold_default = float(
                dominant_fraction_threshold
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                "dominant_fraction_threshold must be a real number"
            ) from e
        self.feedback_templates = merge_feedback_templates(feedback_templates)

        self._primitive_metrics = self._build_primitive_metric_registry(custom_metrics)
        self._warn_on_ambiguous_mapping = bool(warn_on_ambiguous_mapping)

        if confidence_method not in ("margin", "entropy"):
            raise ValueError(
                f"confidence_method must be 'margin' or 'entropy', got {confidence_method!r}"
            )
        try:
            ct = float(confidence_entropy_temperature)
        except (TypeError, ValueError) as e:
            raise ValueError("confidence_entropy_temperature must be a real number") from e
        if not (ct > 0.0) or not np.isfinite(ct):
            raise ValueError(
                f"confidence_entropy_temperature must be finite and > 0, got {confidence_entropy_temperature!r}"
            )
        self._compute_confidence = bool(compute_confidence)
        self._confidence_method = confidence_method
        self._confidence_temperature = ct

        self._validate_importance_sums(schema)
        self._validate_null_scores(schema)
        self._id_scopes, self._scope_order = self._collect_id_scopes(schema)
        self._validator = validator_for(schema)(schema)

    def _build_primitive_metric_registry(self, custom_metrics):
        if custom_metrics is None:
            custom_metrics = {}
        if not isinstance(custom_metrics, Mapping):
            raise TypeError("custom_metrics must be a mapping of schema types to metric-name mappings")

        unsupported_types = sorted(set(custom_metrics) - SUPPORTED_CUSTOM_METRIC_TYPES)
        if unsupported_types:
            raise ValueError(f"Unsupported custom metric types: {unsupported_types}")

        custom_by_type = {schema_type: {} for schema_type in SUPPORTED_CUSTOM_METRIC_TYPES}
        builtin_by_type = {
            "string": BUILTIN_STRING_METRICS,
            "number": BUILTIN_NUMBER_METRICS,
            "integer": BUILTIN_NUMBER_METRICS,
        }

        for schema_type, metrics in custom_metrics.items():
            if not isinstance(metrics, Mapping):
                raise TypeError(
                    f'custom_metrics["{schema_type}"] must be a mapping of metric names to callables'
                )

            duplicate_names = sorted(set(metrics) & set(builtin_by_type[schema_type]))
            if duplicate_names:
                raise ValueError(
                    f'custom_metrics["{schema_type}"] contains names that collide with built-in metrics: {duplicate_names}'
                )

            for metric_name, metric in metrics.items():
                if not isinstance(metric_name, str):
                    raise TypeError(
                        f'custom_metrics["{schema_type}"] metric names must be strings, got {type(metric_name)!r}'
                    )
                if not callable(metric):
                    raise TypeError(
                        f'custom_metrics["{schema_type}"]["{metric_name}"] must be callable'
                    )
                custom_by_type[schema_type][metric_name] = metric

        return {
            "string": {**BUILTIN_STRING_METRICS, **custom_by_type["string"]},
            "number": {**BUILTIN_NUMBER_METRICS, **custom_by_type["number"]},
            "integer": {
                **BUILTIN_NUMBER_METRICS,
                **custom_by_type["number"],
                **custom_by_type["integer"],
            },
        }

    def _resolve_primitive_metric(self, schema_type, metric_name):
        metrics = self._primitive_metrics[schema_type]
        if metric_name not in metrics:
            raise ValueError(
                f'Unsupported score "{metric_name}" for schema type "{schema_type}". '
                f'Supported scores: {sorted(metrics)}'
            )
        return metrics[metric_name]

    @staticmethod
    def _iter_schema_children(node, schema_path):
        """Yield ``(child_node, child_schema_path)`` for each declared
        descent edge (``properties`` / ``items`` / ``prefixItems``)."""
        if not isinstance(node, dict):
            return
        if "properties" in node and isinstance(node["properties"], dict):
            for k, v in node["properties"].items():
                yield v, schema_path + [("properties", k)]
        if "items" in node:
            yield node["items"], schema_path + [("items",)]
        if "prefixItems" in node and isinstance(node["prefixItems"], list):
            for i, sub in enumerate(node["prefixItems"]):
                yield sub, schema_path + [("prefixItems", i)]

    @staticmethod
    def _validate_importance_sums(schema):
        """Pre-walk the schema and raise ValueError on any zero-sum
        importance/weight configuration that would later divide by zero
        (or produce NaN) in alignment. Walks properties/items/prefixItems
        same as the id-scope walker."""

        def walk(node, schema_path):
            if not isinstance(node, dict):
                return
            t = node.get("type")
            if t == "object":
                ki = node.get("keyImportance", 0.0)
                vi = node.get("valueImportance", 1.0)
                if ki + vi == 0:
                    raise ValueError(
                        f"keyImportance and valueImportance cannot both be zero at {path2str([str(e) for e in schema_path])}"
                    )
                if "properties" in node and isinstance(node["properties"], dict):
                    vws = [
                        p.get("valueWeight", 1.0)
                        for p in node["properties"].values()
                        if isinstance(p, dict)
                    ]
                    if vws and sum(vws) == 0:
                        raise ValueError(
                            f"valueWeights across properties must not sum to zero at {path2str([str(e) for e in schema_path])}"
                        )
            if t == "array":
                if "prefixItems" in node and "items" in node:
                    pi = node.get("prefixImportance")
                    ri = node.get("restImportance")
                    # presence is enforced lazily in _align_lists; only check
                    # the zero-sum case when both are explicitly supplied.
                    if pi is not None and ri is not None and pi + ri == 0:
                        raise ValueError(
                            f"prefixImportance and restImportance cannot both be zero at {path2str([str(e) for e in schema_path])}"
                        )
                if "prefixItems" in node and "prefixWeights" in node:
                    pw = node["prefixWeights"]
                    if isinstance(pw, (list, tuple)) and sum(pw) == 0:
                        raise ValueError(
                            f"prefixWeights must not sum to zero at {path2str([str(e) for e in schema_path])}"
                        )
            for child, child_path in ObjectAligner._iter_schema_children(node, schema_path):
                walk(child, child_path)

        walk(schema, [])

    @staticmethod
    def _validate_null_scores(schema):
        """Pre-walk the schema and raise ``ValueError`` if any ``nullScore``
        is not a real number in ``[0, 1]``. Walks ``properties`` / ``items``
        / ``prefixItems`` via `_iter_schema_children`."""

        def walk(node, schema_path):
            if not isinstance(node, dict):
                return
            if "nullScore" in node:
                ns = node["nullScore"]
                if isinstance(ns, bool) or not isinstance(ns, Real):
                    raise ValueError(
                        f"'nullScore' must be a real number, got {type(ns).__name__} "
                        f"at {path2str([str(e) for e in schema_path])}"
                    )
                if not (0.0 <= float(ns) <= 1.0):
                    raise ValueError(
                        f"'nullScore' must be in [0, 1], got {ns} "
                        f"at {path2str([str(e) for e in schema_path])}"
                    )
            for child, child_path in ObjectAligner._iter_schema_children(node, schema_path):
                walk(child, child_path)

        walk(schema, [])

    @staticmethod
    def _enclosing_array_path(schema_path):
        for i in range(len(schema_path) - 1, -1, -1):
            edge = schema_path[i]
            if edge == ("items",) or (isinstance(edge, tuple) and edge and edge[0] == "prefixItems"):
                return tuple(schema_path[: i + 1])
        return None

    def _collect_id_scopes(self, schema):
        scopes = {}
        pending_refs = []  # list of (scope_name, schema_path, node, primitive_type)
        ref_allowed_types = ("string", "integer", "number")

        def walk(node, schema_path):
            if not isinstance(node, dict):
                return

            id_scope_name = node.get("idScope")
            ref_name = node.get("ref")
            if id_scope_name is not None and ref_name is not None:
                raise ValueError(
                    f"Schema node at {path2str([str(e) for e in schema_path])} declares both 'idScope' and 'ref'"
                )

            if id_scope_name is not None:
                if not isinstance(id_scope_name, str) or not id_scope_name:
                    raise ValueError(
                        f"'idScope' must be a non-empty string at {path2str([str(e) for e in schema_path])}"
                    )
                t = node.get("type")
                if t not in ref_allowed_types:
                    raise TypeError(
                        f"'idScope' is only allowed on string/integer/number primitives; got type={t!r} at {path2str([str(e) for e in schema_path])}"
                    )
                if id_scope_name in scopes:
                    raise ValueError(
                        f"'idScope' '{id_scope_name}' declared more than once"
                    )
                array_path = self._enclosing_array_path(schema_path)
                if array_path is None:
                    raise ValueError(
                        f"'idScope' '{id_scope_name}' at {path2str([str(e) for e in schema_path])} must be inside an array (so its definers form an alignable list)"
                    )
                if "score" in node or "threshold" in node:
                    warnings.warn(
                        f"'score'/'threshold' on idScope '{id_scope_name}' are ignored (id fields are matched symbolically)",
                        UserWarning,
                    )
                scopes[id_scope_name] = _IdScope(
                    scope=id_scope_name,
                    primitive_type=t,
                    definer_schema_path=tuple(schema_path),
                    definer_array_path=array_path,
                    definer_node=node,
                )
                return

            if ref_name is not None:
                if not isinstance(ref_name, str) or not ref_name:
                    raise ValueError(
                        f"'ref' must be a non-empty string at {path2str([str(e) for e in schema_path])}"
                    )
                t = node.get("type")
                if t not in ref_allowed_types:
                    raise TypeError(
                        f"'ref' is only allowed on string/integer/number primitives; got type={t!r} at {path2str([str(e) for e in schema_path])}"
                    )
                if "score" in node or "threshold" in node:
                    warnings.warn(
                        f"'score'/'threshold' on ref '{ref_name}' are ignored (refs are matched via the inferred id mapping)",
                        UserWarning,
                    )
                pending_refs.append((ref_name, tuple(schema_path), node, t))
                return

            # Recurse
            for child, child_path in self._iter_schema_children(node, schema_path):
                walk(child, child_path)

        walk(schema, [])

        for name, path, node, t in pending_refs:
            if name not in scopes:
                raise ValueError(
                    f"'ref' to undefined idScope '{name}' at {path2str([str(e) for e in path])}"
                )
            scope = scopes[name]
            if scope.primitive_type != t:
                raise ValueError(
                    f"Type mismatch for scope '{name}': idScope is '{scope.primitive_type}' but ref at {path2str([str(e) for e in path])} is '{t}'"
                )
            scope.ref_paths.append(path)
            scope.ref_nodes.append(node)

        scope_order = self._toposort_scopes(scopes)
        return scopes, scope_order

    @staticmethod
    def _toposort_scopes(scopes):
        # Edge A -> B means: a ref to B appears under A's definer subtree, so B must be resolved first.
        deps = {name: set() for name in scopes}
        for b_name, b_scope in scopes.items():
            for ref_path in b_scope.ref_paths:
                for a_name, a_scope in scopes.items():
                    if a_name == b_name:
                        continue
                    arr = a_scope.definer_array_path
                    if len(ref_path) > len(arr) and ref_path[: len(arr)] == arr:
                        deps[a_name].add(b_name)

        order = []
        remaining = {a: set(s) for a, s in deps.items()}
        ready = sorted(a for a, s in remaining.items() if not s)
        while ready:
            node = ready.pop(0)
            order.append(node)
            new_ready = []
            for other in scopes:
                if other in order or other in ready:
                    continue
                if node in remaining[other]:
                    remaining[other].discard(node)
                if not remaining[other]:
                    new_ready.append(other)
            new_ready.sort()
            ready.extend(new_ready)

        cycle_members = sorted(a for a in scopes if a not in order)
        if cycle_members:
            warnings.warn(
                f"Cycle in idScope dependency graph: {cycle_members}. Falling back to property-only alignment for cycle members.",
                UserWarning,
            )
            for a in cycle_members:
                scopes[a].degraded = True
                order.append(a)

        return tuple(order)

    def _validate_metric_score(self, schema_type, metric_name, score):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(
                f'Metric "{metric_name}" for schema type "{schema_type}" must return a real number in [0, 1], got {score!r}'
            )

        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f'Metric "{metric_name}" for schema type "{schema_type}" must return a score in [0, 1], got {score!r}'
            )
        return score

    @staticmethod
    def _walk_data(data, schema_path_edges, data_path=()):
        """Yield ``(value, data_path)`` for each runtime position reachable along schema_path_edges.

        ``("properties", k)`` descends into dict key, ``("items",)`` iterates a list,
        ``("prefixItems", n)`` indexes a list. Silently skips when the runtime shape
        does not match (caller is responsible for raising if needed).
        """
        if not schema_path_edges:
            yield data, data_path
            return
        edge = schema_path_edges[0]
        rest = schema_path_edges[1:]
        if isinstance(edge, tuple) and edge and edge[0] == "properties":
            key = edge[1]
            if not isinstance(data, dict) or key not in data:
                return
            yield from ObjectAligner._walk_data(data[key], rest, data_path + (key,))
        elif edge == ("items",):
            if not isinstance(data, list):
                return
            for i, item in enumerate(data):
                yield from ObjectAligner._walk_data(item, rest, data_path + (i,))
        elif isinstance(edge, tuple) and edge and edge[0] == "prefixItems":
            idx = edge[1]
            if not isinstance(data, list) or idx >= len(data):
                return
            yield from ObjectAligner._walk_data(data[idx], rest, data_path + (idx,))

    def _validate_referential(self, gold):
        """Validate idScope uniqueness and ref resolvability in gold. Return per-scope id sets."""
        gold_ids = {}
        for scope_name in self._scope_order:
            scope = self._id_scopes[scope_name]
            suffix = scope.definer_schema_path[len(scope.definer_array_path):]
            seen = set()
            for item, item_path in self._walk_data(gold, scope.definer_array_path):
                for val, val_path in self._walk_data(item, suffix, item_path):
                    if val in seen:
                        raise ValidationError(
                            f"Duplicate id {val!r} in idScope '{scope_name}'",
                            path=list(val_path),
                        )
                    seen.add(val)
            gold_ids[scope_name] = seen
        for scope_name, scope in self._id_scopes.items():
            valid = gold_ids[scope_name]
            for ref_path in scope.ref_paths:
                for val, val_path in self._walk_data(gold, ref_path):
                    if val not in valid:
                        raise ValidationError(
                            f"Dangling ref to '{scope_name}': value={val!r} not defined",
                            path=list(val_path),
                        )
        return gold_ids

    def _collect_pred_ids(self, pred):
        """Tolerantly collect pred id sets per scope (first-wins on duplicates; no errors)."""
        pred_ids = {}
        for scope_name, scope in self._id_scopes.items():
            suffix = scope.definer_schema_path[len(scope.definer_array_path):]
            seen = set()
            for item, _ in self._walk_data(pred, scope.definer_array_path):
                for val, _ in self._walk_data(item, suffix):
                    if val not in seen:
                        seen.add(val)
            pred_ids[scope_name] = seen
        return pred_ids

    @staticmethod
    def _get_schema_node(schema, edges):
        node = schema
        for edge in edges:
            if edge == ("items",):
                node = node["items"]
            elif edge[0] == "properties":
                node = node["properties"][edge[1]]
            elif edge[0] == "prefixItems":
                node = node["prefixItems"][edge[1]]
        return node

    def _derive_id_mappings(self, gold, pred, ctx):
        """Derive per-scope bijection in topological order using _align_helper under masking flags."""
        mappings = {}
        pred_excess = {}
        for scope_name in self._scope_order:
            scope = self._id_scopes[scope_name]
            ctx.mask_scope = scope_name
            ctx.mask_all_refs = scope.degraded
            try:
                mapping, excess = self._derive_single_scope(gold, pred, scope, ctx)
            finally:
                ctx.mask_scope = None
                ctx.mask_all_refs = False
            mappings[scope_name] = mapping
            pred_excess[scope_name] = excess
            ctx.current_mappings[scope_name] = mapping
        return mappings, pred_excess

    def _derive_single_scope(self, gold, pred, scope, ctx):
        item_schema = self._get_schema_node(self.schema, scope.definer_array_path)
        suffix = scope.definer_schema_path[len(scope.definer_array_path):]

        gold_items = list(self._walk_data(gold, scope.definer_array_path))
        pred_items = list(self._walk_data(pred, scope.definer_array_path))
        n, m = len(gold_items), len(pred_items)

        def extract_id(item):
            for val, _ in self._walk_data(item, suffix):
                return val
            return None

        gold_id_list = [extract_id(it) for it, _ in gold_items]
        pred_id_list = [extract_id(it) for it, _ in pred_items]

        if n == 0 and m == 0:
            return {}, set()

        d = max(n, m)
        cost = np.zeros((d, d))
        for i in range(n):
            for j in range(m):
                g_item = gold_items[i][0]
                p_item = pred_items[j][0]
                try:
                    aligned = self._align_helper(g_item, p_item, item_schema, ctx)
                    cost[i][j] = aligned["match"].score
                except (TypeError, ValueError, KeyError):
                    cost[i][j] = 0.0

        row_ind, col_ind = linear_sum_assignment(-cost)

        self._maybe_warn_ambiguity(cost, n, m, gold_id_list, scope.scope)

        mapping = {}
        matched_pred_ids = set()
        for ri, ci in zip(row_ind, col_ind):
            if ri < n:
                g_id = gold_id_list[ri]
                if g_id is None:
                    continue
                if ci < m:
                    p_id = pred_id_list[ci]
                    if p_id is None or p_id in matched_pred_ids:
                        mapping[g_id] = None
                    else:
                        mapping[g_id] = p_id
                        matched_pred_ids.add(p_id)
                else:
                    mapping[g_id] = None

        all_pred_ids = {pid for pid in pred_id_list if pid is not None}
        excess = all_pred_ids - matched_pred_ids
        return mapping, excess

    def _maybe_warn_ambiguity(self, cost, n, m, gold_id_list, scope_name):
        if not self._warn_on_ambiguous_mapping or n < 2:
            return
        seen = {}
        ambiguous = set()
        for i in range(n):
            key = tuple(cost[i, :m].tolist()) if m > 0 else ()
            if key in seen:
                ambiguous.add(gold_id_list[seen[key]])
                ambiguous.add(gold_id_list[i])
            else:
                seen[key] = i
        ambiguous.discard(None)
        if ambiguous:
            warnings.warn(
                f"Ambiguous mapping in idScope '{scope_name}': gold ids {sorted(ambiguous, key=repr)} could be paired multiple ways with equal cost; arbitrary assignment used.",
                UserWarning,
            )

    def _align_primitive(self, g, p, schema, *, schema_type, default_score):
        score_type = schema.get("score", default_score)
        threshold = schema.get("threshold", 0.0)
        scoref = self._resolve_primitive_metric(schema_type, score_type)
        score = self._validate_metric_score(schema_type, score_type, scoref(g, p))
        score = 0.0 if score < threshold else score
        return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p)}

    def _list_norm(self, aligned_gold, aligned_pred, schema):
        ignore_excess = schema.get("ignoreExcess", False)
        ignore_missing = schema.get("ignoreMissing", False)
        D = 0
        for ag, ap in zip(aligned_gold, aligned_pred):
            if ag is None and ignore_excess:
                continue
            if ap is None and ignore_missing:
                continue
            D += 1
        return D

    def _align_numbers(self, g, p, schema):
        # Resolve the primitive type when the schema declares a union
        # (e.g. `type: ["integer", "null"]`): pick the numeric branch and
        # ignore "null", which only governs the null-aware leaf above.
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            primitive_type = "integer" if "integer" in schema_type else "number"
        else:
            primitive_type = schema_type
        return self._align_primitive(g, p, schema, schema_type=primitive_type, default_score="invdiff")

    def _align_strings(self, g, p, schema):
        return self._align_primitive(g, p, schema, schema_type="string", default_score="jaro")

    def _align_lists_reorder(self, gold, pred, schema, ctx):
        n, m = len(gold), len(pred)
        d = max(n, m)

        if d == 0:
            return {"gold": gold, "pred": pred, "match": MatchList(score=1.0, children=[], kind="reorder")}

        similarity_matrix = np.zeros((d, d))
        subs = np.empty((n, m), dtype=object)

        for i in range(n):
            for j in range(m):
                aligned = self._align_helper(gold[i], pred[j], schema["items"], ctx)
                similarity_matrix[i][j] = aligned["match"].score
                subs[i][j] = (aligned["gold"], aligned["pred"], aligned["match"])

        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        if ctx.compute_confidence:
            pair_conf, node_conf = _hungarian_confidence(
                similarity_matrix, row_ind, col_ind, n, m,
                method=ctx.confidence_method,
                temperature=ctx.confidence_temperature,
            )
        else:
            pair_conf = None
            node_conf = 1.0

        aligned_gold = []
        aligned_pred = []
        aligned_scores = []
        for i in range(len(row_ind)):
            ri, ci = row_ind[i], col_ind[i]
            similarity = similarity_matrix[ri][ci]
            pc = float(pair_conf[i]) if pair_conf is not None else 1.0
            if ri < n and ci < m:
                sg, sp, sscore = subs[ri][ci]
                if sscore.score > 0.0:
                    aligned_gold.append(sg)
                    aligned_pred.append(sp)
                    if pair_conf is not None:
                        aligned_scores.append(_with_confidence(sscore, pc))
                    else:
                        aligned_scores.append(sscore)
                else:
                    if sp is not None:
                        aligned_gold.append(None)
                        aligned_pred.append(sp)
                        aligned_scores.append(MatchItem(0.0, gold=None, pred=sp))
                    if sg is not None:
                        aligned_gold.append(sg)
                        aligned_pred.append(None)
                        aligned_scores.append(MatchItem(0.0, gold=sg, pred=None))
            elif ri < n:
                aligned_gold.append(gold[ri])
                aligned_pred.append(None)
                aligned_scores.append(MatchItem(0.0, gold=gold[ri], pred=None))
            elif ci < m:
                aligned_gold.append(None)
                aligned_pred.append(pred[ci])
                aligned_scores.append(MatchItem(0.0, gold=None, pred=pred[ci]))

        D = self._list_norm(aligned_gold, aligned_pred, schema)
        if D == 0:
            score = 1.0 if len(aligned_scores) == 0 else 0.0
        else:
            score = float(np.sum([s.score for s in aligned_scores])) / D
        score = max(0.0, min(1.0, score))
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_scores, kind="reorder", confidence=node_conf)}

    def _align_lists_fixed(self, gold, pred, schema, ctx):
        n, m = len(gold), len(pred)
        if n == 0 and m == 0:
            return {"gold": [], "pred": [], "match": MatchList(score=1.0, children=[], kind="fixed")}
        if n == 0:
            return {
                "gold": [None] * m,
                "pred": pred,
                "match": MatchList(score=0.0, children=[MatchItem(score=0.0, gold=None, pred=e) for e in pred], kind="fixed"),
            }
        if m == 0:
            return {
                "gold": gold,
                "pred": [None] * n,
                "match": MatchList(score=0.0, children=[MatchItem(score=0.0, gold=e, pred=None) for e in gold], kind="fixed"),
            }
        dp = np.zeros((n + 1, m + 1))
        subs = np.zeros((n + 1, m + 1), dtype=object)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                aligned = self._align_helper(gold[i - 1], pred[j - 1], schema["items"], ctx)
                match = dp[i - 1][j - 1] + aligned["match"].score
                skip_pred = dp[i - 1][j]
                skip_gold = dp[i][j - 1]

                dp[i][j] = max(match, skip_pred, skip_gold)

                if dp[i][j] == match:
                    subs[i][j] = (aligned["gold"], aligned["pred"], aligned["match"])
                elif dp[i][j] == skip_pred:
                    subs[i][j] = (gold[i - 1], None, MatchItem(0.0, gold=gold[i - 1], pred=None))
                else:
                    subs[i][j] = (None, pred[j - 1], MatchItem(0.0, gold=None, pred=pred[j - 1]))

        aligned_gold = []
        aligned_pred = []
        aligned_scores = []

        i, j = n, m
        while i > 0 and j > 0:
            sg, sp, sscore = subs[i][j]

            if sscore.score > 0.0:
                aligned_gold.append(sg)
                aligned_pred.append(sp)
                aligned_scores.append(sscore)
                # A scored "match" cell always consumes one element from
                # each side, even if either value is literally ``None``
                # (a nullable item schema). Decrement both unconditionally.
                i -= 1
                j -= 1
            else:
                if sp is not None:
                    aligned_gold.append(None)
                    aligned_pred.append(sp)
                    aligned_scores.append(MatchItem(0.0, gold=None, pred=sp))
                if sg is not None:
                    aligned_gold.append(sg)
                    aligned_pred.append(None)
                    aligned_scores.append(MatchItem(0.0, gold=sg, pred=None))
                if sg is not None:
                    i -= 1
                if sp is not None:
                    j -= 1

        if i > 0:
            if j > 0:
                raise RuntimeError("internal: DP traceback ended with both indices positive")
            while i > 0:
                aligned_gold.append(subs[i][1][0])
                aligned_pred.append(None)
                aligned_scores.append(MatchItem(0.0, gold=subs[i][1][0], pred=None))
                i -= 1
        if j > 0:
            if i > 0:
                raise RuntimeError("internal: DP traceback ended with both indices positive")
            while j > 0:
                aligned_gold.append(None)
                aligned_pred.append(subs[1][j][1])
                aligned_scores.append(MatchItem(0.0, gold=None, pred=subs[1][j][1]))
                j -= 1

        aligned_gold.reverse()
        aligned_pred.reverse()
        aligned_scores.reverse()

        if len(aligned_gold) != len(aligned_pred):
            raise RuntimeError("internal: aligned gold/pred length mismatch")
        D = self._list_norm(aligned_gold, aligned_pred, schema)
        if D == 0:
            score = 1.0 if len(aligned_scores) == 0 else 0.0
        else:
            score = float(dp[n][m]) / D
        score = max(0.0, min(1.0, score))
        if ctx.compute_confidence and aligned_scores:
            node_conf = float(np.mean([float(s.confidence) for s in aligned_scores]))
        else:
            node_conf = 1.0
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_scores, kind="fixed", confidence=node_conf)}

    def _align_lists_prefix(self, gold, pred, schema, ctx):
        aligned_gold = []
        aligned_pred = []
        aligned_matches = []
        prefix_items = schema["prefixItems"]
        for i, sub_schema in enumerate(prefix_items):
            g_present = i < len(gold)
            p_present = i < len(pred)
            if g_present and p_present:
                aligned = self._align_helper(gold[i], pred[i], sub_schema, ctx)
                aligned_gold.append(aligned["gold"])
                aligned_pred.append(aligned["pred"])
                aligned_matches.append(aligned["match"])
            elif g_present:
                aligned_gold.append(gold[i])
                aligned_pred.append(None)
                aligned_matches.append(MatchItem(score=0.0, gold=gold[i], pred=None))
            elif p_present:
                aligned_gold.append(None)
                aligned_pred.append(pred[i])
                aligned_matches.append(MatchItem(score=0.0, gold=None, pred=pred[i]))
            else:
                # Both sides shorter than len(prefixItems). Emit a sentinel
                # pair so the denominator (sum of prefixWeights) stays
                # correct; renderer skips dual-None children silently.
                aligned_gold.append(None)
                aligned_pred.append(None)
                aligned_matches.append(MatchItem(score=0.0, gold=None, pred=None))
        weights = np.array(schema.get("prefixWeights", np.ones(len(aligned_matches))), dtype=np.float64)
        weights = weights / weights.sum()
        score = float(np.sum([e.score * w for e, w in zip(aligned_matches, weights)]))
        score = max(0.0, min(1.0, score))
        if ctx.compute_confidence and aligned_matches:
            node_conf = float(np.sum([float(e.confidence) * w for e, w in zip(aligned_matches, weights)]))
        else:
            node_conf = 1.0
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_matches, kind="prefix", confidence=node_conf)}

    def _align_lists(self, g, p, schema, ctx):
        if "prefixItems" not in schema and "items" not in schema:
            raise ValueError("array schema must declare 'prefixItems' or 'items'")

        rets = []
        prefix_len = 0
        if "prefixItems" in schema:
            prefix_len = len(schema["prefixItems"])
            rets.append(self._align_lists_prefix(g[:prefix_len], p[:prefix_len], schema, ctx))

        if "items" in schema:
            ordering = schema.get("order", "fixed")
            if ordering not in ("align", "fixed"):
                raise ValueError(f"'order' must be 'align' or 'fixed', got {ordering!r}")
            if ordering == "fixed":
                rets.append(self._align_lists_fixed(g[prefix_len:], p[prefix_len:], schema, ctx))
            else:
                rets.append(self._align_lists_reorder(g[prefix_len:], p[prefix_len:], schema, ctx))

        if len(rets) == 1:
            return rets[0]
        if "prefixImportance" not in schema or "restImportance" not in schema:
            raise ValueError("'prefixImportance' and 'restImportance' must both be set when both 'prefixItems' and 'items' are present")
        pi = schema["prefixImportance"]
        ri = schema["restImportance"]
        impsum = pi + ri
        pi /= impsum
        ri /= impsum
        gold = rets[0]["gold"] + rets[1]["gold"]
        pred = rets[0]["pred"] + rets[1]["pred"]
        pscore = rets[0]["match"].score
        rscore = rets[1]["match"].score
        score = pi * pscore + ri * rscore
        children = rets[0]["match"].children + rets[1]["match"].children
        if ctx.compute_confidence:
            pconf = float(rets[0]["match"].confidence)
            rconf = float(rets[1]["match"].confidence)
            combined_conf = pi * pconf + ri * rconf
            combined_conf = min(1.0, max(0.0, combined_conf))
        else:
            combined_conf = 1.0
        return {"gold": gold, "pred": pred, "match": MatchList(score=score, children=children, kind="combined", confidence=combined_conf)}

    def _align_dicts(self, g, p, schema, ctx):
        match_key = schema.get("keyScore", "jaro")
        if match_key not in ("exact", "jaro"):
            raise ValueError(f"'keyScore' must be 'exact' or 'jaro', got {match_key!r}")
        key_threshold = schema.get("keyThreshold", 0.0)
        scoref = similarity_exact if match_key == "exact" else similarity_string_jaro

        key_importance = schema.get("keyImportance", 0.0)
        value_importance = schema.get("valueImportance", 1.0)

        gkeys = list(g.keys())
        pkeys = list(p.keys())

        if len(gkeys) == 0 and len(pkeys) == 0:
            return {"gold": {}, "pred": {}, "match": MatchDict(score=1.0, children={})}

        n, m = len(gkeys), len(pkeys)
        d = max(n, m)
        similarity_matrix = np.zeros((d, d))
        for i in range(n):
            for j in range(m):
                sc = scoref(gkeys[i], pkeys[j])
                similarity_matrix[i][j] = 0.0 if sc < key_threshold else sc
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        if ctx.compute_confidence:
            pair_conf, keys_node_conf = _hungarian_confidence(
                similarity_matrix, row_ind, col_ind, n, m,
                method=ctx.confidence_method,
                temperature=ctx.confidence_temperature,
            )
        else:
            pair_conf = None
            keys_node_conf = 1.0

        aligned_gkeys = []
        aligned_pkeys = []
        aligned_key_scores = []
        aligned_key_confs = []
        for i in range(len(row_ind)):
            ri, ci = row_ind[i], col_ind[i]
            pc = float(pair_conf[i]) if pair_conf is not None else 1.0
            if ri < n and ci < m:
                sg, sp, sim = gkeys[ri], pkeys[ci], similarity_matrix[ri][ci]
                if sim > 0:
                    aligned_gkeys.append(sg)
                    aligned_pkeys.append(sp)
                    aligned_key_scores.append(sim)
                    aligned_key_confs.append(pc)
                else:
                    if sp is not None:
                        aligned_gkeys.append(None)
                        aligned_pkeys.append(sp)
                        aligned_key_scores.append(sim)
                        aligned_key_confs.append(1.0)
                    if sg is not None:
                        aligned_gkeys.append(sg)
                        aligned_pkeys.append(None)
                        aligned_key_scores.append(sim)
                        aligned_key_confs.append(1.0)
            elif ri < n:
                aligned_gkeys.append(gkeys[ri])
                aligned_pkeys.append(None)
                aligned_key_scores.append(0.0)
                aligned_key_confs.append(1.0)
            elif ci < m:
                aligned_gkeys.append(None)
                aligned_pkeys.append(pkeys[ci])
                aligned_key_scores.append(0.0)
                aligned_key_confs.append(1.0)

        keys_score = float(np.mean(aligned_key_scores))

        aligned_values = []
        value_weights = []
        for gk, pk in zip(aligned_gkeys, aligned_pkeys):
            ag = g.get(gk)
            ap = p.get(pk)
            if gk is None and pk is None:
                raise ValueError("dict alignment produced a key pair with both sides None (None used as a dict key?)")
            if gk is not None and pk is not None:
                aux_schema = schema["properties"][gk]
                value_weights.append(schema["properties"][gk].get("valueWeight", 1.0))

                if type(ag) is not type(ap):
                    if ag is None or ap is None:
                        # Null-aware: delegate to _align_helper, which routes
                        # through `_align_null` and consults this property's
                        # `nullScore` (default 0.0).
                        aligned_value = self._align_helper(ag, ap, aux_schema, ctx)
                    elif ctx.skip_validation:
                        # Soft-zero under skip_validation: caller opted into
                        # looser semantics, so type-mismatched values score 0
                        # rather than raising.
                        aligned_value = {"gold": ag, "pred": ap, "match": MatchItem(score=0.0, gold=ag, pred=ap)}
                    else:
                        raise TypeError(f"dict value types differ for key {gk!r}: {type(ag).__name__} vs {type(ap).__name__}")
                else:
                    aligned_value = self._align_helper(ag, ap, aux_schema, ctx)
            else:
                aligned_value = {"gold": ag, "pred": ap, "match": MatchItem(score=0.0, gold=ag, pred=ap)}
                value_weights.append(1.0)
            aligned_values.append(aligned_value)
        value_scores = np.array([e["match"].score for e in aligned_values])
        value_weights = np.array(value_weights) / np.sum(value_weights)
        values_score = float(np.sum(value_weights * value_scores))

        aligned_gold = {}
        aligned_pred = {}
        children = {}
        for gk, pk, aligned_value, key_score, key_conf in zip(
            aligned_gkeys, aligned_pkeys, aligned_values, aligned_key_scores, aligned_key_confs
        ):
            if gk is not None:
                aligned_gold[gk] = aligned_value["gold"]
            if pk is not None:
                aligned_pred[pk] = aligned_value["pred"]
            children[
                MatchItem(score=key_score, gold=gk, pred=pk, confidence=key_conf)
            ] = aligned_value["match"]

        score = float(key_importance * keys_score + value_importance * values_score) / (key_importance + value_importance)
        score = max(0.0, min(1.0, score))

        if ctx.compute_confidence:
            child_confs = np.array([float(e["match"].confidence) for e in aligned_values], dtype=np.float64)
            vw = np.array(value_weights, dtype=np.float64) if not isinstance(value_weights, np.ndarray) else value_weights
            values_conf = float(np.sum(vw * child_confs)) if len(child_confs) else 1.0
            dict_conf = (key_importance * keys_node_conf + value_importance * values_conf) / (key_importance + value_importance)
            dict_conf = min(1.0, max(0.0, dict_conf))
        else:
            dict_conf = 1.0

        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchDict(score=score, children=children, confidence=dict_conf)}

    def _align_booleans(self, g, p, schema):
        score = similarity_exact(g, p)
        return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p)}

    def _align_null(self, g, p, schema):
        # Both-None always scores 1.0; asymmetric uses the schema's
        # `nullScore` (default 0.0). Range/type already validated at
        # construction by `_validate_null_scores`.
        if g is None and p is None:
            score = 1.0
        else:
            score = float(schema.get("nullScore", 0.0)) if isinstance(schema, dict) else 0.0
        return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p, kind="null")}

    def _align_helper(self, g, p, schema, ctx):
        if isinstance(schema, dict):
            if schema.get("idScope") is not None:
                return {"gold": g, "pred": p, "match": MatchItem(score=1.0, gold=g, pred=p, kind="id")}
            ref_scope = schema.get("ref")
            if ref_scope is not None:
                # Two mask cases handled by this branch:
                #  * `ctx.mask_scope == ref_scope`: we're computing the cost
                #    matrix for this scope's own definers, so refs into the
                #    scope must score 1.0 to avoid self-referential
                #    bootstrapping.
                #  * `ctx.mask_all_refs`: this scope is a cycle member and is
                #    being aligned property-only; treat all refs as 1.0.
                if ctx.mask_all_refs or ref_scope == ctx.mask_scope:
                    return {"gold": g, "pred": p, "match": MatchItem(score=1.0, gold=g, pred=p, kind="ref")}
                # Defensive fallback: under correct topological ordering,
                # any non-masked scope referenced here should already be in
                # `ctx.current_mappings`. Reaching this line implies either a
                # cycle that escaped detection or a future regression in
                # `_collect_id_scopes` / `_toposort_scopes`.
                if ref_scope not in ctx.current_mappings:
                    return {"gold": g, "pred": p, "match": MatchItem(score=1.0, gold=g, pred=p, kind="ref")}
                mapping = ctx.current_mappings[ref_scope]
                pred_ids = ctx.pred_ids.get(ref_scope, set())
                mapped = mapping.get(g)
                if mapped is None or p not in pred_ids:
                    score = 0.0
                elif mapped == p:
                    score = 1.0
                else:
                    score = 0.0
                return {"gold": g, "pred": p, "match": MatchItem(
                    score=score, gold=g, pred=p, kind="ref",
                    aux={"mapped_pred": mapped},
                )}
        if g is None or p is None:
            return self._align_null(g, p, schema)
        schema_type = schema.get("type")
        if isinstance(g, bool):
            if not _schema_allows_type(schema_type, "boolean"):
                raise TypeError(f"schema declares type={schema_type!r} but data is bool")
            aligned = self._align_booleans(g, p, schema)
        elif isinstance(g, (int, float)):
            if not (_schema_allows_type(schema_type, "number") or _schema_allows_type(schema_type, "integer")):
                raise TypeError(f"schema declares type={schema_type!r} but data is {type(g).__name__}")
            aligned = self._align_numbers(g, p, schema)
        elif isinstance(g, str):
            if not _schema_allows_type(schema_type, "string"):
                raise TypeError(f"schema declares type={schema_type!r} but data is str")
            aligned = self._align_strings(g, p, schema)
        elif isinstance(g, list):
            if not _schema_allows_type(schema_type, "array"):
                raise TypeError(f"schema declares type={schema_type!r} but data is list")
            aligned = self._align_lists(g, p, schema, ctx)
        elif isinstance(g, dict):
            if not _schema_allows_type(schema_type, "object"):
                raise TypeError(f"schema declares type={schema_type!r} but data is dict")
            aligned = self._align_dicts(g, p, schema, ctx)
        else:
            raise TypeError(f"unsupported data type: {type(g).__name__}")

        return aligned

    def _serialize_match_debug(self, aligned):
        if isinstance(aligned, MatchItem):
            out = {
                "kind": "item",
                "score": float(aligned.score),
                "gold": to_python_value(aligned.gold),
                "pred": to_python_value(aligned.pred),
            }
            if aligned.kind:
                out["marker"] = aligned.kind
            if abs(float(aligned.confidence) - 1.0) > 1e-12:
                out["confidence"] = float(aligned.confidence)
            if aligned.aux is not None:
                out["aux"] = {k: to_python_value(v) for k, v in aligned.aux.items()}
            return out

        if isinstance(aligned, MatchList):
            out = {
                "kind": "list",
                "score": float(aligned.score),
                "children": [self._serialize_match_debug(child) for child in aligned.children],
            }
            if abs(float(aligned.confidence) - 1.0) > 1e-12:
                out["confidence"] = float(aligned.confidence)
            return out

        if isinstance(aligned, MatchDict):
            out = {
                "kind": "dict",
                "score": float(aligned.score),
                "children": [
                    {
                        "key": self._serialize_match_debug(key),
                        "value": self._serialize_match_debug(child),
                    }
                    for key, child in aligned.children.items()
                ],
            }
            if abs(float(aligned.confidence) - 1.0) > 1e-12:
                out["confidence"] = float(aligned.confidence)
            return out

        raise TypeError(f"Unknown match instance: {aligned!r}")

    def align(self, g, p, skip_validation=False):
        """Align gold to pred and return the match tree.

        Builds a per-call context, so concurrent calls on the same
        `ObjectAligner` instance are safe. See
        [`docs/concepts.md`](../concepts.md) for the algorithmic flow.

        Args:
            g: Gold (reference) object. Must match the schema unless
                `skip_validation=True`.
            p: Predicted object. Must match the schema unless
                `skip_validation=True`.
            skip_validation: If `True`, skip JSON Schema validation of both
                inputs (caller is responsible for ensuring well-formedness).

        Returns:
            A frozen `MatchItem`, `MatchList`, or `MatchDict` whose `.score`
            is in `[0, 1]`. The concrete type is selected by the schema's
            top-level type.

        Raises:
            jsonschema.ValidationError: If validation is enabled and either
                input fails.
            TypeError: If `g` and `p` are not of the same Python type.
        """
        match, _ = self._align_with_ctx(g, p, skip_validation=skip_validation)
        return match

    def _align_with_ctx(self, g, p, *, skip_validation=False):
        """Internal: return both the match tree and the per-call ``_AlignContext``.

        ``repair()`` uses this to access ``ctx.current_mappings`` for ``ref_fix``
        repair ops. The public ``align()`` discards the context.
        """
        if type(g) is not type(p) and g is not None and p is not None:
            raise TypeError(f"gold and pred must be the same type, got {type(g).__name__} and {type(p).__name__}")
        if not skip_validation:
            self._validator.validate(g)
            self._validator.validate(p)
        ctx = _AlignContext(
            skip_validation=bool(skip_validation),
            compute_confidence=self._compute_confidence,
            confidence_method=self._confidence_method,
            confidence_temperature=self._confidence_temperature,
        )
        if self._id_scopes:
            ctx.gold_ids = self._validate_referential(g)
            ctx.pred_ids = self._collect_pred_ids(p)
            ctx.current_mappings, ctx.pred_excess_ids = self._derive_id_mappings(g, p, ctx)
        match = self._align_helper(g, p, self.schema, ctx)["match"]
        return match, ctx

    def attribute(
        self,
        gold,
        pred,
        *,
        granularity="leaf",
        include_empty_positions=False,
        skip_validation=False,
    ):
        """Decompose the score deficit into per-path contributions.

        Runs `align()` internally then walks the resulting match tree. See
        [`docs/attribution.md`](../attribution.md) for examples.

        Args:
            gold: Gold (reference) object.
            pred: Predicted object.
            granularity: `"leaf"` (default) emits one entry per leaf; other
                values control subtree-level rollups.
            include_empty_positions: When `True`, list-position gaps with
                zero contribution are emitted as explicit entries.
            skip_validation: If `True`, skip JSON Schema validation of
                `gold` and `pred`.

        Returns:
            `AttributionResult` whose `entries` is ranked by per-path
            contribution. If `pred` fails validation, returns an empty
            result with `score=0.0`.

        Raises:
            jsonschema.ValidationError: If `gold` fails validation
                (validation enabled).
        """
        if not skip_validation:
            self._validator.validate(gold)
            try:
                self._validator.validate(pred)
            except ValidationError:
                return AttributionResult(
                    score=0.0,
                    entries=(),
                    granularity=granularity,
                    total_contribution=0.0,
                    residual=-1.0,
                )

        match_tree = self.align(gold, pred, skip_validation=True)
        return tree_walk_attribution(
            match_tree,
            self.schema,
            granularity=granularity,
            include_empty_positions=include_empty_positions,
        )

    def attribute_from_match(
        self,
        match_tree,
        *,
        granularity="leaf",
        include_empty_positions=False,
    ):
        """Attribute an already-computed match tree without re-running `align()`.

        Useful when you have already produced a match tree (e.g., to derive
        both `metric()` and `attribute()` outputs without aligning twice).

        Args:
            match_tree: A match tree returned by `align()`.
            granularity: See `attribute()`.
            include_empty_positions: See `attribute()`.

        Returns:
            `AttributionResult` — same shape as `attribute()`.
        """
        return tree_walk_attribution(
            match_tree,
            self.schema,
            granularity=granularity,
            include_empty_positions=include_empty_positions,
        )

    def repair(
        self,
        gold,
        pred,
        *,
        granularity="leaf",
        min_contribution=0.0,
        skip_validation=False,
        rank_by="score_delta",
        include_pairing_ambiguous=False,
        ambiguity_threshold=0.30,
    ):
        """Emit a ranked list of scored repair ops for `(gold, pred)`.

        Each `RepairOp` carries an estimated `score_delta` — how much of the
        deficit `1 - S` applying the op would close. v1 implements the
        *approximate* flavor only; deltas come from the tree-walk math.
        See [`docs/repair.md`](../repair.md) for examples.

        Args:
            gold: Gold (reference) object.
            pred: Predicted object.
            granularity: `"leaf"` (default) or subtree-level rollups.
            min_contribution: Drop ops whose `score_delta` falls below this
                threshold.
            skip_validation: If `True`, skip JSON Schema validation.
            rank_by: Sort key for the returned ops. One of
                `"score_delta"` (default — current behavior),
                `"expected_gain"` (`score_delta × confidence`), or
                `"confidence"`. See [`docs/confidence.md`](../confidence.md).
            include_pairing_ambiguous: If `True`, append a
                `pairing_ambiguous` diagnostic op for every Hungarian
                container whose `confidence` falls below
                `ambiguity_threshold`. Diagnostic only — not applied by
                `RepairResult.apply_to`. Off by default.
            ambiguity_threshold: Confidence threshold for the
                `pairing_ambiguous` walker. Defaults to `0.30`.

        Returns:
            `RepairResult` whose `ops` is ranked by `rank_by`. If
            `pred` fails validation, returns an empty result with
            `score=0.0`.

        Raises:
            jsonschema.ValidationError: If `gold` fails validation
                (validation enabled).
        """
        if not skip_validation:
            self._validator.validate(gold)
            try:
                self._validator.validate(pred)
            except ValidationError:
                return RepairResult(
                    score=0.0,
                    ops=(),
                    granularity=granularity,
                    total_delta=0.0,
                    residual=-1.0,
                )

        match_tree, ctx = self._align_with_ctx(gold, pred, skip_validation=True)
        return generate_repairs(
            match_tree,
            self.schema,
            gold,
            pred,
            ctx.current_mappings,
            granularity=granularity,
            min_contribution=min_contribution,
            rank_by=rank_by,
            include_pairing_ambiguous=include_pairing_ambiguous,
            ambiguity_threshold=ambiguity_threshold,
        )

    def repair_from_match(
        self,
        match_tree,
        gold,
        pred,
        mappings=None,
        *,
        granularity="leaf",
        min_contribution=0.0,
        rank_by="score_delta",
        include_pairing_ambiguous=False,
        ambiguity_threshold=0.30,
    ):
        """Generate repair ops from an already-computed match tree.

        Args:
            match_tree: A match tree returned by `align()`.
            gold: Gold object (used to read source values for `add` ops).
            pred: Predicted object (used to read source values for
                `remove` ops).
            mappings: The `ctx.current_mappings` dict from the align-time
                context, needed for `ref_fix` ops. If your schema has no
                `ref` fields it can be `None` or `{}`. If `match_tree` was
                produced by `align()` and your schema declares `ref`
                fields, prefer `repair()` (which captures mappings
                automatically).
            granularity: See `repair()`.
            min_contribution: See `repair()`.
            rank_by: See `repair()`.
            include_pairing_ambiguous: See `repair()`.
            ambiguity_threshold: See `repair()`.

        Returns:
            `RepairResult` — same shape as `repair()`.
        """
        return generate_repairs(
            match_tree,
            self.schema,
            gold,
            pred,
            mappings,
            granularity=granularity,
            min_contribution=min_contribution,
            rank_by=rank_by,
            include_pairing_ambiguous=include_pairing_ambiguous,
            ambiguity_threshold=ambiguity_threshold,
        )

    def feedback(
        self,
        gold,
        pred,
        *,
        top_k=5,
        min_score_delta=0.0,
        style=None,
        include_synthesis_line=True,
        include_metadata=False,
        dominant_fraction_threshold=None,
        granularity="leaf",
        skip_validation=False,
        rank_by="score_delta",
        include_pairing_ambiguous=False,
        ambiguity_threshold=0.30,
    ):
        """Render prompt-optimizer feedback for `(gold, pred)`.

        Aligns once internally and walks the repair tree; never invokes an
        LLM. The output is deterministic and template-driven. See
        [`docs/feedback.md`](../feedback.md) for examples.

        Args:
            gold: Gold (reference) object.
            pred: Predicted object.
            top_k: Maximum number of feedback entries to render.
            min_score_delta: Drop entries whose `score_delta` falls below
                this threshold before ranking.
            style: Override the constructor `feedback_style`. `None`
                defers to the instance default.
            include_synthesis_line: When `True`, append a one-line
                synthesis at the end (e.g., "Single dominant error:
                year extractor."). Phrasing controlled by
                `dominant_fraction_threshold`.
            include_metadata: When `True`, include op-kind / α-chain
                metadata in each entry (for ablation work).
            dominant_fraction_threshold: Override the constructor
                threshold for switching between "single dominant" and
                "mixed" synthesis-line phrasing. `None` defers to the
                instance default.
            granularity: See `attribute()` / `repair()`.
            skip_validation: If `True`, skip JSON Schema validation.
            rank_by: `"score_delta"` (default), `"expected_gain"`, or
                `"confidence"`. See [`docs/confidence.md`](../confidence.md).
                Default preserves byte-identical output of earlier
                releases.
            include_pairing_ambiguous: If `True`, surface a "Diagnostic
                notes" trailing section listing Hungarian containers
                whose `confidence` fell below `ambiguity_threshold`.
                Off by default.
            ambiguity_threshold: Confidence threshold for the diagnostic
                walker. Default `0.30`.

        Returns:
            `FeedbackResult` whose `text` is suitable for pasting into a
            DSPy / GEPA / TextGrad reflection slot. On validation failure
            of `pred`, returns a degenerate result with `score=0.0` and a
            rendered validation-error message as `text`.

        Raises:
            jsonschema.ValidationError: If `gold` fails validation
                (validation enabled).
        """
        if not skip_validation:
            self._validator.validate(gold)
            try:
                self._validator.validate(pred)
            except ValidationError as e:
                resolved_style = style or self.feedback_style_default
                error_text = self.feedback_templates[
                    "feedback.validation_error"
                ].format(path=path2str(e.path), message=e.message)
                return FeedbackResult(
                    score=0.0,
                    text=error_text,
                    entries=(),
                    style=resolved_style,
                    truncated=False,
                    n_total_ops=0,
                    error_breakdown={},
                )

        match_tree, ctx = self._align_with_ctx(gold, pred, skip_validation=True)
        repair_result = self.repair_from_match(
            match_tree, gold, pred, ctx.current_mappings,
            granularity=granularity,
            rank_by=rank_by,
            include_pairing_ambiguous=include_pairing_ambiguous,
            ambiguity_threshold=ambiguity_threshold,
        )
        return render_feedback(
            repair_result,
            top_k=top_k,
            min_score_delta=min_score_delta,
            style=style or self.feedback_style_default,
            include_synthesis_line=include_synthesis_line,
            include_metadata=include_metadata,
            templates=self.feedback_templates,
            dominant_fraction_threshold=(
                self.dominant_fraction_threshold_default
                if dominant_fraction_threshold is None
                else dominant_fraction_threshold
            ),
        )

    def feedback_from_match(
        self,
        match_tree,
        gold,
        pred,
        mappings=None,
        *,
        top_k=5,
        min_score_delta=0.0,
        style=None,
        include_synthesis_line=True,
        include_metadata=False,
        dominant_fraction_threshold=None,
        granularity="leaf",
        rank_by="score_delta",
        include_pairing_ambiguous=False,
        ambiguity_threshold=0.30,
    ):
        """Render feedback from an already-computed match tree.

        Args:
            match_tree: A match tree returned by `align()`.
            gold: Gold object.
            pred: Predicted object.
            mappings: `ctx.current_mappings` from the align-time context.
                Can be `None` or `{}` if your schema has no `ref` fields.
            top_k: See `feedback()`.
            min_score_delta: See `feedback()`.
            style: See `feedback()`.
            include_synthesis_line: See `feedback()`.
            include_metadata: See `feedback()`.
            dominant_fraction_threshold: See `feedback()`.
            granularity: See `feedback()`.
            rank_by: See `feedback()`.
            include_pairing_ambiguous: See `feedback()`.
            ambiguity_threshold: See `feedback()`.

        Returns:
            `FeedbackResult` — same shape as `feedback()`.
        """
        repair_result = self.repair_from_match(
            match_tree, gold, pred, mappings,
            granularity=granularity,
            rank_by=rank_by,
            include_pairing_ambiguous=include_pairing_ambiguous,
            ambiguity_threshold=ambiguity_threshold,
        )
        return render_feedback(
            repair_result,
            top_k=top_k,
            min_score_delta=min_score_delta,
            style=style or self.feedback_style_default,
            include_synthesis_line=include_synthesis_line,
            include_metadata=include_metadata,
            templates=self.feedback_templates,
            dominant_fraction_threshold=(
                self.dominant_fraction_threshold_default
                if dominant_fraction_threshold is None
                else dominant_fraction_threshold
            ),
        )

    def describe(
        self,
        gold,
        pred,
        *,
        style=None,
        skip_validation=False,
        show_confidence=False,
        include_ambiguous=False,
        ambiguity_threshold=0.30,
    ):
        """Render a plain-English description of `(gold, pred)`.

        Aligns once internally and walks the match tree; never invokes an
        LLM. The output is deterministic and template-driven. See
        [`docs/describe.md`](../describe.md) for examples.

        Args:
            gold: Gold (reference) object.
            pred: Predicted object.
            style: Override the constructor `description_style`. `None`
                defers to the instance default.
            skip_validation: If `True`, skip JSON Schema validation.
            show_confidence: If `True`, append a banded confidence
                suffix to every per-node line whose `confidence` falls
                below `0.70`. Requires the owning aligner to have been
                constructed with `compute_confidence=True`. Default
                preserves byte-identical output of earlier releases.
                See [`docs/confidence.md`](../confidence.md).
            include_ambiguous: If `True`, emit an extra entry above
                every Hungarian-paired container whose `confidence`
                falls below `ambiguity_threshold`. Off by default.
            ambiguity_threshold: Confidence threshold for the
                ambiguous-entry emission. Default `0.30`.

        Returns:
            `DescriptionResult` whose `text` is the rendered indented
            prose (or `""` in `"json"` style) and `entries` is a
            traversal-ordered list of `DescriptionEntry`. On validation
            failure of `pred`, returns a degenerate result with
            `score=0.0` and a rendered validation-error message as
            `text`.

        Raises:
            jsonschema.ValidationError: If `gold` fails validation
                (validation enabled).
        """
        if not skip_validation:
            self._validator.validate(gold)
            try:
                self._validator.validate(pred)
            except ValidationError as e:
                return render_description_validation_error(
                    e, self.description_templates,
                )

        match_tree, _ctx = self._align_with_ctx(gold, pred, skip_validation=True)
        return render_description(
            match_tree,
            style=style or self.description_style_default,
            templates=self.description_templates,
            show_confidence=show_confidence,
            include_ambiguous=include_ambiguous,
            ambiguity_threshold=ambiguity_threshold,
        )

    def describe_from_match(
        self,
        match_tree,
        *,
        style=None,
        show_confidence=False,
        include_ambiguous=False,
        ambiguity_threshold=0.30,
    ):
        """Render a description from an already-computed match tree.

        Args:
            match_tree: A match tree returned by `align()`.
            style: See `describe()`.
            show_confidence: See `describe()`.
            include_ambiguous: See `describe()`.
            ambiguity_threshold: See `describe()`.

        Returns:
            `DescriptionResult` — same shape as `describe()`.
        """
        return render_description(
            match_tree,
            style=style or self.description_style_default,
            templates=self.description_templates,
            show_confidence=show_confidence,
            include_ambiguous=include_ambiguous,
            ambiguity_threshold=ambiguity_threshold,
        )

    def metric(
        self,
        gold,
        pred,
        debug=False,
        generate_description=None,
        generate_feedback=None,
    ):
        """Score `pred` against `gold` and return a result dict.

        Always validates `gold`. If `pred` fails validation, returns
        `{"score": 0.0}` (with a `"description"` / `"feedback"` describing
        the error when those are enabled). Safe to call concurrently on a
        single instance.

        Args:
            gold: Gold (reference) object. Must pass schema validation.
            pred: Predicted object. Validation failure here is non-fatal:
                a score of `0.0` is returned.
            debug: When `True`, the returned dict also contains a
                `"debug"` key with a structured alignment tree built out
                of basic Python container/scalar types.
            generate_description: Per-call override of the constructor
                default. `None` defers to the instance setting. Accepts
                `False` / `True` (renders description as a string under
                `"description"`) or `"full"` (a structured dict — the
                same shape as `DescriptionResult.to_dict()`). Any other
                value raises `ValueError`. See
                [`docs/describe.md`](../describe.md).
            generate_feedback: Per-call override of the constructor
                default. `None` uses the instance setting. Accepts
                `False` / `True` (renders feedback as a string under
                `"feedback"`) or `"full"` (a structured dict — the same
                shape as `FeedbackResult.to_dict()`). Any other value
                raises `ValueError`. See
                [`docs/feedback.md`](../feedback.md).

        Returns:
            Dict with required key `"score"` (Python `float` in `[0, 1]`)
            and optional keys `"description"` (str or dict),
            `"feedback"` (str or dict), and `"debug"` (dict) based on
            the flags.

        Raises:
            jsonschema.ValidationError: If `gold` fails validation.
            ValueError: If `generate_description` or `generate_feedback`
                is not `None` / `False` / `True` / `"full"`.
        """
        self._validator.validate(gold)

        description_mode = (
            self.generate_description_default
            if generate_description is None
            else generate_description
        )
        if description_mode not in (False, True, "full"):
            raise ValueError(
                "generate_description must be None, False, True, or 'full'; "
                f"got {generate_description!r}"
            )
        feedback_mode = (
            self.generate_feedback_default
            if generate_feedback is None
            else generate_feedback
        )
        if feedback_mode not in (False, True, "full"):
            raise ValueError(
                "generate_feedback must be None, False, True, or 'full'; "
                f"got {generate_feedback!r}"
            )

        try:
            self._validator.validate(pred)
        except ValidationError as e:
            result = {"score": 0.0}
            if description_mode:
                dr = render_description_validation_error(
                    e, self.description_templates,
                )
                result["description"] = (
                    dr.to_dict() if description_mode == "full" else dr.text
                )
            if feedback_mode:
                error_text = self.feedback_templates[
                    "feedback.validation_error"
                ].format(path=path2str(e.path), message=e.message)
                if feedback_mode == "full":
                    result["feedback"] = {
                        "score": 0.0,
                        "text": error_text,
                        "entries": [],
                        "style": self.feedback_style_default,
                        "truncated": False,
                        "n_total_ops": 0,
                        "error_breakdown": {},
                    }
                else:
                    result["feedback"] = error_text
            return result

        match_tree, ctx = self._align_with_ctx(gold, pred, skip_validation=True)
        result = {"score": float(match_tree.score)}
        if description_mode:
            dr = render_description(
                match_tree,
                style=self.description_style_default,
                templates=self.description_templates,
            )
            result["description"] = (
                dr.to_dict() if description_mode == "full" else dr.text
            )
        if feedback_mode:
            fb = self.feedback_from_match(
                match_tree, gold, pred, ctx.current_mappings,
                include_metadata=(feedback_mode == "full"),
            )
            result["feedback"] = (
                fb.to_dict() if feedback_mode == "full" else fb.text
            )
        if debug:
            result["debug"] = self._serialize_match_debug(match_tree)
        return result

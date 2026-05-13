import string
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
    score: float
    gold: Any
    pred: Any
    kind: str = ""

    def __post_init__(self):
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class MatchList:
    score: float
    children: list = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class MatchDict:
    score: float
    children: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "score", float(self.score))


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


DEFAULT_REASONING_TEMPLATES = {
    "metric.perfect": "The predicted output perfectly matches the gold.",
    "metric.imperfect_intro": "The predicted output scores overall {score_pct}, let us align the predicted output to the gold and analyze the differences:\n",
    "item.match": '{indent}The predicted value "{pred}" exactly matches the gold.\n',
    "item.mismatch": '{indent}The predicted value "{pred}" does not match the gold "{gold}" (score={score_pct}).\n',
    "ref.match": '{indent}The predicted reference "{pred}" matches the gold reference "{gold}" under the inferred id mapping.\n',
    "ref.mismatch": '{indent}The predicted reference "{pred}" does not match the gold reference "{gold}" under the inferred id mapping (score={score_pct}).\n',
    # id rows are intentionally empty: id fields exist for referential
    # bookkeeping and should not appear in human-readable reasoning output.
    "id.match": "",
    "id.mismatch": "",
    "list.match": "{indent}The predicted list perfectly matches the gold one:\n",
    "list.mismatch": "{indent}The predicted list scores {score_pct}:\n",
    "list.excess": '{indent}The predicted list item "{pred}" is excessive, it was not in the gold.\n',
    "list.missing": '{indent}The predicted output misses the "{gold}" list item from the gold.\n',
    "dict.match": "{indent}The predicted dictionary perfectly matches the gold one:\n",
    "dict.mismatch": "{indent}The predicted dictionary scores {score_pct}:\n",
    "dict.key.match": '{indent}KEY = The predicted key "{pred}" exactly matches the gold.\n',
    "dict.key.mismatch": '{indent}KEY = The predicted key "{pred}" does not match the gold "{gold}" (score={score_pct}).\n',
    "dict.value.prefix": "{indent}VALUE = ",
    "validation.error": 'JSON Schema validation failed for path="{path}". Error message: {message}.',
}

# Placeholders the renderer supplies for each template key. Used by
# _merge_reasoning_templates to reject user overrides containing typos
# at construction time instead of at render time. Keep in sync with
# _ReasoningRenderer.
_TEMPLATE_PLACEHOLDERS = {
    "metric.perfect": frozenset(),
    "metric.imperfect_intro": frozenset({"score", "score_pct"}),
    "item.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "item.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "ref.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "ref.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "id.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "id.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "list.match": frozenset({"indent", "score", "score_pct"}),
    "list.mismatch": frozenset({"indent", "score", "score_pct"}),
    "list.excess": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "list.missing": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "dict.match": frozenset({"indent", "score", "score_pct"}),
    "dict.mismatch": frozenset({"indent", "score", "score_pct"}),
    "dict.key.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "dict.key.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "dict.value.prefix": frozenset({"indent"}),
    "validation.error": frozenset({"path", "message"}),
}


def path2str(p):
    return "/" + "/".join([str(d) for d in p])


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


def to_pct_str(v):
    return f"{100 * v:.0f}%"


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


class _ReasoningRenderer:
    def __init__(self, templates):
        self.templates = templates

    def render(self, aligned):
        if aligned.score == 1.0:
            return self.templates["metric.perfect"]
        reasoning = self.templates["metric.imperfect_intro"].format(
            score=aligned.score,
            score_pct=to_pct_str(aligned.score),
        )
        reasoning += self._render_match(aligned, level=0).rstrip()
        return reasoning

    def render_validation_error(self, error):
        return self.templates["validation.error"].format(
            path=path2str(error.path),
            message=error.message,
        )

    def _render_match(self, aligned, level=0):
        indent = "  " * level
        child_indent = "  " * (level + 1)

        if isinstance(aligned, MatchItem):
            kind = getattr(aligned, "kind", "") or "item"
            template_key = f"{kind}.match" if aligned.score == 1.0 else f"{kind}.mismatch"
            return self.templates[template_key].format(
                indent=indent,
                gold=aligned.gold,
                pred=aligned.pred,
                score=aligned.score,
                score_pct=to_pct_str(aligned.score),
            )

        if isinstance(aligned, MatchList):
            template_key = "list.match" if aligned.score == 1.0 else "list.mismatch"
            fragments = [
                self.templates[template_key].format(
                    indent=indent,
                    score=aligned.score,
                    score_pct=to_pct_str(aligned.score),
                )
            ]
            for child in aligned.children:
                if isinstance(child, MatchItem) and child.gold is None and child.pred is None:
                    # Sentinel emitted by _align_lists_prefix when both sides
                    # are shorter than prefixItems at this position; nothing
                    # useful to say to the user.
                    continue
                if isinstance(child, MatchItem) and child.gold is None:
                    fragments.append(
                        self.templates["list.excess"].format(
                            indent=child_indent,
                            pred=child.pred,
                            gold=child.gold,
                            score=child.score,
                            score_pct=to_pct_str(child.score),
                        )
                    )
                elif isinstance(child, MatchItem) and child.pred is None:
                    fragments.append(
                        self.templates["list.missing"].format(
                            indent=child_indent,
                            gold=child.gold,
                            pred=child.pred,
                            score=child.score,
                            score_pct=to_pct_str(child.score),
                        )
                    )
                else:
                    fragments.append(self._render_match(child, level=level + 1))
            return "".join(fragments)

        if isinstance(aligned, MatchDict):
            template_key = "dict.match" if aligned.score == 1.0 else "dict.mismatch"
            fragments = [
                self.templates[template_key].format(
                    indent=indent,
                    score=aligned.score,
                    score_pct=to_pct_str(aligned.score),
                )
            ]
            for key, child in aligned.children.items():
                key_template_key = "dict.key.match" if key.score == 1.0 else "dict.key.mismatch"
                fragments.append(
                    self.templates[key_template_key].format(
                        indent=child_indent,
                        gold=key.gold,
                        pred=key.pred,
                        score=key.score,
                        score_pct=to_pct_str(key.score),
                    )
                )
                fragments.append(
                    self.templates["dict.value.prefix"].format(indent=child_indent)
                    + self._render_match(child, level=level + 1).lstrip()
                    + "\n"
                )
            return "".join(fragments)

        raise AssertionError(f"Unknown match instance: {aligned}")


class ObjectAligner:
    def __init__(self, schema, *, custom_metrics=None, generate_reasoning=False, reasoning_templates=None, warn_on_ambiguous_mapping=False):
        """Create an object aligner.

        custom_metrics maps schema types ("string", "number", "integer") to
        mappings of metric name -> callable. Each callable must accept
        ``(gold, pred)`` and return a real-valued score in ``[0, 1]``. Integer
        schemas use built-in number metrics and fall back to custom ``number``
        metrics unless overridden by a custom ``integer`` metric with the same
        name.

        warn_on_ambiguous_mapping enables a ``UserWarning`` whenever the
        Hungarian-derived id mapping for an ``idScope`` is non-unique because
        of tied costs. Off by default.
        """

        self.schema = schema
        self.generate_reasoning_default = generate_reasoning
        self.reasoning_templates = self._merge_reasoning_templates(reasoning_templates)
        self._reasoning_renderer = _ReasoningRenderer(self.reasoning_templates)
        self._primitive_metrics = self._build_primitive_metric_registry(custom_metrics)
        self._warn_on_ambiguous_mapping = bool(warn_on_ambiguous_mapping)
        self._validate_importance_sums(schema)
        self._id_scopes, self._scope_order = self._collect_id_scopes(schema)
        self._validator = validator_for(schema)(schema)

    def _merge_reasoning_templates(self, reasoning_templates):
        if reasoning_templates is None:
            return dict(DEFAULT_REASONING_TEMPLATES)
        if not isinstance(reasoning_templates, Mapping):
            raise TypeError("reasoning_templates must be a mapping of template keys to strings")

        overrides = dict(reasoning_templates)
        unknown_keys = sorted(set(overrides) - set(DEFAULT_REASONING_TEMPLATES))
        if unknown_keys:
            raise ValueError(f"Unknown reasoning template keys: {unknown_keys}")

        formatter = string.Formatter()
        for key, template in overrides.items():
            if not isinstance(template, str):
                raise TypeError(
                    f'reasoning_templates["{key}"] must be a string, got {type(template).__name__}'
                )
            allowed = _TEMPLATE_PLACEHOLDERS[key]
            used = {name for _, name, _, _ in formatter.parse(template) if name}
            extra = used - allowed
            if extra:
                raise ValueError(
                    f'reasoning_templates["{key}"] uses unknown placeholder(s) '
                    f'{sorted(extra)}; allowed: {sorted(allowed)}'
                )

        templates = dict(DEFAULT_REASONING_TEMPLATES)
        templates.update(overrides)
        return templates

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
                ki = node.get("keyImportance", 1.0)
                vi = node.get("valueImportance", 1.0)
                if ki + vi == 0:
                    raise ValueError(
                        f"keyImportance and valueImportance cannot both be zero at {path2str([str(e) for e in schema_path])}"
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
            if "properties" in node and isinstance(node["properties"], dict):
                for k, v in node["properties"].items():
                    walk(v, schema_path + [("properties", k)])
            if "items" in node:
                walk(node["items"], schema_path + [("items",)])
            if "prefixItems" in node and isinstance(node["prefixItems"], list):
                for i, sub in enumerate(node["prefixItems"]):
                    walk(sub, schema_path + [("prefixItems", i)])

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
            if "properties" in node and isinstance(node["properties"], dict):
                for k, v in node["properties"].items():
                    walk(v, schema_path + [("properties", k)])
            if "items" in node:
                walk(node["items"], schema_path + [("items",)])
            if "prefixItems" in node and isinstance(node["prefixItems"], list):
                for i, sub in enumerate(node["prefixItems"]):
                    walk(sub, schema_path + [("prefixItems", i)])

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
        return self._align_primitive(g, p, schema, schema_type=schema["type"], default_score="invdiff")

    def _align_strings(self, g, p, schema):
        return self._align_primitive(g, p, schema, schema_type="string", default_score="jaro")

    def _align_lists_reorder(self, gold, pred, schema, ctx):
        n, m = len(gold), len(pred)
        d = max(n, m)

        if d == 0:
            return {"gold": gold, "pred": pred, "match": MatchList(score=1.0, children=[])}

        similarity_matrix = np.zeros((d, d))
        subs = np.empty((n, m), dtype=object)

        for i in range(n):
            for j in range(m):
                aligned = self._align_helper(gold[i], pred[j], schema["items"], ctx)
                similarity_matrix[i][j] = aligned["match"].score
                subs[i][j] = (aligned["gold"], aligned["pred"], aligned["match"])

        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        aligned_gold = []
        aligned_pred = []
        aligned_scores = []
        for i in range(len(row_ind)):
            ri, ci = row_ind[i], col_ind[i]
            similarity = similarity_matrix[ri][ci]
            if ri < n and ci < m:
                sg, sp, sscore = subs[ri][ci]
                if sscore.score > 0.0:
                    aligned_gold.append(sg)
                    aligned_pred.append(sp)
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
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_scores)}

    def _align_lists_fixed(self, gold, pred, schema, ctx):
        n, m = len(gold), len(pred)
        if n == 0 and m == 0:
            return {"gold": [], "pred": [], "match": MatchList(score=1.0, children=[])}
        if n == 0:
            return {
                "gold": [None] * m,
                "pred": pred,
                "match": MatchList(score=0.0, children=[MatchItem(score=0.0, gold=None, pred=e) for e in pred]),
            }
        if m == 0:
            return {
                "gold": gold,
                "pred": [None] * n,
                "match": MatchList(score=0.0, children=[MatchItem(score=0.0, gold=e, pred=None) for e in gold]),
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
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_scores)}

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
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_matches)}

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
        return {"gold": gold, "pred": pred, "match": MatchList(score=score, children=children)}

    def _align_dicts(self, g, p, schema, ctx):
        match_key = schema.get("keyScore", "jaro")
        if match_key not in ("exact", "jaro"):
            raise ValueError(f"'keyScore' must be 'exact' or 'jaro', got {match_key!r}")
        key_threshold = schema.get("keyThreshold", 0.0)
        scoref = similarity_exact if match_key == "exact" else similarity_string_jaro

        key_importance = schema.get("keyImportance", 1.0)
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

        aligned_gkeys = []
        aligned_pkeys = []
        aligned_key_scores = []
        for i in range(len(row_ind)):
            ri, ci = row_ind[i], col_ind[i]
            if ri < n and ci < m:
                sg, sp, sim = gkeys[ri], pkeys[ci], similarity_matrix[ri][ci]
                if sim > 0:
                    aligned_gkeys.append(sg)
                    aligned_pkeys.append(sp)
                    aligned_key_scores.append(sim)
                else:
                    if sp is not None:
                        aligned_gkeys.append(None)
                        aligned_pkeys.append(sp)
                        aligned_key_scores.append(sim)
                    if sg is not None:
                        aligned_gkeys.append(sg)
                        aligned_pkeys.append(None)
                        aligned_key_scores.append(sim)
            elif ri < n:
                aligned_gkeys.append(gkeys[ri])
                aligned_pkeys.append(None)
                aligned_key_scores.append(0.0)
            elif ci < m:
                aligned_gkeys.append(None)
                aligned_pkeys.append(pkeys[ci])
                aligned_key_scores.append(0.0)

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
                    if ctx.skip_validation:
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
        for gk, pk, aligned_value, key_score in zip(aligned_gkeys, aligned_pkeys, aligned_values, aligned_key_scores):
            if gk is not None:
                aligned_gold[gk] = aligned_value["gold"]
            if pk is not None:
                aligned_pred[pk] = aligned_value["pred"]
            children[MatchItem(score=key_score, gold=gk, pred=pk)] = aligned_value["match"]

        score = float(key_importance * keys_score + value_importance * values_score) / (key_importance + value_importance)
        score = max(0.0, min(1.0, score))
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchDict(score=score, children=children)}

    def _align_booleans(self, g, p, schema):
        score = similarity_exact(g, p)
        return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p)}

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
                return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p, kind="ref")}
        if isinstance(g, bool):
            if schema.get("type") != "boolean":
                raise TypeError(f"schema declares type={schema.get('type')!r} but data is bool")
            aligned = self._align_booleans(g, p, schema)
        elif isinstance(g, (int, float)):
            if schema.get("type") not in ("number", "integer"):
                raise TypeError(f"schema declares type={schema.get('type')!r} but data is {type(g).__name__}")
            aligned = self._align_numbers(g, p, schema)
        elif isinstance(g, str):
            if schema.get("type") != "string":
                raise TypeError(f"schema declares type={schema.get('type')!r} but data is str")
            aligned = self._align_strings(g, p, schema)
        elif isinstance(g, list):
            if schema.get("type") != "array":
                raise TypeError(f"schema declares type={schema.get('type')!r} but data is list")
            aligned = self._align_lists(g, p, schema, ctx)
        elif isinstance(g, dict):
            if schema.get("type") != "object":
                raise TypeError(f"schema declares type={schema.get('type')!r} but data is dict")
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
            return out

        if isinstance(aligned, MatchList):
            return {
                "kind": "list",
                "score": float(aligned.score),
                "children": [self._serialize_match_debug(child) for child in aligned.children],
            }

        if isinstance(aligned, MatchDict):
            return {
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

        raise TypeError(f"Unknown match instance: {aligned!r}")

    def align(self, g, p, skip_validation=False):
        """Align ``g`` (gold) to ``p`` (pred) and return a match tree.

        Returns a ``MatchItem`` / ``MatchList`` / ``MatchDict`` whose ``.score``
        is in ``[0, 1]``. ``gold`` and ``pred`` are both validated against the
        schema unless ``skip_validation=True``.

        Each call builds its own per-call context, so concurrent calls on the
        same ``ObjectAligner`` instance are safe.
        """
        if type(g) is not type(p):
            raise TypeError(f"gold and pred must be the same type, got {type(g).__name__} and {type(p).__name__}")
        if not skip_validation:
            self._validator.validate(g)
            self._validator.validate(p)
        ctx = _AlignContext(skip_validation=bool(skip_validation))
        if self._id_scopes:
            ctx.gold_ids = self._validate_referential(g)
            ctx.pred_ids = self._collect_pred_ids(p)
            ctx.current_mappings, ctx.pred_excess_ids = self._derive_id_mappings(g, p, ctx)
        return self._align_helper(g, p, self.schema, ctx)["match"]

    def metric(self, gold, pred, debug=False, generate_reasoning=None):
        """Score ``pred`` against ``gold`` and return a result dict.

        Always validates ``gold`` against the schema. If ``pred`` fails
        validation, returns ``{"score": 0.0}`` (with ``"reasoning"`` describing
        the validation error when reasoning is enabled). Otherwise returns
        ``{"score": float}`` plus optional ``"reasoning"`` and ``"debug"``
        entries.

        Like ``align()``, this method is safe to call concurrently on a single
        instance.
        """
        self._validator.validate(gold)

        should_generate_reasoning = self.generate_reasoning_default if generate_reasoning is None else generate_reasoning

        try:
            self._validator.validate(pred)
        except ValidationError as e:
            if should_generate_reasoning:
                return {"score": 0.0, "reasoning": self._reasoning_renderer.render_validation_error(e)}
            return {"score": 0.0}

        aligned = self.align(gold, pred, skip_validation=True)
        result = {"score": float(aligned.score)}
        if should_generate_reasoning:
            result["reasoning"] = self._reasoning_renderer.render(aligned)
        if debug:
            result["debug"] = self._serialize_match_debug(aligned)
        return result

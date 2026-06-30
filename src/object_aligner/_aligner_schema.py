"""Schema-analysis and metric-registry methods of :class:`ObjectAligner`.

Mixed into ``ObjectAligner``; not instantiated on its own. Covers the
custom/built-in primitive-metric registry, schema-child iteration, and the
construction-time validators for importance sums, ``nullScore`` ranges, and
metric scores.
"""
from collections.abc import Mapping
from numbers import Real

from object_aligner._metrics import (
    BUILTIN_NUMBER_METRICS,
    BUILTIN_STRING_METRICS,
    SUPPORTED_CUSTOM_METRIC_TYPES,
    path2str,
)


class _SchemaMixin:
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
            for child, child_path in _SchemaMixin._iter_schema_children(node, schema_path):
                walk(child, child_path)

        walk(schema, [])

    @staticmethod
    def _validate_ignore_flags(schema):
        """Pre-walk the schema and raise ``ValueError`` if any array node
        sets both ``ignoreExcess`` and ``ignoreMissing``. The combination
        would score the mean over matched pairs only, which rewards omitting
        hard items (a strictly closer candidate can score lower); it is
        rejected at construction instead of emitting a gameable score."""

        def walk(node, schema_path):
            if not isinstance(node, dict):
                return
            if node.get("ignoreExcess") and node.get("ignoreMissing"):
                raise ValueError(
                    "'ignoreExcess' and 'ignoreMissing' cannot both be true at "
                    f"{path2str([str(e) for e in schema_path])}"
                )
            for child, child_path in _SchemaMixin._iter_schema_children(node, schema_path):
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
            for child, child_path in _SchemaMixin._iter_schema_children(node, schema_path):
                walk(child, child_path)

        walk(schema, [])

    @staticmethod
    def _enclosing_array_path(schema_path):
        for i in range(len(schema_path) - 1, -1, -1):
            edge = schema_path[i]
            if edge == ("items",) or (isinstance(edge, tuple) and edge and edge[0] == "prefixItems"):
                return tuple(schema_path[: i + 1])
        return None

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


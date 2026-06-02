"""Weisfeiler-Leman id-disambiguation methods of :class:`ObjectAligner`.

Mixed into ``ObjectAligner``. Builds the per-side ref graph from the schema,
instance, and already-resolved higher-scope mappings, and folds the resulting
structural colors into the cost matrix. The pure refinement lives in
``object_aligner._wl``.
"""
import numpy as np

from object_aligner._metrics import _schema_allows_type
from object_aligner._wl import RefGraph, _RefEdge


class _WLMixin:
    def _apply_wl(self, cost, w, n, m):
        """Fold the WL agreement matrix ``w`` into the property cost.

        ``cost`` and ``w`` are ``d×d`` score matrices (higher = better);
        ``w[i][j] == 1`` iff gold item ``i`` and pred item ``j`` carry the
        same stable WL token. Returns the score matrix for the Hungarian step.
        """
        if self._wl_integration == "blend":
            lam = self._wl_blend_lambda
            return (1.0 - lam) * cost + lam * w
        # tie_break: add eps*w with eps strictly below the smallest positive
        # gap among distinct property scores, so any pair already separated by
        # content keeps its ranking — only exact ties are broken by structure.
        sub = cost[:n, :m]
        vals = np.unique(sub)
        if vals.size >= 2:
            positive = np.diff(vals)
            positive = positive[positive > 0]
            min_gap = float(positive.min()) if positive.size else 1.0
        else:
            min_gap = 1.0
        eps = min_gap / (n * m + 1.0)
        out = cost.copy()
        out[:n, :m] = sub + eps * w[:n, :m]
        return out

    @staticmethod
    def _path_under(path, prefix):
        """True iff ``prefix`` is a strict prefix of ``path``."""
        return len(path) > len(prefix) and path[: len(prefix)] == prefix

    def _carrier_path(self, ref_path):
        """Schema path of the carrier object that "owns" a ref site.

        The carrier is the shortest prefix of ``ref_path`` whose schema node
        is an ``object`` sitting as an array item (ends in ``("items",)`` /
        ``("prefixItems", n)``). For an edge object ``{source, target}`` this is
        the edge item (so both refs group into one directed relation); for a
        scalar ``members[*]`` ref array this is the *group* item (so co-members
        form one k-ary relation, not isolated unary tags). Falls back to the
        ref's enclosing array — and finally to the ref's parent — when no such
        object-array-item ancestor exists.
        """
        for i in range(1, len(ref_path) + 1):
            edge = ref_path[i - 1]
            is_array_item = edge == ("items",) or (
                isinstance(edge, tuple) and edge and edge[0] == "prefixItems"
            )
            if not is_array_item:
                continue
            node = self._get_schema_node(self.schema, ref_path[:i])
            if isinstance(node, dict) and _schema_allows_type(node.get("type"), "object"):
                return ref_path[:i]
        enclosing = self._enclosing_array_path(ref_path)
        return enclosing if enclosing is not None else ref_path[:-1]

    @staticmethod
    def _is_exact_comparable(node):
        """True for schema nodes compared by exact equality in WL labels.

        Allows ``string`` / ``integer`` / ``boolean`` and any node with an
        ``enum``; excludes floats (``number``) so structurally identical edges
        are not split by rounding noise, and excludes id/ref primitives (they
        carry no comparable raw value across sides).
        """
        if not isinstance(node, dict):
            return False
        if node.get("idScope") is not None or node.get("ref") is not None:
            return False
        if "enum" in node:
            return True
        return node.get("type") in ("string", "integer", "boolean")

    def _exact_scalars(self, obj, obj_schema):
        """Sorted tuple of an object's direct-child exactly-comparable scalars."""
        if not isinstance(obj, dict) or not isinstance(obj_schema, dict):
            return ()
        props = obj_schema.get("properties")
        if not isinstance(props, dict):
            return ()
        triples = []
        for key, child in props.items():
            if not self._is_exact_comparable(child) or key not in obj:
                continue
            val = obj[key]
            if isinstance(val, (dict, list)):
                continue
            triples.append((type(val).__name__, key, val))
        return tuple(sorted(triples, key=repr))

    def _carrier_label(self, carrier_obj, carrier_path, scope, ctx, is_gold):
        """Build the (hashable, repr-sortable) label for a carrier relation.

        Combines the carrier's own exactly-comparable scalars (e.g. an edge
        ``type``) with any refs it carries to an *already-resolved higher*
        scope. The higher-scope target is mapped to pred space on the gold
        side and taken raw on the pred side, so identical cross-scope structure
        yields identical labels on both sides without bootstrapping the current
        scope.
        """
        carrier_schema = self._get_schema_node(self.schema, carrier_path)
        label = list(self._exact_scalars(carrier_obj, carrier_schema))
        for other_name, other_scope in self._id_scopes.items():
            if other_name == scope.scope or other_name not in ctx.current_mappings:
                continue
            mapping = ctx.current_mappings[other_name]
            for ref_path in other_scope.ref_paths:
                if not self._path_under(ref_path, carrier_path):
                    continue
                rel = ref_path[len(carrier_path):]
                for val, _ in self._walk_data(carrier_obj, rel):
                    resolved = mapping.get(val) if is_gold else val
                    label.append(("xref", other_name, repr(rel), resolved))
        return tuple(sorted(label, key=repr))

    @staticmethod
    def _emit_incidences(endpoints, label, graph, hub_counter):
        """Append the incidences for one carrier relation to ``graph``.

        ``endpoints`` is a list of ``(role, vertex_id)`` already filtered to
        known vertices. A single endpoint becomes a unary self-tag; exactly two
        endpoints with *distinct* roles become a directed edge (e.g.
        ``source → target``); everything else (symmetric collections such as
        ``members``, or higher arity) stars to a fresh per-carrier hub vertex so
        the relation is order-invariant and stays within 1-WL.
        """
        if not endpoints:
            return
        roles = {role for role, _ in endpoints}
        if len(endpoints) == 1:
            role, vid = endpoints[0]
            graph.incidences.append(_RefEdge(src=vid, dst=vid, role=("unary", role), label=label))
            return
        if len(endpoints) == 2 and len(roles) == 2:
            (r0, a), (r1, b) = sorted(endpoints, key=lambda rv: repr(rv[0]))
            graph.incidences.append(_RefEdge(src=a, dst=b, role=("edge", r0, r1), label=label))
            return
        hub = ("__hub__", hub_counter[0])
        hub_counter[0] += 1
        graph.vertices[hub] = ()
        for role, vid in endpoints:
            graph.incidences.append(_RefEdge(src=vid, dst=hub, role=("member", role), label=label))

    def _build_ref_graph(self, instance, scope, ctx, *, is_gold):
        """Build the per-side directed labeled ref graph for ``scope``."""
        suffix = scope.definer_schema_path[len(scope.definer_array_path):]
        item_schema = self._get_schema_node(self.schema, scope.definer_array_path)
        want_scalars = self._wl_integration == "blend"

        graph = RefGraph()
        for item, _ in self._walk_data(instance, scope.definer_array_path):
            vid = next((v for v, _ in self._walk_data(item, suffix)), None)
            if vid is None or vid in graph.vertices:
                continue
            graph.vertices[vid] = self._exact_scalars(item, item_schema) if want_scalars else ()

        groups = {}
        for ref_path in scope.ref_paths:
            carrier_path = self._carrier_path(ref_path)
            groups.setdefault(carrier_path, []).append(ref_path[len(carrier_path):])

        hub_counter = [0]
        for carrier_path, role_remainders in groups.items():
            role_remainders = sorted(role_remainders, key=repr)
            for carrier_obj, _ in self._walk_data(instance, carrier_path):
                endpoints = []
                for rel in role_remainders:
                    for val, _ in self._walk_data(carrier_obj, rel):
                        if val in graph.vertices:
                            endpoints.append((rel, val))
                label = self._carrier_label(carrier_obj, carrier_path, scope, ctx, is_gold)
                self._emit_incidences(endpoints, label, graph, hub_counter)
        return graph


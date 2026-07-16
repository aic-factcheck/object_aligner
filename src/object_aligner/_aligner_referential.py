"""idScope / ref discovery, validation, and bijection derivation.

Mixed into ``ObjectAligner``. Collects id scopes from the schema, topologically
orders them, validates the gold/pred reference graphs, and derives the per-scope
gold-to-pred id bijection via the Hungarian assignment (optionally WL-refined).

This implements the paper's *referential alignment*: a *relabel-invariant*
scoring mode in which an ``idScope`` field is an identifier and ``ref`` fields
are references to it (like primary/foreign keys). A scope is a named set of
*records* (the items bearing the ``idScope`` field); the derived bijection maps
gold ids to candidate ids so references are scored through it rather than by
raw value equality.
"""
import warnings

import numpy as np
from jsonschema import ValidationError
from scipy.optimize import linear_sum_assignment

from object_aligner._matchtypes import _IdScope
from object_aligner._metrics import path2str
from object_aligner._wl import wl_tokens


class _ReferentialMixin:
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
            yield from _ReferentialMixin._walk_data(data[key], rest, data_path + (key,))
        elif edge == ("items",):
            if not isinstance(data, list):
                return
            for i, item in enumerate(data):
                yield from _ReferentialMixin._walk_data(item, rest, data_path + (i,))
        elif isinstance(edge, tuple) and edge and edge[0] == "prefixItems":
            idx = edge[1]
            if not isinstance(data, list) or idx >= len(data):
                return
            yield from _ReferentialMixin._walk_data(data[idx], rest, data_path + (idx,))

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

        # Parent handles for context-aware leaf metrics on definer items.
        # A metric that dereferenced a None parent here would raise, be
        # swallowed by the except below, and silently corrupt the
        # bijection (cost 0.0) — so pass the definer item lists as the
        # enclosing parents and the item's data path.
        gold_item_list = [it for it, _ in gold_items]
        pred_item_list = [it for it, _ in pred_items]

        d = max(n, m)
        cost = np.zeros((d, d))
        for i in range(n):
            for j in range(m):
                g_item = gold_items[i][0]
                p_item = pred_items[j][0]
                try:
                    aligned = self._align_helper(
                        g_item, p_item, item_schema, ctx,
                        gold_parent=gold_item_list, pred_parent=pred_item_list,
                        path=gold_items[i][1],
                    )
                    cost[i][j] = aligned["match"].score
                except (TypeError, ValueError, KeyError):
                    cost[i][j] = 0.0

        wl_active = (
            self._id_disambiguation == "wl"
            and not scope.degraded
            and (n > 1 or m > 1)
        )
        if wl_active:
            gold_tokens, pred_tokens = wl_tokens(
                self._build_ref_graph(gold, scope, ctx, is_gold=True),
                self._build_ref_graph(pred, scope, ctx, is_gold=False),
                mode=self._wl_integration,
                rounds=self._wl_rounds,
            )
            w = np.zeros((d, d))
            for i in range(n):
                gtok = gold_tokens.get(gold_id_list[i])
                if gtok is None:
                    continue
                for j in range(m):
                    if pred_tokens.get(pred_id_list[j]) == gtok:
                        w[i][j] = 1.0
            score_matrix = self._apply_wl(cost, w, n, m)
        else:
            score_matrix = cost

        row_ind, col_ind = linear_sum_assignment(-score_matrix)

        self._maybe_warn_ambiguity(
            score_matrix, n, m, gold_id_list, scope.scope, wl_active=wl_active
        )

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

    def _maybe_warn_ambiguity(self, matrix, n, m, gold_id_list, scope_name, *, wl_active=False):
        # `matrix` is the cost matrix actually fed to the Hungarian step: the
        # raw property cost under `id_disambiguation="none"`, or the
        # WL-integrated score matrix under `"wl"`. Two equal rows therefore
        # signal *residual* ambiguity in the WL case — pairs WL could not
        # separate (genuine automorphisms or 1-WL blind spots).
        if not self._warn_on_ambiguous_mapping or n < 2:
            return
        seen = {}
        ambiguous = set()
        for i in range(n):
            key = tuple(matrix[i, :m].tolist()) if m > 0 else ()
            if key in seen:
                ambiguous.add(gold_id_list[seen[key]])
                ambiguous.add(gold_id_list[i])
            else:
                seen[key] = i
        ambiguous.discard(None)
        if not ambiguous:
            return
        ids = sorted(ambiguous, key=repr)
        if wl_active:
            warnings.warn(
                f"Residual ambiguous mapping in idScope '{scope_name}' after WL "
                f"refinement: gold ids {ids} remain structurally indistinguishable "
                f"(graph automorphism or 1-WL blind spot); arbitrary assignment used.",
                UserWarning,
            )
        else:
            warnings.warn(
                f"Ambiguous mapping in idScope '{scope_name}': gold ids {ids} could "
                f"be paired multiple ways with equal cost; arbitrary assignment used.",
                UserWarning,
            )

"""Endpoint descriptors for the opt-in "semantic" referential-feedback mode.

Mixed into ``ObjectAligner``. For each ``ref_fix`` / ``ref_fix_no_target``
repair op, builds a small read-only :class:`_RefEndpointDesc` describing the
*gold endpoint node* the reference should resolve to — by its discriminative
direct-child scalar properties, plus the relation/edge label and the (wrong)
candidate endpoint — so feedback can render a *transferable* lesson instead of
an opaque id renumbering. See ``docs/feedback.md`` (semantic referential
feedback) and ``render_feedback(referential_feedback="semantic")``.

This module never mutates state, never raises on malformed input (it degrades
to ``usable=False`` so the renderer falls back to the literal line), and reuses
the same discriminative-property / carrier-label machinery as the WL
disambiguator (:mod:`object_aligner._aligner_wl`). Repair output is untouched —
every datum here is recovered from the ops plus ``gold`` / ``pred`` / schema.
"""
from dataclasses import dataclass
from typing import Any

from object_aligner._metrics import _schema_allows_type

_REF_OP_KINDS = ("ref_fix", "ref_fix_no_target")
_SCALAR_TYPES = ("string", "integer", "number", "boolean")


@dataclass(frozen=True)
class _RefEndpointDesc:
    """Per-ref-op endpoint description consumed by the feedback renderer.

    Raw (unformatted) values are stored so the renderer can apply the caller's
    ``value_formatter``. ``usable=False`` means "no discriminator available —
    render the literal line instead".
    """

    usable: bool
    scope: str = ""
    # Ordered ``(key, raw_value)`` pairs for the gold endpoint the ref should
    # point to (the transferable signal).
    gold_props: tuple = ()
    # Same shape for the node actually referenced, or ``None`` when the
    # candidate id resolves to no definer (dangling candidate ref).
    pred_props: Any = None
    # Raw values of the carrier/edge's discriminative scalars (the relation
    # label, e.g. ``":ARG0"`` / ``"double"``); empty when the edge has none.
    relation_values: tuple = ()
    # ``False`` when another gold definer in the scope shares the same property
    # signature (property-twin) — the renderer hedges the description.
    endpoint_certain: bool = True


_UNUSABLE = _RefEndpointDesc(usable=False)


def _decode_pointer_token(tok: str) -> str:
    """Inverse of RFC 6901 escaping: ``~1`` -> ``/`` then ``~0`` -> ``~``."""
    return tok.replace("~1", "/").replace("~0", "~")


class _ReferentialFeedbackMixin:
    """``ObjectAligner`` methods that build semantic ref-endpoint descriptors."""

    def _build_ref_endpoint_descriptors(self, gold, pred, ops):
        """Return ``{op.path: _RefEndpointDesc}`` for every ref op in ``ops``.

        Caches per-scope definer indexes and gold property-signature
        multiplicities so repeated ref ops into the same scope are cheap.
        """
        out = {}
        gold_idx_cache: dict = {}
        pred_idx_cache: dict = {}
        gold_sig_mult: dict = {}
        for op in ops:
            if op.kind not in _REF_OP_KINDS:
                continue
            out[op.path] = self._describe_ref_endpoint(
                gold, pred, op, gold_idx_cache, pred_idx_cache, gold_sig_mult,
            )
        return out

    def _describe_ref_endpoint(
        self, gold, pred, op, gold_idx_cache, pred_idx_cache, gold_sig_mult,
    ):
        tokens = [_decode_pointer_token(t) for t in op.path.split("/")[1:]]
        resolved = self._schema_edges_for_pointer(tokens)
        if resolved is None:
            return _UNUSABLE
        ref_edges, leaf_node = resolved
        scope_name = leaf_node.get("ref") if isinstance(leaf_node, dict) else None
        if scope_name is None or scope_name not in self._id_scopes:
            return _UNUSABLE
        scope = self._id_scopes[scope_name]
        item_schema = self._get_schema_node(self.schema, scope.definer_array_path)

        gold_index = self._definer_index_cached(gold, scope, gold_idx_cache)
        gold_item = gold_index.get(op.gold)
        if gold_item is None:
            return _UNUSABLE
        gold_props = self._definer_props(gold_item, item_schema)
        if not gold_props:
            # No discriminating property — a literal line is the honest output.
            return _UNUSABLE

        mult = self._gold_signature_multiplicity(
            gold, scope, item_schema, gold_idx_cache, gold_sig_mult,
        )
        endpoint_certain = mult.get(gold_props, 0) <= 1

        pred_index = self._definer_index_cached(pred, scope, pred_idx_cache)
        pred_item = pred_index.get(op.pred)
        pred_props = (
            self._definer_props(pred_item, item_schema)
            if pred_item is not None
            else None
        )

        relation_values = self._relation_label_values(pred, tokens, ref_edges)

        return _RefEndpointDesc(
            usable=True,
            scope=scope_name,
            gold_props=gold_props,
            pred_props=pred_props,
            relation_values=relation_values,
            endpoint_certain=endpoint_certain,
        )

    # -- schema / data navigation ------------------------------------------

    def _schema_edges_for_pointer(self, tokens):
        """Map an RFC-6901 data path to schema edges + the leaf schema node.

        Returns ``(edges, leaf_node)`` (mirroring ``_walk_data`` /
        ``_get_schema_node`` edge encoding) or ``None`` when the path cannot be
        resolved against the schema's declared shape.
        """
        node = self.schema
        edges: list = []
        for tok in tokens:
            if not isinstance(node, dict):
                return None
            props = node.get("properties")
            if isinstance(props, dict) and tok in props:
                edges.append(("properties", tok))
                node = props[tok]
                continue
            prefix_items = node.get("prefixItems")
            try:
                idx = int(tok)
            except (TypeError, ValueError):
                return None
            if isinstance(prefix_items, list) and idx < len(prefix_items):
                edges.append(("prefixItems", idx))
                node = prefix_items[idx]
            elif "items" in node:
                edges.append(("items",))
                node = node["items"]
            else:
                return None
        return edges, node

    @staticmethod
    def _data_at(obj, tokens):
        """Follow decoded RFC-6901 ``tokens`` into ``obj``; ``None`` if absent."""
        cur = obj
        for tok in tokens:
            if isinstance(cur, dict):
                if tok not in cur:
                    return None
                cur = cur[tok]
            elif isinstance(cur, list):
                try:
                    idx = int(tok)
                except (TypeError, ValueError):
                    return None
                if not (0 <= idx < len(cur)):
                    return None
                cur = cur[idx]
            else:
                return None
        return cur

    # -- definer properties -------------------------------------------------

    def _definer_index_cached(self, instance, scope, cache):
        idx = cache.get(scope.scope)
        if idx is None:
            idx = self._definer_index(instance, scope)
            cache[scope.scope] = idx
        return idx

    def _definer_index(self, instance, scope):
        """Map ``id -> definer item`` for a scope on one instance (first-wins)."""
        suffix = scope.definer_schema_path[len(scope.definer_array_path):]
        idx: dict = {}
        for item, _ in self._walk_data(instance, scope.definer_array_path):
            vid = next((v for v, _ in self._walk_data(item, suffix)), None)
            if vid is not None and vid not in idx:
                idx[vid] = item
        return idx

    @staticmethod
    def _definer_props(item, item_schema):
        """Ordered ``(key, value)`` of a definer's direct-child scalar props.

        Excludes the ``idScope`` field, any ``ref`` fields, and non-scalar
        children — the same fields the referential cost matrix discriminates on.
        """
        if not isinstance(item, dict) or not isinstance(item_schema, dict):
            return ()
        props = item_schema.get("properties")
        if not isinstance(props, dict):
            return ()
        out = []
        for key, child in props.items():
            if not isinstance(child, dict):
                continue
            if child.get("idScope") is not None or child.get("ref") is not None:
                continue
            t = child.get("type")
            if not any(_schema_allows_type(t, s) for s in _SCALAR_TYPES):
                continue
            if key not in item:
                continue
            val = item[key]
            if isinstance(val, (dict, list)):
                continue
            out.append((key, val))
        return tuple(out)

    def _gold_signature_multiplicity(
        self, gold, scope, item_schema, gold_idx_cache, gold_sig_mult,
    ):
        mult = gold_sig_mult.get(scope.scope)
        if mult is not None:
            return mult
        index = self._definer_index_cached(gold, scope, gold_idx_cache)
        mult = {}
        for item in index.values():
            sig = self._definer_props(item, item_schema)
            mult[sig] = mult.get(sig, 0) + 1
        gold_sig_mult[scope.scope] = mult
        return mult

    # -- relation label -----------------------------------------------------

    def _relation_label_values(self, pred, tokens, ref_edges):
        """Discriminative scalar values of the candidate carrier (edge).

        Read from ``pred`` because the op path is pred-space; refs/ids are
        already excluded by ``_exact_scalars``. Empty when the carrier carries
        no such label.
        """
        carrier_edges = self._carrier_path(tuple(ref_edges))
        carrier_obj = self._data_at(pred, tokens[: len(carrier_edges)])
        if not isinstance(carrier_obj, dict):
            return ()
        carrier_schema = self._get_schema_node(self.schema, carrier_edges)
        scalars = self._exact_scalars(carrier_obj, carrier_schema)
        return tuple(val for _typename, _key, val in scalars)

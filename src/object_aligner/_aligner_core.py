"""Primitive / list / dict / null / boolean alignment and type dispatch.

Mixed into ``ObjectAligner``. The recursive ``_align_*`` family plus the
``_align_helper`` dispatcher and the ``_serialize_match_debug`` debug-tree
serializer.
"""
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from object_aligner._confidence import _hungarian_confidence, _with_confidence
from object_aligner._matchtypes import MatchDict, MatchItem, MatchList, to_python_value
from object_aligner._metrics import _schema_allows_type, similarity_exact, similarity_string_jaro


class _CoreAlignMixin:
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
            # D == 0 means every unmatched entry was ignored (or there were
            # no entries at all): a vacuous match. Matched pairs always count
            # in D, and ignoreExcess/ignoreMissing are mutually exclusive at
            # construction, so no content can hide behind D == 0.
            score = 1.0
        else:
            score = float(np.sum([s.score for s in aligned_scores])) / D
        score = max(0.0, min(1.0, score))
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_scores, kind="reorder", confidence=node_conf)}

    def _align_lists_fixed(self, gold, pred, schema, ctx):
        n, m = len(gold), len(pred)
        if n == 0 and m == 0:
            return {"gold": [], "pred": [], "match": MatchList(score=1.0, children=[], kind="fixed")}
        if n == 0:
            # Every pred item is excess; with ignoreExcess none of them
            # counts, so the match is vacuous (consistent with the D == 0
            # rule on the main path).
            score = 1.0 if schema.get("ignoreExcess", False) else 0.0
            return {
                "gold": [None] * m,
                "pred": pred,
                "match": MatchList(score=score, children=[MatchItem(score=0.0, gold=None, pred=e) for e in pred], kind="fixed"),
            }
        if m == 0:
            # Every gold item is missing; vacuous under ignoreMissing.
            score = 1.0 if schema.get("ignoreMissing", False) else 0.0
            return {
                "gold": gold,
                "pred": [None] * n,
                "match": MatchList(score=score, children=[MatchItem(score=0.0, gold=e, pred=None) for e in gold], kind="fixed"),
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
            # Same rule as the reorder path: D == 0 ⇔ every entry was
            # ignored, so the match is vacuous.
            score = 1.0
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
        present = []
        prefix_items = schema["prefixItems"]
        for i, sub_schema in enumerate(prefix_items):
            g_present = i < len(gold)
            p_present = i < len(pred)
            present.append(g_present or p_present)
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
                # Both sides shorter than len(prefixItems): the position is
                # vacuous — pred cannot be blamed for a slot gold does not
                # have either. Emit a kind="absent" sentinel so the tree
                # keeps one child per prefix position, but exclude it from
                # the weight normalization below (else metric(g, g) < 1).
                aligned_gold.append(None)
                aligned_pred.append(None)
                aligned_matches.append(MatchItem(score=0.0, gold=None, pred=None, kind="absent"))
        weights = np.array(schema.get("prefixWeights", np.ones(len(aligned_matches))), dtype=np.float64)
        weights = weights * np.array(present, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            # All positions absent (or the present positions carry zero
            # weight): nothing to grade, vacuous match.
            score = 1.0
            node_conf = 1.0
            return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_matches, kind="prefix", confidence=node_conf)}
        weights = weights / total
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
                properties = schema.get("properties", {})
                if gk not in properties:
                    # Open-world JSON Schema (additionalProperties defaults
                    # to true) lets gold carry keys the schema never
                    # declares. There is no schema node to score against:
                    # soft-zero the pair and warn instead of crashing.
                    warnings.warn(
                        f"gold key {gk!r} is not declared in the schema's "
                        "'properties'; scored 0.0 — declare the property or "
                        "set additionalProperties: false",
                        UserWarning,
                    )
                    aligned_value = {"gold": ag, "pred": ap, "match": MatchItem(score=0.0, gold=ag, pred=ap)}
                    value_weights.append(1.0)
                    aligned_values.append(aligned_value)
                    continue
                aux_schema = properties[gk]
                value_weights.append(aux_schema.get("valueWeight", 1.0))

                if type(ag) is not type(ap):
                    if ag is None or ap is None:
                        # Null-aware: delegate to _align_helper, which routes
                        # through `_align_null` and consults this property's
                        # `nullScore` (default 0.0).
                        aligned_value = self._align_helper(ag, ap, aux_schema, ctx)
                    else:
                        # Soft-zero: a fuzzily paired or union-typed value
                        # whose Python types differ scores 0 rather than
                        # raising, so align() and metric() agree on
                        # schema-valid inputs.
                        aligned_value = {"gold": ag, "pred": ap, "match": MatchItem(score=0.0, gold=ag, pred=ap)}
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


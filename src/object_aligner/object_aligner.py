from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from jsonschema import ValidationError, validate
from rapidfuzz.distance import Jaro
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchItem:
    score: float
    gold: Any
    pred: Any


@dataclass(frozen=True)
class MatchList:
    score: float
    children: list = field(default_factory=list)


@dataclass(frozen=True)
class MatchDict:
    score: float
    children: dict = field(default_factory=dict)


DEFAULT_REASONING_TEMPLATES = {
    "metric.perfect": "The predicted output perfectly matches the gold.",
    "metric.imperfect_intro": "The predicted output scores overall {score_pct}, let us align the predicted output to the gold and analyze the differences:\n",
    "item.match": '{indent}The predicted value "{pred}" exactly matches the gold.\n',
    "item.mismatch": '{indent}The predicted value "{pred}" does not match the gold "{gold}" (score={score_pct}).\n',
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


def to_pct_str(v):
    return f"{100 * v:.0f}%"


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
            template_key = "item.match" if aligned.score == 1.0 else "item.mismatch"
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
    def __init__(self, schema, *, generate_reasoning=False, reasoning_templates=None):
        self.schema = schema
        self.generate_reasoning_default = generate_reasoning
        self.reasoning_templates = self._merge_reasoning_templates(reasoning_templates)
        self._reasoning_renderer = _ReasoningRenderer(self.reasoning_templates)

    def _merge_reasoning_templates(self, reasoning_templates):
        if reasoning_templates is None:
            return dict(DEFAULT_REASONING_TEMPLATES)
        if not isinstance(reasoning_templates, Mapping):
            raise TypeError("reasoning_templates must be a mapping of template keys to strings")

        overrides = dict(reasoning_templates)
        unknown_keys = sorted(set(overrides) - set(DEFAULT_REASONING_TEMPLATES))
        if unknown_keys:
            raise ValueError(f"Unknown reasoning template keys: {unknown_keys}")

        templates = dict(DEFAULT_REASONING_TEMPLATES)
        templates.update(overrides)
        return templates

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
        score_type = schema.get("score", "invdiff")
        threshold = schema.get("threshold", 0.0)

        assert score_type in ["exact", "invdiff"]
        if score_type == "exact":
            score = similarity_exact(g, p)
        else:
            score = similarity_num_inv_diff(g, p)

        score = 0.0 if score < threshold else score
        assert 0 <= score <= 1, score
        return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p)}

    def _align_strings(self, g, p, schema):
        score_type = schema.get("score", "jaro")
        threshold = schema.get("threshold", 0.0)
        assert score_type in ["exact", "jaro"]
        if score_type == "exact":
            score = similarity_exact(g, p)
        else:
            score = similarity_string_jaro(g, p)
        score = 0.0 if score < threshold else score
        assert 0 <= score <= 1, score
        return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p)}

    def _align_lists_reorder(self, gold, pred, schema):
        n, m = len(gold), len(pred)
        d = max(n, m)

        if d == 0:
            return {"gold": gold, "pred": pred, "match": MatchList(score=1.0, children=[])}

        similarity_matrix = np.zeros((d, d))
        subs = np.empty((n, m), dtype=object)

        for i in range(n):
            for j in range(m):
                aligned = self._align_helper(gold[i], pred[j], schema["items"])
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
                    if sp:
                        aligned_gold.append(None)
                        aligned_pred.append(sp)
                        aligned_scores.append(MatchItem(0.0, gold=None, pred=sp))
                    if sg:
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
            score = np.sum([s.score for s in aligned_scores]) / D
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_scores)}

    def _align_lists_fixed(self, gold, pred, schema):
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
                aligned = self._align_helper(gold[i - 1], pred[j - 1], schema["items"])
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
                if sp:
                    aligned_gold.append(None)
                    aligned_pred.append(sp)
                    aligned_scores.append(MatchItem(0.0, gold=None, pred=sp))
                if sg:
                    aligned_gold.append(sg)
                    aligned_pred.append(None)
                    aligned_scores.append(MatchItem(0.0, gold=sg, pred=None))

            if sg is not None:
                i -= 1
            if sp is not None:
                j -= 1

        if i > 0:
            assert j <= 0
            while i > 0:
                aligned_gold.append(subs[i][1][0])
                aligned_pred.append(None)
                aligned_scores.append(MatchItem(0.0, gold=subs[i][1][0], pred=None))
                i -= 1
        if j > 0:
            assert i <= 0
            while j > 0:
                aligned_gold.append(None)
                aligned_pred.append(subs[1][j][1])
                aligned_scores.append(MatchItem(0.0, gold=None, pred=subs[1][j][1]))
                j -= 1

        aligned_gold.reverse()
        aligned_pred.reverse()
        aligned_scores.reverse()

        assert len(aligned_gold) == len(aligned_pred)
        D = self._list_norm(aligned_gold, aligned_pred, schema)
        if D == 0:
            score = 1.0 if len(aligned_scores) == 0 else 0.0
        else:
            score = dp[n][m] / D
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_scores)}

    def _align_lists_prefix(self, gold, pred, schema):
        aligned_gold = []
        aligned_pred = []
        aligned_matches = []
        for g, p, schema_ in zip(gold, pred, schema["prefixItems"]):
            aligned = self._align_helper(g, p, schema_)
            aligned_gold.append(aligned["gold"])
            aligned_pred.append(aligned["pred"])
            aligned_matches.append(aligned["match"])
        weights = np.array(schema.get("prefixWeights", np.ones(len(aligned_gold))), dtype=np.float64)
        weights = weights / weights.sum()
        score = np.sum([e.score * w for e, w in zip(aligned_matches, weights)])
        ret = {"gold": aligned_gold, "pred": aligned_pred, "match": MatchList(score=score, children=aligned_matches)}
        return ret

    def _align_lists(self, g, p, schema):
        assert "prefixItems" in schema or "items" in schema

        rets = []
        prefix_len = 0
        if "prefixItems" in schema:
            prefix_len = len(schema["prefixItems"])
            rets.append(self._align_lists_prefix(g[:prefix_len], p[:prefix_len], schema))

        if "items" in schema:
            ordering = schema.get("order", "fixed")
            assert ordering in ["align", "fixed"]
            if ordering == "fixed":
                rets.append(self._align_lists_fixed(g[prefix_len:], p[prefix_len:], schema))
            else:
                rets.append(self._align_lists_reorder(g[prefix_len:], p[prefix_len:], schema))

        if len(rets) == 1:
            return rets[0]
        assert "prefixImportance" in schema and "restImportance" in schema, "prefixImportance and restImportance must be set if both prefixItems and items are present!"
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

    def _align_dicts(self, g, p, schema):
        match_key = schema.get("keyScore", "jaro")
        assert match_key in ["exact", "jaro"]
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
                    if sp:
                        aligned_gkeys.append(None)
                        aligned_pkeys.append(sp)
                        aligned_key_scores.append(sim)
                    if sg:
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

        keys_score = np.mean(aligned_key_scores)

        aligned_values = []
        value_weights = []
        for gk, pk in zip(aligned_gkeys, aligned_pkeys):
            ag = g.get(gk)
            ap = p.get(pk)
            assert gk is not None or pk is not None, "At least one has to be aligned, check key alignment above!"
            if gk is not None and pk is not None:
                aux_schema = schema["properties"][gk]
                value_weights.append(schema["properties"][gk].get("valueWeight", 1.0))

                if type(ag) != type(ap):
                    raise ValueError(f"The keys are currently matched ignoring types of the respective values: {type(ag)} != {type(ap)}")
                aligned_value = self._align_helper(ag, ap, aux_schema)
            else:
                aligned_value = {"gold": ag, "pred": ap, "match": MatchItem(score=0.0, gold=ag, pred=ap)}
                value_weights.append(1.0)
            aligned_values.append(aligned_value)
        value_scores = np.array([e["match"].score for e in aligned_values])
        value_weights = np.array(value_weights) / np.sum(value_weights)
        values_score = np.sum(value_weights * value_scores)

        aligned_gold = {}
        aligned_pred = {}
        children = {}
        for gk, pk, aligned_value, key_score in zip(aligned_gkeys, aligned_pkeys, aligned_values, aligned_key_scores):
            if gk is not None:
                aligned_gold[gk] = aligned_value["gold"]
            if pk is not None:
                aligned_pred[pk] = aligned_value["pred"]
            children[MatchItem(score=key_score, gold=gk, pred=pk)] = aligned_value["match"]

        score = (key_importance * keys_score + value_importance * values_score) / (key_importance + value_importance)
        return {"gold": aligned_gold, "pred": aligned_pred, "match": MatchDict(score=score, children=children)}

    def _align_booleans(self, g, p, schema):
        score = similarity_exact(g, p)
        return {"gold": g, "pred": p, "match": MatchItem(score=score, gold=g, pred=p)}

    def _align_helper(self, g, p, schema):
        if isinstance(g, bool):
            assert schema["type"] == "boolean", schema["type"]
            aligned = self._align_booleans(g, p, schema)
        elif isinstance(g, (int, float)):
            assert schema["type"] in ["number", "integer"], schema["type"]
            aligned = self._align_numbers(g, p, schema)
        elif isinstance(g, str):
            assert schema["type"] == "string", schema["type"]
            aligned = self._align_strings(g, p, schema)
        elif isinstance(g, list):
            assert schema["type"] == "array", schema["type"]
            aligned = self._align_lists(g, p, schema)
        elif isinstance(g, dict):
            assert schema["type"] == "object", schema["type"]
            aligned = self._align_dicts(g, p, schema)
        else:
            raise ValueError(f"Not yet implemented for {type(g)}!")

        assert 0 <= aligned["match"].score <= 1, aligned
        return aligned

    def align(self, g, p, skip_validation=False):
        assert type(g) == type(p), f"The schemas must be the same, got different types: {type(g)} and {type(p)}"
        if not skip_validation:
            validate(instance=g, schema=self.schema)
            validate(instance=p, schema=self.schema)
        return self._align_helper(g, p, self.schema)["match"]

    def metric(self, gold, pred, debug=False, generate_reasoning=None):
        del debug
        validate(instance=gold, schema=self.schema)

        should_generate_reasoning = self.generate_reasoning_default if generate_reasoning is None else generate_reasoning

        try:
            validate(instance=pred, schema=self.schema)
        except ValidationError as e:
            if should_generate_reasoning:
                return {"score": 0.0, "reasoning": self._reasoning_renderer.render_validation_error(e)}
            return {"score": 0.0}

        aligned = self.align(gold, pred, skip_validation=True)
        result = {"score": aligned.score}
        if should_generate_reasoning:
            result["reasoning"] = self._reasoning_renderer.render(aligned)
        return result

"""Top-level aligner: construction, public API, and class assembly.

``ObjectAligner`` is assembled here from the subsystem mixins
(:class:`~object_aligner._aligner_schema._SchemaMixin`,
:class:`~object_aligner._aligner_referential._ReferentialMixin`,
:class:`~object_aligner._aligner_wl._WLMixin`,
:class:`~object_aligner._aligner_core._CoreAlignMixin`) and holds the
constructor plus the public API (``align`` / ``metric`` / ``attribute`` /
``repair`` / ``feedback`` / ``describe``). The leaf match types, metric
registries, and confidence helpers are re-exported from here so that
``object_aligner.object_aligner.<name>`` keeps resolving for existing imports.
"""
import numpy as np
from jsonschema import ValidationError
from jsonschema.validators import validator_for

# Leaf types / helpers re-exported from this module. Sibling subpackages and the
# test suite import several of these directly from ``object_aligner.object_aligner``,
# so they must stay bound in this module's namespace.
from object_aligner._matchtypes import (
    MatchDict,
    MatchItem,
    MatchList,
    _AlignContext,
    _IdScope,
    to_python_value,
)
from object_aligner._metrics import (
    BUILTIN_NUMBER_METRICS,
    BUILTIN_STRING_METRICS,
    SUPPORTED_CUSTOM_METRIC_TYPES,
    _schema_allows_type,
    path2str,
    similarity_exact,
    similarity_num_inv_diff,
    similarity_string_damerau_levenshtein,
    similarity_string_indel,
    similarity_string_jaro,
    similarity_string_jaro_winkler,
    similarity_string_lcsseq,
    similarity_string_levenshtein,
    similarity_string_osa,
)
from object_aligner._confidence import _hungarian_confidence, _with_confidence
from object_aligner._aligner_schema import _SchemaMixin
from object_aligner._aligner_referential import _ReferentialMixin
from object_aligner._aligner_wl import _WLMixin
from object_aligner._aligner_reffeedback import _ReferentialFeedbackMixin
from object_aligner._aligner_core import _CoreAlignMixin

# Cross-module imports are placed here -- *after* the leaf types above are bound
# -- so the sibling modules (which import MatchItem/MatchList/MatchDict off this
# module at their own top level) resolve them off the partially-loaded
# ``object_aligner.object_aligner``. The one genuinely-circular pair
# (``describe._walk`` needs the Match types at render time) is still broken by a
# lazy import inside that function.
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


class ObjectAligner(
    _SchemaMixin,
    _ReferentialMixin,
    _WLMixin,
    _ReferentialFeedbackMixin,
    _CoreAlignMixin,
):
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
        referential_feedback="literal",
        dominant_fraction_threshold=0.60,
        warn_on_ambiguous_mapping=False,
        compute_confidence=False,
        confidence_method="margin",
        confidence_entropy_temperature=8.0,
        id_disambiguation="wl",
        wl_integration="tie_break",
        wl_rounds=None,
        wl_blend_lambda=0.5,
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
            referential_feedback: How `feedback()` renders `ref` /
                `idScope` mismatches. `"literal"` (default) uses opaque
                ids and is byte-identical to earlier releases.
                `"semantic"` instead describes the gold endpoint node the
                reference should connect to by its discriminative
                properties and the relation label — a transferable lesson
                for prompt optimizers. Only `feedback().text` changes;
                scores and every other output are identical. A no-op on
                schemas without `ref` / `idScope`.
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
            id_disambiguation: Strategy for resolving the per-scope
                `idScope` bijection when definer items are not fully
                distinguished by their own properties. `"wl"` (default)
                runs Weisfeiler–Leman color refinement over the same-scope
                ref graph, computed independently per side, so structurally
                distinct definers align by structure rather than emission
                order. `"none"` reproduces the pre-WL behavior exactly
                (property-only cost plus an arbitrary tie-break). See
                [`docs/referential.md`](../referential.md).
            wl_integration: How the structural color enters the cost matrix
                when `id_disambiguation="wl"`. `"tie_break"` (default) lets
                the color break only *exact* ties in the property cost, so
                already-determined alignments never move. `"blend"` mixes
                the property cost and structural agreement with weight
                `wl_blend_lambda`, letting structure override near-tied (but
                not exactly tied) property costs.
            wl_rounds: Cap on WL refinement rounds. `None` (default) runs to
                a stable partition (at most `|definers|` rounds).
            wl_blend_lambda: Blend weight `λ ∈ [0, 1]` consulted only when
                `wl_integration="blend"`; the combined cost is
                `(1 - λ)·property_cost + λ·structural_term`. Defaults to
                `0.5`. Ignored under `"tie_break"`.

        Raises:
            ValueError: If `custom_metrics` contains an unsupported schema
                type, collides with a built-in metric name,
                `feedback_style` is not a registered style,
                `description_style` is not a registered style,
                `id_disambiguation` / `wl_integration` is not a registered
                value, `wl_rounds` is negative or not an int/None, or
                `wl_blend_lambda` is not a finite float in `[0, 1]`.
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
        if referential_feedback not in ("literal", "semantic"):
            raise ValueError(
                "referential_feedback must be 'literal' or 'semantic', "
                f"got {referential_feedback!r}"
            )
        self.referential_feedback_default = referential_feedback
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

        if id_disambiguation not in ("none", "wl"):
            raise ValueError(
                f"id_disambiguation must be 'none' or 'wl', got {id_disambiguation!r}"
            )
        if wl_integration not in ("tie_break", "blend"):
            raise ValueError(
                f"wl_integration must be 'tie_break' or 'blend', got {wl_integration!r}"
            )
        if wl_rounds is None:
            wr = None
        else:
            if isinstance(wl_rounds, bool):
                raise ValueError("wl_rounds must be an int or None")
            try:
                wr = int(wl_rounds)
            except (TypeError, ValueError) as e:
                raise ValueError("wl_rounds must be an int or None") from e
            if wr < 0:
                raise ValueError(f"wl_rounds must be >= 0, got {wl_rounds!r}")
        try:
            bl = float(wl_blend_lambda)
        except (TypeError, ValueError) as e:
            raise ValueError("wl_blend_lambda must be a real number") from e
        if not np.isfinite(bl) or not (0.0 <= bl <= 1.0):
            raise ValueError(
                f"wl_blend_lambda must be finite and in [0, 1], got {wl_blend_lambda!r}"
            )
        self._id_disambiguation = id_disambiguation
        self._wl_integration = wl_integration
        self._wl_rounds = wr
        self._wl_blend_lambda = bl

        self._validate_importance_sums(schema)
        self._validate_null_scores(schema)
        self._id_scopes, self._scope_order = self._collect_id_scopes(schema)
        self._validator = validator_for(schema)(schema)


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
        referential_feedback=None,
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
            referential_feedback: Override the constructor
                `referential_feedback` (`"literal"` / `"semantic"`).
                `None` defers to the instance default.

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
        ref_mode = (
            self.referential_feedback_default
            if referential_feedback is None
            else referential_feedback
        )
        ref_endpoints = None
        if ref_mode == "semantic" and self._id_scopes:
            ref_endpoints = self._build_ref_endpoint_descriptors(
                gold, pred, repair_result.ops
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
            referential_feedback=ref_mode,
            ref_endpoints=ref_endpoints,
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
        referential_feedback=None,
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
            referential_feedback: See `feedback()`.

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
        ref_mode = (
            self.referential_feedback_default
            if referential_feedback is None
            else referential_feedback
        )
        ref_endpoints = None
        if ref_mode == "semantic" and self._id_scopes:
            ref_endpoints = self._build_ref_endpoint_descriptors(
                gold, pred, repair_result.ops
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
            referential_feedback=ref_mode,
            ref_endpoints=ref_endpoints,
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

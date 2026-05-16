"""Plain-English description rendering.

Turns a match tree produced by ``ObjectAligner.align()`` into a
deterministic, indented English walk of the alignment plus a flat list of
structured per-node entries — *no LLM*.

The module is a *renderer*, not an analyser: every value it surfaces
(score, gold, pred, container outcomes) is read off the
``MatchItem`` / ``MatchList`` / ``MatchDict`` nodes that ``align`` already
produced.

Two style presets are supported:

* ``"default"`` — multi-line, indented English walk of the match tree.
* ``"json"`` — bypasses the prose surface; ``DescriptionResult.text`` is
  empty but ``.entries`` is populated with one ``DescriptionEntry`` per
  visited node (including matched ones).

Templates are validated at construction time via
``object_aligner._templates.validate_templates``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from object_aligner._templates import _load_packaged_template, validate_templates
from object_aligner.attribution import _encode_pointer_token, _join_path

# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------

__all__ = [
    "DescriptionEntry",
    "DescriptionResult",
    "render_description",
    "DEFAULT_DESCRIPTION_TEMPLATES",
]


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_VALID_STYLES = ("default", "json")


# -----------------------------------------------------------------------------
# Default templates and placeholder allowlist
# -----------------------------------------------------------------------------

# Default templates ship as TOML data under ``src/object_aligner/templates/``.
# The Python source only holds the placeholder allowlist (the renderer's
# API contract).
DEFAULT_DESCRIPTION_TEMPLATES = _load_packaged_template("describe.toml")

# Optional confidence placeholders. Available on every per-node template
# key — user-supplied overrides may include them; the shipped defaults do
# not so byte-identical output is preserved when `show_confidence=False`.
_CONFIDENCE_PLACEHOLDERS = frozenset({"confidence", "confidence_pct", "confidence_suffix"})


def _node_placeholders(*names) -> frozenset:
    return frozenset(names) | _CONFIDENCE_PLACEHOLDERS


# Placeholders allowed in each template key. Used by construction-time
# validation to reject typos in user-supplied overrides. Keep in sync with
# the rendering code below.
_DESCRIPTION_PLACEHOLDERS = {
    "describe.intro.perfect": frozenset(),
    "describe.intro.imperfect": frozenset({"score", "score_pct"}),
    "describe.item.match": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.item.mismatch": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.ref.match": _node_placeholders("indent", "pred", "score", "score_pct"),
    "describe.ref.mismatch": _node_placeholders("indent", "pred", "value", "score", "score_pct"),
    "describe.ref.no_target": _node_placeholders("indent", "pred", "score", "score_pct"),
    "describe.id.match": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.id.mismatch": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.null.match": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.null.mismatch": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.list.match": _node_placeholders("indent", "score", "score_pct"),
    "describe.list.mismatch": _node_placeholders("indent", "score", "score_pct"),
    "describe.list.excess": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.list.missing": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    # Opt-in: emitted only when render_description(include_ambiguous=True)
    # and the node's confidence falls below the threshold.
    "describe.list.ambiguous": frozenset({
        "indent", "confidence", "confidence_pct", "n_gold", "n_pred",
    }),
    "describe.dict.match": _node_placeholders("indent", "score", "score_pct"),
    "describe.dict.mismatch": _node_placeholders("indent", "score", "score_pct"),
    "describe.dict.ambiguous": frozenset({
        "indent", "confidence", "confidence_pct",
    }),
    "describe.dict.key.match": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.dict.key.mismatch": _node_placeholders("indent", "gold", "pred", "score", "score_pct"),
    "describe.dict.value.prefix": frozenset({"indent"}),
    "describe.validation_error": frozenset({"path", "message"}),
}

# Self-check: surface typos in the shipped describe.toml at import time
# rather than at first render.
validate_templates(
    DEFAULT_DESCRIPTION_TEMPLATES,
    DEFAULT_DESCRIPTION_TEMPLATES,
    _DESCRIPTION_PLACEHOLDERS,
    kind="description",
)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DescriptionEntry:
    """One visited match-tree node plus its rendered prose line.

    Emitted in match-tree traversal order — *not* ranked by score. Every
    node visited produces an entry, including matched ones (this is what
    differentiates ``describe`` from ``feedback``).

    Attributes:
        path: Indicative JSON Pointer at this node. For ``order: "align"``
            lists the path stops at the list itself (children share the
            list path) because Hungarian-matched indices are not stable;
            for fixed/prefix lists the index is included.
        depth: 0-indexed nesting depth (matches the visual indent depth).
        match_kind: One of ``"item"``, ``"list"``, ``"dict"``, ``"key"``,
            ``"ref"``, ``"id"``, or ``"ambiguous"`` (opt-in low-confidence
            container marker emitted only when
            ``include_ambiguous=True``).
        outcome: One of ``"match"``, ``"mismatch"``, ``"excess"``,
            ``"missing"``, ``"ambiguous"``, ``"no_target"`` (the last is
            ref-only — emitted when the gold referent has no counterpart
            in the prediction under the derived bijection).
        score: Similarity in ``[0, 1]`` at this node.
        text: Rendered template body for this node. May be ``""`` for
            silenced templates (default ``describe.id.match`` /
            ``describe.id.mismatch`` are empty).
        confidence: Stability of the pairing that produced this node in
            ``[0, 1]``. Inherited from the originating Match node and
            always populated. ``1.0`` everywhere when
            ``compute_confidence=False`` on the owning ``ObjectAligner``.
    """

    path: str
    depth: int
    match_kind: str
    outcome: str
    score: float
    text: str
    confidence: float = 1.0


@dataclass(frozen=True)
class DescriptionResult:
    """Result of a single `render_description()` call.

    ``str(result)`` returns ``result.text``. Iterable: ``for entry in
    result`` yields ``DescriptionEntry``s in match-tree traversal order.

    Attributes:
        score: Overall similarity in ``[0, 1]``.
        text: Fully rendered description string (empty in ``"json"``
            style).
        entries: Structured backing entries, in traversal order.
        style: The style name used to render (``"default"`` or
            ``"json"``).
    """

    score: float
    text: str
    entries: tuple = field(default_factory=tuple)
    style: str = "default"

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def __str__(self):
        return self.text

    def to_dict(self) -> dict:
        """Serialize to a basic-types dict — usable as JSON.

        Returns:
            Dict with the same shape as ``DescriptionResult`` but with
            ``entries`` as a list of dicts. ``gold`` / ``pred`` are not
            included on entries (they appear in ``text``); paths and
            scores are.
        """
        return {
            "score": self.score,
            "text": self.text,
            "entries": [
                {
                    "path": e.path,
                    "depth": e.depth,
                    "match_kind": e.match_kind,
                    "outcome": e.outcome,
                    "score": e.score,
                    "text": e.text,
                    "confidence": e.confidence,
                }
                for e in self.entries
            ],
            "style": self.style,
        }


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _validate_style(style: str) -> None:
    if style not in _VALID_STYLES:
        raise ValueError(
            f"style must be one of {_VALID_STYLES!r}, got {style!r}"
        )


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def merge_description_templates(user_templates) -> dict:
    """Validate user overrides and merge with defaults.

    Public-by-convention helper that ``ObjectAligner`` calls at
    construction time. Returns a fully populated dict (defaults overlaid
    with user overrides).
    """
    return validate_templates(
        user_templates,
        DEFAULT_DESCRIPTION_TEMPLATES,
        _DESCRIPTION_PLACEHOLDERS,
        kind="description",
    )


def _path2str(path_parts) -> str:
    """Render a path of (key|index) tokens as an RFC 6901 JSON Pointer.

    Uses the shared ``_encode_pointer_token`` / ``_join_path`` helpers from
    ``attribution.py`` so that paths emitted by ``DescriptionEntry`` line
    up byte-for-byte with the ones from ``AttributionEntry`` / ``RepairOp``.
    """
    out = ""
    for p in path_parts:
        out = _join_path(out, p)
    return out


def _list_child_path(parent_path: str, index: int, list_kind: str) -> str:
    """Compute child path for a list item.

    For Hungarian-aligned lists (``"reorder"`` / ``"combined"``) the
    indices in the match tree are post-matching and not stable into the
    original ``pred`` array, so children share the list path. For
    ``"fixed"`` / ``"prefix"`` lists the original index is appended.
    """
    if list_kind in ("reorder", "combined"):
        return parent_path
    return _join_path(parent_path, index)


def _dict_child_path(parent_path: str, key: Any) -> str:
    return _join_path(parent_path, key)


# -----------------------------------------------------------------------------
# Walker
# -----------------------------------------------------------------------------

def _confidence_suffix(confidence: float, show: bool) -> str:
    """Banded suffix string for the optional ``{confidence_suffix}`` placeholder.

    Emits ``""`` when off (preserving byte-identical output) or when the
    confidence is high (``>= 0.70``). Otherwise emits a parenthetical
    tag that escalates phrasing as confidence drops.
    """
    if not show:
        return ""
    c = float(confidence)
    if c >= 0.70:
        return ""
    if c >= 0.40:
        return f" (confidence {c:.2f})"
    return f" (low confidence {c:.2f})"


def _apply_suffix(text: str, suffix: str) -> str:
    """Inject ``suffix`` immediately before the trailing newline of ``text``.

    Most describe templates end in ``\\n``. Inserting before the newline
    keeps the prose flowing visually and matches what a hand-written
    template-with-suffix would produce.
    """
    if not suffix:
        return text
    if text.endswith("\n"):
        return text[:-1] + suffix + "\n"
    return text + suffix


def _walk(
    aligned,
    *,
    path: str,
    depth: int,
    templates: dict,
    entries: list,
    show_confidence: bool = False,
    include_ambiguous: bool = False,
    ambiguity_threshold: float = 0.30,
) -> str:
    """Recursively render a match node, append entries, return prose."""
    # Late import to avoid a circular dependency: describe.py is imported by
    # object_aligner.py which defines these dataclasses.
    from object_aligner.object_aligner import MatchDict, MatchItem, MatchList

    indent = "  " * depth
    child_indent = "  " * (depth + 1)

    if isinstance(aligned, MatchItem):
        match_kind = getattr(aligned, "kind", "") or "item"
        if aligned.score == 1.0:
            outcome = "match"
        elif match_kind == "ref" and (
            aligned.aux is None
            or aligned.aux.get("mapped_pred") is None
        ):
            # The gold referent has no counterpart in the prediction under
            # the derived bijection. Route to the dedicated template so the
            # user-facing text does not surface a gold-space id.
            outcome = "no_target"
        else:
            outcome = "mismatch"
        template_key = f"describe.{match_kind}.{outcome}"
        mapped_pred = (
            aligned.aux.get("mapped_pred")
            if (match_kind == "ref" and aligned.aux is not None)
            else None
        )
        text = templates[template_key].format(
            indent=indent,
            gold=aligned.gold,
            pred=aligned.pred,
            value=mapped_pred,
            score=aligned.score,
            score_pct=_pct(aligned.score),
            confidence=float(aligned.confidence),
            confidence_pct=_pct(float(aligned.confidence)),
            confidence_suffix=_confidence_suffix(aligned.confidence, show_confidence),
        )
        # If the template did not consume {confidence_suffix}, inject the
        # banded suffix here (preserves byte-identical output when off).
        if "{confidence_suffix}" not in templates[template_key]:
            text = _apply_suffix(text, _confidence_suffix(aligned.confidence, show_confidence))
        entries.append(DescriptionEntry(
            path=path,
            depth=depth,
            match_kind=match_kind,
            outcome=outcome,
            score=float(aligned.score),
            text=text,
            confidence=float(aligned.confidence),
        ))
        return text

    if isinstance(aligned, MatchList):
        outcome = "match" if aligned.score == 1.0 else "mismatch"
        template_key = f"describe.list.{outcome}"
        header = templates[template_key].format(
            indent=indent,
            score=aligned.score,
            score_pct=_pct(aligned.score),
            confidence=float(aligned.confidence),
            confidence_pct=_pct(float(aligned.confidence)),
            confidence_suffix=_confidence_suffix(aligned.confidence, show_confidence),
        )
        if "{confidence_suffix}" not in templates[template_key]:
            header = _apply_suffix(header, _confidence_suffix(aligned.confidence, show_confidence))
        entries.append(DescriptionEntry(
            path=path,
            depth=depth,
            match_kind="list",
            outcome=outcome,
            score=float(aligned.score),
            text=header,
            confidence=float(aligned.confidence),
        ))
        fragments = [header]
        list_kind = getattr(aligned, "kind", "") or ""
        # Opt-in low-confidence container note. Emitted at child_indent so
        # it visually sits under the header. Only for Hungarian-paired
        # list aggregators; fixed/prefix/combined are not flagged here
        # because their confidence is just a child average.
        if (
            include_ambiguous
            and list_kind == "reorder"
            and float(aligned.confidence) < float(ambiguity_threshold)
        ):
            n_gold = sum(
                1 for c in aligned.children
                if not (isinstance(c, MatchItem) and c.gold is None)
            )
            n_pred = sum(
                1 for c in aligned.children
                if not (isinstance(c, MatchItem) and c.pred is None)
            )
            amb_text = templates["describe.list.ambiguous"].format(
                indent=child_indent,
                confidence=float(aligned.confidence),
                confidence_pct=_pct(float(aligned.confidence)),
                n_gold=n_gold,
                n_pred=n_pred,
            )
            entries.append(DescriptionEntry(
                path=path,
                depth=depth + 1,
                match_kind="ambiguous",
                outcome="ambiguous",
                score=float(aligned.score),
                text=amb_text,
                confidence=float(aligned.confidence),
            ))
            fragments.append(amb_text)
        for index, child in enumerate(aligned.children):
            child_path = _list_child_path(path, index, list_kind)
            if isinstance(child, MatchItem) and child.gold is None and child.pred is None:
                # Sentinel from _align_lists_prefix when both sides are
                # shorter than prefixItems at this position; skip silently.
                continue
            if isinstance(child, MatchItem) and child.gold is None:
                text = templates["describe.list.excess"].format(
                    indent=child_indent,
                    pred=child.pred,
                    gold=child.gold,
                    score=child.score,
                    score_pct=_pct(child.score),
                    confidence=float(child.confidence),
                    confidence_pct=_pct(float(child.confidence)),
                    confidence_suffix=_confidence_suffix(child.confidence, show_confidence),
                )
                if "{confidence_suffix}" not in templates["describe.list.excess"]:
                    text = _apply_suffix(text, _confidence_suffix(child.confidence, show_confidence))
                entries.append(DescriptionEntry(
                    path=child_path,
                    depth=depth + 1,
                    match_kind="item",
                    outcome="excess",
                    score=float(child.score),
                    text=text,
                    confidence=float(child.confidence),
                ))
                fragments.append(text)
                continue
            if isinstance(child, MatchItem) and child.pred is None:
                text = templates["describe.list.missing"].format(
                    indent=child_indent,
                    gold=child.gold,
                    pred=child.pred,
                    score=child.score,
                    score_pct=_pct(child.score),
                    confidence=float(child.confidence),
                    confidence_pct=_pct(float(child.confidence)),
                    confidence_suffix=_confidence_suffix(child.confidence, show_confidence),
                )
                if "{confidence_suffix}" not in templates["describe.list.missing"]:
                    text = _apply_suffix(text, _confidence_suffix(child.confidence, show_confidence))
                entries.append(DescriptionEntry(
                    path=child_path,
                    depth=depth + 1,
                    match_kind="item",
                    outcome="missing",
                    score=float(child.score),
                    text=text,
                    confidence=float(child.confidence),
                ))
                fragments.append(text)
                continue
            fragments.append(_walk(
                child,
                path=child_path,
                depth=depth + 1,
                templates=templates,
                entries=entries,
                show_confidence=show_confidence,
                include_ambiguous=include_ambiguous,
                ambiguity_threshold=ambiguity_threshold,
            ))
        return "".join(fragments)

    if isinstance(aligned, MatchDict):
        outcome = "match" if aligned.score == 1.0 else "mismatch"
        template_key = f"describe.dict.{outcome}"
        header = templates[template_key].format(
            indent=indent,
            score=aligned.score,
            score_pct=_pct(aligned.score),
            confidence=float(aligned.confidence),
            confidence_pct=_pct(float(aligned.confidence)),
            confidence_suffix=_confidence_suffix(aligned.confidence, show_confidence),
        )
        if "{confidence_suffix}" not in templates[template_key]:
            header = _apply_suffix(header, _confidence_suffix(aligned.confidence, show_confidence))
        entries.append(DescriptionEntry(
            path=path,
            depth=depth,
            match_kind="dict",
            outcome=outcome,
            score=float(aligned.score),
            text=header,
            confidence=float(aligned.confidence),
        ))
        fragments = [header]
        if (
            include_ambiguous
            and float(aligned.confidence) < float(ambiguity_threshold)
        ):
            amb_text = templates["describe.dict.ambiguous"].format(
                indent=child_indent,
                confidence=float(aligned.confidence),
                confidence_pct=_pct(float(aligned.confidence)),
            )
            entries.append(DescriptionEntry(
                path=path,
                depth=depth + 1,
                match_kind="ambiguous",
                outcome="ambiguous",
                score=float(aligned.score),
                text=amb_text,
                confidence=float(aligned.confidence),
            ))
            fragments.append(amb_text)
        for key, child in aligned.children.items():
            key_outcome = "match" if key.score == 1.0 else "mismatch"
            key_template_key = f"describe.dict.key.{key_outcome}"
            child_path = _dict_child_path(path, key.gold if key.gold is not None else key.pred)
            key_text = templates[key_template_key].format(
                indent=child_indent,
                gold=key.gold,
                pred=key.pred,
                score=key.score,
                score_pct=_pct(key.score),
                confidence=float(key.confidence),
                confidence_pct=_pct(float(key.confidence)),
                confidence_suffix=_confidence_suffix(key.confidence, show_confidence),
            )
            if "{confidence_suffix}" not in templates[key_template_key]:
                key_text = _apply_suffix(key_text, _confidence_suffix(key.confidence, show_confidence))
            entries.append(DescriptionEntry(
                path=child_path,
                depth=depth + 1,
                match_kind="key",
                outcome=key_outcome,
                score=float(key.score),
                text=key_text,
                confidence=float(key.confidence),
            ))
            fragments.append(key_text)
            value_prefix = templates["describe.dict.value.prefix"].format(
                indent=child_indent,
            )
            child_prose = _walk(
                child,
                path=child_path,
                depth=depth + 1,
                templates=templates,
                entries=entries,
                show_confidence=show_confidence,
                include_ambiguous=include_ambiguous,
                ambiguity_threshold=ambiguity_threshold,
            )
            fragments.append(value_prefix + child_prose.lstrip() + "\n")
        return "".join(fragments)

    raise AssertionError(f"Unknown match instance: {aligned}")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def render_description(
    match_tree,
    *,
    style: str = "default",
    templates: dict | None = None,
    show_confidence: bool = False,
    include_ambiguous: bool = False,
    ambiguity_threshold: float = 0.30,
) -> DescriptionResult:
    """Render a match tree as a deterministic English description.

    See [`docs/describe.md`](../describe.md) for the full design and
    examples.

    Args:
        match_tree: A match tree returned by ``ObjectAligner.align()``.
        style: ``"default"`` (multi-line indented prose walk of the
            match tree) or ``"json"`` (empty ``.text``, structured
            ``.entries`` only).
        templates: User template overrides (full dict, partial dict, or
            ``None`` for defaults).
        show_confidence: If ``True``, append a banded confidence suffix
            (e.g. ``" (low confidence 0.23)"``) to every per-node line
            whose confidence falls below ``0.70``. Default ``False``
            preserves byte-identical output of pre-confidence releases.
        include_ambiguous: If ``True``, emit a dedicated
            ``describe.list.ambiguous`` / ``describe.dict.ambiguous``
            entry before walking any Hungarian-paired container whose
            confidence falls below ``ambiguity_threshold``. Off by
            default.
        ambiguity_threshold: Confidence threshold for the ambiguous-entry
            emission. Default ``0.30``.

    Returns:
        ``DescriptionResult`` whose ``text`` is the fully rendered
        indented description (or ``""`` in ``"json"`` style).

    Raises:
        ValueError: If ``style`` is not a registered style.
    """
    _validate_style(style)
    if templates is None:
        templates = DEFAULT_DESCRIPTION_TEMPLATES
    elif templates is not DEFAULT_DESCRIPTION_TEMPLATES and set(templates) != set(
        DEFAULT_DESCRIPTION_TEMPLATES
    ):
        # Partial dict — merge with defaults so format() can find all keys.
        templates = merge_description_templates(templates)

    score = float(match_tree.score)

    if style == "default" and score == 1.0:
        return DescriptionResult(
            score=score,
            text=templates["describe.intro.perfect"],
            entries=(),
            style=style,
        )

    entries: list = []
    body = _walk(
        match_tree,
        path="",
        depth=0,
        templates=templates,
        entries=entries,
        show_confidence=show_confidence,
        include_ambiguous=include_ambiguous,
        ambiguity_threshold=ambiguity_threshold,
    )

    if style == "json":
        return DescriptionResult(
            score=score,
            text="",
            entries=tuple(entries),
            style=style,
        )

    intro = templates["describe.intro.imperfect"].format(
        score=score,
        score_pct=_pct(score),
    )
    return DescriptionResult(
        score=score,
        text=intro + body.rstrip(),
        entries=tuple(entries),
        style=style,
    )


def render_validation_error(error, templates: dict) -> DescriptionResult:
    """Render a JSON-Schema validation error as a degenerate result.

    Used by ``ObjectAligner.describe()`` and ``metric()`` when ``pred``
    fails schema validation.
    """
    text = templates["describe.validation_error"].format(
        path=_path2str(error.path),
        message=error.message,
    )
    return DescriptionResult(
        score=0.0,
        text=text,
        entries=(),
        style="default",
    )

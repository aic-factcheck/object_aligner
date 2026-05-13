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

# Placeholders allowed in each template key. Used by construction-time
# validation to reject typos in user-supplied overrides. Keep in sync with
# the rendering code below.
_DESCRIPTION_PLACEHOLDERS = {
    "describe.intro.perfect": frozenset(),
    "describe.intro.imperfect": frozenset({"score", "score_pct"}),
    "describe.item.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.item.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.ref.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.ref.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.id.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.id.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.list.match": frozenset({"indent", "score", "score_pct"}),
    "describe.list.mismatch": frozenset({"indent", "score", "score_pct"}),
    "describe.list.excess": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.list.missing": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.dict.match": frozenset({"indent", "score", "score_pct"}),
    "describe.dict.mismatch": frozenset({"indent", "score", "score_pct"}),
    "describe.dict.key.match": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
    "describe.dict.key.mismatch": frozenset({"indent", "gold", "pred", "score", "score_pct"}),
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
            ``"ref"``, ``"id"``.
        outcome: One of ``"match"``, ``"mismatch"``, ``"excess"``,
            ``"missing"``.
        score: Similarity in ``[0, 1]`` at this node.
        text: Rendered template body for this node. May be ``""`` for
            silenced templates (default ``describe.id.match`` /
            ``describe.id.mismatch`` are empty).
    """

    path: str
    depth: int
    match_kind: str
    outcome: str
    score: float
    text: str


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

def _walk(
    aligned,
    *,
    path: str,
    depth: int,
    templates: dict,
    entries: list,
) -> str:
    """Recursively render a match node, append entries, return prose."""
    # Late import to avoid a circular dependency: describe.py is imported by
    # object_aligner.py which defines these dataclasses.
    from object_aligner.object_aligner import MatchDict, MatchItem, MatchList

    indent = "  " * depth
    child_indent = "  " * (depth + 1)

    if isinstance(aligned, MatchItem):
        match_kind = getattr(aligned, "kind", "") or "item"
        outcome = "match" if aligned.score == 1.0 else "mismatch"
        template_key = f"describe.{match_kind}.{outcome}"
        text = templates[template_key].format(
            indent=indent,
            gold=aligned.gold,
            pred=aligned.pred,
            score=aligned.score,
            score_pct=_pct(aligned.score),
        )
        entries.append(DescriptionEntry(
            path=path,
            depth=depth,
            match_kind=match_kind,
            outcome=outcome,
            score=float(aligned.score),
            text=text,
        ))
        return text

    if isinstance(aligned, MatchList):
        outcome = "match" if aligned.score == 1.0 else "mismatch"
        template_key = f"describe.list.{outcome}"
        header = templates[template_key].format(
            indent=indent,
            score=aligned.score,
            score_pct=_pct(aligned.score),
        )
        entries.append(DescriptionEntry(
            path=path,
            depth=depth,
            match_kind="list",
            outcome=outcome,
            score=float(aligned.score),
            text=header,
        ))
        fragments = [header]
        list_kind = getattr(aligned, "kind", "") or ""
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
                )
                entries.append(DescriptionEntry(
                    path=child_path,
                    depth=depth + 1,
                    match_kind="item",
                    outcome="excess",
                    score=float(child.score),
                    text=text,
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
                )
                entries.append(DescriptionEntry(
                    path=child_path,
                    depth=depth + 1,
                    match_kind="item",
                    outcome="missing",
                    score=float(child.score),
                    text=text,
                ))
                fragments.append(text)
                continue
            fragments.append(_walk(
                child,
                path=child_path,
                depth=depth + 1,
                templates=templates,
                entries=entries,
            ))
        return "".join(fragments)

    if isinstance(aligned, MatchDict):
        outcome = "match" if aligned.score == 1.0 else "mismatch"
        template_key = f"describe.dict.{outcome}"
        header = templates[template_key].format(
            indent=indent,
            score=aligned.score,
            score_pct=_pct(aligned.score),
        )
        entries.append(DescriptionEntry(
            path=path,
            depth=depth,
            match_kind="dict",
            outcome=outcome,
            score=float(aligned.score),
            text=header,
        ))
        fragments = [header]
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
            )
            entries.append(DescriptionEntry(
                path=child_path,
                depth=depth + 1,
                match_kind="key",
                outcome=key_outcome,
                score=float(key.score),
                text=key_text,
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

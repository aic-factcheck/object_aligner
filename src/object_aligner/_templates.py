"""Shared template-validation and -loading helpers.

* ``validate_templates`` — used by both ``_merge_reasoning_templates`` (in
  ``object_aligner.py``) and the feedback-template merger (in
  ``feedback.py``). Validates a user-provided overrides mapping against a
  default dict and a per-key allowed-placeholder table, then returns a
  merged dict (defaults overlaid by overrides).
* ``_load_packaged_template`` — loads a TOML template file that ships with
  the package (under ``object_aligner/templates/``). Used at module import
  time to populate the ``DEFAULT_*_TEMPLATES`` constants.
* ``load_templates_from_toml`` — public helper: load templates from a
  user-supplied TOML file. Returns a flat ``dict[str, str]`` suitable for
  passing as ``reasoning_templates`` or ``feedback_templates``.
"""

from __future__ import annotations

import string
import tomllib
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Validation (existing public-by-convention helper used by ObjectAligner)
# -----------------------------------------------------------------------------

def validate_templates(
    overrides,
    defaults: Mapping[str, str],
    placeholders_by_key: Mapping[str, frozenset],
    *,
    kind: str,
) -> dict:
    """Merge ``overrides`` over ``defaults`` with full validation.

    Parameters
    ----------
    overrides
        Mapping of template-key -> template-string. May be ``None``.
    defaults
        The default template dict for this template family. Defines the
        allowed key set.
    placeholders_by_key
        Mapping from each template key to the set of placeholder names
        permitted in that template.
    kind
        Either ``"reasoning"`` or ``"feedback"`` — used only in error
        messages so the caller doesn't have to format them.

    Returns
    -------
    dict
        Merged dict: ``{**defaults, **overrides}``.
    """
    if overrides is None:
        return dict(defaults)
    if not isinstance(overrides, Mapping):
        raise TypeError(
            f"{kind}_templates must be a mapping of template keys to strings"
        )

    overrides = dict(overrides)
    unknown_keys = sorted(set(overrides) - set(defaults))
    if unknown_keys:
        raise ValueError(f"Unknown {kind} template keys: {unknown_keys}")

    formatter = string.Formatter()
    for key, template in overrides.items():
        if not isinstance(template, str):
            raise TypeError(
                f'{kind}_templates["{key}"] must be a string, '
                f"got {type(template).__name__}"
            )
        allowed = placeholders_by_key[key]
        used = {name for _, name, _, _ in formatter.parse(template) if name}
        extra = used - allowed
        if extra:
            raise ValueError(
                f'{kind}_templates["{key}"] uses unknown placeholder(s) '
                f"{sorted(extra)}; allowed: {sorted(allowed)}"
            )

    merged = dict(defaults)
    merged.update(overrides)
    return merged


# -----------------------------------------------------------------------------
# TOML loading
# -----------------------------------------------------------------------------

def _coerce_to_string_dict(data: Any, *, source: str) -> dict[str, str]:
    """Flatten any nested-table structure and type-check leaf values.

    Supports both styles in user files:

    * Flat (recommended, matches the package files):
      ``"feedback.op.key_add" = "..."``
    * Nested (TOML tables): ``[feedback.op]`` then ``key_add = "..."``

    Both produce the same flat ``{"feedback.op.key_add": "..."}`` dict.
    """
    flat: dict[str, str] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
            return
        if isinstance(obj, str):
            flat[prefix] = obj
            return
        raise TypeError(
            f"template value at key {prefix!r} in {source} must be a string, "
            f"got {type(obj).__name__}"
        )

    walk("", data)
    return flat


def _load_packaged_template(name: str) -> dict[str, str]:
    """Load a TOML template file that ships with the package.

    Called at module-import time by ``object_aligner.py`` and ``feedback.py``
    to populate the ``DEFAULT_*_TEMPLATES`` constants. Resolves the file via
    ``importlib.resources`` so it works from installed wheels as well as
    source checkouts.
    """
    resource = files("object_aligner.templates").joinpath(name)
    raw = tomllib.loads(resource.read_text(encoding="utf-8"))
    return _coerce_to_string_dict(raw, source=f"package: {name}")


def load_templates_from_toml(path) -> dict[str, str]:
    """Load templates from a user-supplied TOML file.

    Accepts both flat (`"feedback.op.key_add" = "..."`) and nested-table
    (`[feedback.op]` ... `key_add = "..."`) styles. Does **not** validate
    keys or placeholders — that happens when `ObjectAligner` merges the
    dict against the defaults for its template family. Bad keys or
    placeholders surface as `ValueError` from the constructor; bad TOML
    or bad value types surface here.

    Args:
        path: Path-like pointing to a TOML file containing template
            overrides.

    Returns:
        A flat `{template_key: template_string}` dict suitable for
        passing as `feedback_templates=` or `reasoning_templates=` to
        `ObjectAligner(...)`.

    Raises:
        FileNotFoundError: `path` does not exist.
        tomllib.TOMLDecodeError: The file is not valid TOML.
        TypeError: A value in the file is not a string.
    """
    p = Path(path)
    with p.open("rb") as f:
        raw = tomllib.load(f)
    return _coerce_to_string_dict(raw, source=str(p))

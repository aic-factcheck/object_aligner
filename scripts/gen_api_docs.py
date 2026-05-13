"""Generate `docs/api.md` as plain Markdown from `object_aligner` docstrings.

Stdlib-only. Walks every name in `object_aligner.__all__`, introspects the
symbol with `inspect` / `dataclasses`, parses its Google-style docstring
(`Args:` / `Returns:` / `Raises:` / `Attributes:`), and writes a Markdown
file readable on GitHub without any build step.

Run: ``uv run python scripts/gen_api_docs.py``

The output is idempotent: running twice produces no diff.
"""

from __future__ import annotations

import dataclasses
import inspect
import io
import re
import sys
import textwrap
from pathlib import Path

# Make `src/` importable when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import object_aligner  # noqa: E402

OUTPUT = REPO_ROOT / "docs" / "api.md"

# Feature grouping. Each entry is (group title, list of __all__ names in
# the order we want them rendered). Names not in any group are appended at
# the end under "Other".
GROUPS: list[tuple[str, list[str]]] = [
    ("Aligner", ["ObjectAligner"]),
    ("Match tree", ["MatchItem", "MatchList", "MatchDict"]),
    ("Templates", ["load_templates_from_toml"]),
    (
        "Attribution",
        ["tree_walk_attribution", "AttributionEntry", "AttributionResult"],
    ),
    ("Repair", ["generate_repairs", "RepairOp", "RepairResult"]),
    (
        "Description",
        [
            "render_description",
            "DescriptionEntry",
            "DescriptionResult",
            "DEFAULT_DESCRIPTION_TEMPLATES",
        ],
    ),
    (
        "Feedback",
        [
            "render_feedback",
            "FeedbackEntry",
            "FeedbackResult",
            "DEFAULT_FEEDBACK_TEMPLATES",
        ],
    ),
]

# ObjectAligner public methods, in source order. Anything not in this list
# is treated as private and skipped.
OBJECT_ALIGNER_METHODS: list[str] = [
    "align",
    "metric",
    "attribute",
    "attribute_from_match",
    "repair",
    "repair_from_match",
    "describe",
    "describe_from_match",
    "feedback",
    "feedback_from_match",
]


# ---------------------------------------------------------------------------
# Docstring parser (Google-style)
# ---------------------------------------------------------------------------

_SECTION_HEADERS = ("Args:", "Returns:", "Raises:", "Attributes:")


def parse_docstring(doc: str | None) -> dict:
    """Return {summary, body, args, returns, raises, attributes}.

    - summary: first paragraph (joined onto one line).
    - body: remaining narrative paragraphs above the first section header.
    - args / raises / attributes: ordered list of (name, description-lines).
    - returns: list of description-lines.
    """
    out: dict = {
        "summary": "",
        "body": "",
        "args": [],
        "returns": [],
        "raises": [],
        "attributes": [],
    }
    if not doc:
        return out

    lines = doc.splitlines()

    # Split into "pre-sections narrative" and per-section blocks.
    narrative: list[str] = []
    sections: dict[str, list[str]] = {h: [] for h in _SECTION_HEADERS}
    current: str | None = None
    for ln in lines:
        stripped = ln.strip()
        if stripped in _SECTION_HEADERS:
            current = stripped
            continue
        if current is None:
            narrative.append(ln)
        else:
            sections[current].append(ln)

    # Narrative: first paragraph is summary, rest is body.
    narrative_text = "\n".join(narrative).strip("\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", narrative_text) if p.strip()]
    if paragraphs:
        out["summary"] = " ".join(paragraphs[0].splitlines()).strip()
        out["body"] = "\n\n".join(paragraphs[1:])

    # Parameter-style sections (Args / Raises / Attributes).
    for header, key in (
        ("Args:", "args"),
        ("Raises:", "raises"),
        ("Attributes:", "attributes"),
    ):
        out[key] = _parse_param_block(sections[header])

    # Returns: just a list of stripped description lines (one paragraph).
    rtn_lines = [ln for ln in sections["Returns:"] if ln.strip()]
    if rtn_lines:
        # Detect minimum indent and dedent so the paragraph reads naturally.
        out["returns"] = textwrap.dedent("\n".join(rtn_lines)).strip()

    return out


def _parse_param_block(block_lines: list[str]) -> list[tuple[str, str]]:
    """Parse an Args:/Raises:/Attributes: block into [(name, description)].

    Each entry is `name: description`; continuation lines have deeper
    indentation than the entry line. The block as a whole is indented
    one level under the section header.
    """
    if not block_lines:
        return []

    # Strip the common leading indent off the whole block first.
    dedented = textwrap.dedent("\n".join(block_lines))
    text_lines = dedented.splitlines()

    entries: list[tuple[str, str]] = []
    current_name: str | None = None
    current_desc: list[str] = []

    # Allow dotted names (e.g. `tomllib.TOMLDecodeError`) in Raises:/Attributes:.
    entry_re = re.compile(r"^([\w.]+)\s*:\s*(.*)$")

    def flush():
        if current_name is not None:
            desc = " ".join(s.strip() for s in current_desc if s.strip())
            entries.append((current_name, desc))

    for ln in text_lines:
        if not ln.strip():
            # Blank line is allowed between entries; treat as continuation
            # delimiter without breaking the current entry.
            continue
        # An entry starts at column 0 of the dedented block. A continuation
        # line has leading whitespace.
        if ln[0].isspace():
            current_desc.append(ln)
            continue
        m = entry_re.match(ln)
        if not m:
            # Not an entry header — append to current description.
            current_desc.append(ln)
            continue
        # New entry. Flush previous.
        flush()
        current_name = m.group(1)
        first_desc = m.group(2).strip()
        current_desc = [first_desc] if first_desc else []
    flush()
    return entries


# ---------------------------------------------------------------------------
# Type / default formatting
# ---------------------------------------------------------------------------


def fmt_type(t) -> str:
    """Render a dataclass field type or annotation for the field table."""
    if t is None or t is type(None):
        return "None"
    if isinstance(t, str):
        return t
    name = getattr(t, "__name__", None)
    if name:
        return name
    return repr(t)


def fmt_default(field: dataclasses.Field) -> str:
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            sample = field.default_factory()
        except Exception:
            return f"{field.default_factory.__name__}()"
        return f"{field.default_factory.__name__}() → {sample!r}"
    return "*(required)*"


def fmt_signature(obj, *, qualname: str | None = None) -> str:
    """Build a one-line Python signature string for a callable.

    Always drops a leading `self` parameter — both when rendering a method
    against its class (`Class.method(...)`) and when rendering the
    constructor of a non-dataclass class (we want `Class(arg, ...)` rather
    than `Class(self, arg, ...)`).
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{qualname or obj.__name__}(...)"
    params = [p for n, p in sig.parameters.items() if n != "self"]
    sig = sig.replace(parameters=params)
    return f"{qualname or obj.__name__}{sig}"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def emit_heading(buf: io.StringIO, level: int, label_md: str, anchor: str) -> None:
    """Emit a heading plus a machine-readable anchor comment.

    The comment line `<!-- anchor: foo -->` lets cross-link maintenance grep
    for the canonical slug without running mkdocs.
    """
    hashes = "#" * level
    buf.write(f"{hashes} {label_md}\n")
    buf.write(f"<!-- anchor: {anchor} -->\n\n")


def emit_signature_block(buf: io.StringIO, sig: str) -> None:
    buf.write("```python\n")
    buf.write(sig)
    buf.write("\n```\n\n")


def emit_parsed_sections(buf: io.StringIO, parsed: dict) -> None:
    """Render summary/body/args/returns/raises (not attributes — that's a table)."""
    if parsed["summary"]:
        buf.write(parsed["summary"] + "\n\n")
    if parsed["body"]:
        buf.write(parsed["body"] + "\n\n")
    if parsed["args"]:
        buf.write("**Parameters**\n\n")
        for name, desc in parsed["args"]:
            buf.write(f"- **`{name}`** — {desc}\n")
        buf.write("\n")
    if parsed["returns"]:
        buf.write("**Returns** — " + parsed["returns"].replace("\n", " ") + "\n\n")
    if parsed["raises"]:
        buf.write("**Raises**\n\n")
        for exc, desc in parsed["raises"]:
            buf.write(f"- **`{exc}`** — {desc}\n")
        buf.write("\n")


def emit_attributes_table(buf: io.StringIO, cls, parsed_attrs: list[tuple[str, str]]) -> None:
    """Render the dataclass field table (Field | Type | Default | Description)."""
    desc_by_name = dict(parsed_attrs)
    buf.write("| Field | Type | Default | Description |\n")
    buf.write("|-------|------|---------|-------------|\n")
    for f in dataclasses.fields(cls):
        desc = desc_by_name.get(f.name, "")
        buf.write(
            f"| `{f.name}` | `{fmt_type(f.type)}` | {fmt_default(f)} | {desc} |\n"
        )
    buf.write("\n")


# ---------------------------------------------------------------------------
# Symbol emitters
# ---------------------------------------------------------------------------


def emit_class(buf: io.StringIO, name: str, cls) -> None:
    anchor = name.lower()
    emit_heading(buf, 3, f"`{name}`", anchor)

    parsed = parse_docstring(inspect.getdoc(cls))

    # Constructor signature is the most useful overview.
    if dataclasses.is_dataclass(cls):
        sig = fmt_signature(cls, qualname=name)
    else:
        sig = fmt_signature(cls.__init__, qualname=name)
    emit_signature_block(buf, sig)

    # For non-dataclasses, pull constructor argument / raises documentation
    # from __init__'s docstring when the class docstring is narrative-only.
    # The class signature shown above already comes from __init__, so the
    # parameter names line up.
    if not dataclasses.is_dataclass(cls):
        init_doc = inspect.getdoc(cls.__init__)
        if init_doc and init_doc != inspect.getdoc(object.__init__):
            init_parsed = parse_docstring(init_doc)
            if not parsed["args"] and init_parsed["args"]:
                parsed["args"] = init_parsed["args"]
            if not parsed["raises"] and init_parsed["raises"]:
                parsed["raises"] = init_parsed["raises"]

    emit_parsed_sections(buf, parsed)

    if dataclasses.is_dataclass(cls):
        buf.write("**Fields**\n\n")
        emit_attributes_table(buf, cls, parsed["attributes"])

    if name == "ObjectAligner":
        for method_name in OBJECT_ALIGNER_METHODS:
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            emit_method(buf, name, method_name, method)


def emit_method(buf: io.StringIO, class_name: str, method_name: str, method) -> None:
    qualname = f"{class_name}.{method_name}"
    anchor = qualname.lower().replace(".", "")
    emit_heading(buf, 4, f"`{qualname}()`", anchor)
    emit_signature_block(buf, fmt_signature(method, qualname=qualname))
    parsed = parse_docstring(inspect.getdoc(method))
    emit_parsed_sections(buf, parsed)


def emit_function(buf: io.StringIO, name: str, fn) -> None:
    anchor = name.lower()
    emit_heading(buf, 3, f"`{name}`", anchor)
    emit_signature_block(buf, fmt_signature(fn, qualname=name))
    parsed = parse_docstring(inspect.getdoc(fn))
    emit_parsed_sections(buf, parsed)


def emit_constant(buf: io.StringIO, name: str, value) -> None:
    anchor = f"{name.lower()}-constant"
    emit_heading(buf, 3, f"`{name}` (constant)", anchor)
    type_name = type(value).__name__
    note = ""
    if isinstance(value, dict):
        note = f"Dict with {len(value)} keys."
    buf.write(f"Type: `{type_name}`. {note}\n\n")
    buf.write(
        "Default template strings live under "
        "`src/object_aligner/templates/`. Import this name and pass it (or "
        "an override dict) into `ObjectAligner(...)` to customize.\n\n"
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


HEADER = """\
<!-- THIS FILE IS GENERATED by scripts/gen_api_docs.py. DO NOT EDIT BY HAND.
     To update: edit the corresponding docstring in src/object_aligner/,
     then run `uv run python scripts/gen_api_docs.py`. -->

# API Reference

[Docs](index.md) › API Reference

Public Python surface of `object_aligner`. Generated from package
docstrings — every signature, parameter description, and field table on
this page is read live from the source at generation time.

Each section's `<!-- anchor: ... -->` comment names the canonical slug
chapters use to deep-link into this page.

"""

FOOTER = """\
---

[← Documentation home](index.md)
"""


def build() -> str:
    buf = io.StringIO()
    buf.write(HEADER)

    rendered: set[str] = set()
    for group_title, names in GROUPS:
        buf.write(f"---\n\n## {group_title}\n\n")
        for name in names:
            if name not in object_aligner.__all__:
                raise RuntimeError(
                    f"{name!r} listed in GROUPS but missing from __all__"
                )
            obj = getattr(object_aligner, name)
            rendered.add(name)
            if inspect.isclass(obj):
                emit_class(buf, name, obj)
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                emit_function(buf, name, obj)
            else:
                emit_constant(buf, name, obj)

    missing = sorted(set(object_aligner.__all__) - rendered)
    if missing:
        buf.write("---\n\n## Other\n\n")
        for name in missing:
            obj = getattr(object_aligner, name)
            if inspect.isclass(obj):
                emit_class(buf, name, obj)
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                emit_function(buf, name, obj)
            else:
                emit_constant(buf, name, obj)

    buf.write(FOOTER)
    return buf.getvalue()


def main() -> None:
    content = build()
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}: {len(content.splitlines())} lines")


if __name__ == "__main__":
    main()

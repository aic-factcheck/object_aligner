"""Tests for the externalized TOML template system."""

import tomllib
from pathlib import Path

import pytest

from object_aligner import (
    DEFAULT_DESCRIPTION_TEMPLATES,
    DEFAULT_FEEDBACK_TEMPLATES,
    ObjectAligner,
    load_templates_from_toml,
)
from object_aligner._templates import (
    _coerce_to_string_dict,
    _load_packaged_template,
    validate_templates,
)
from object_aligner.describe import _DESCRIPTION_PLACEHOLDERS
from object_aligner.feedback import _COMPACT_OVERRIDES, _FEEDBACK_PLACEHOLDERS

from tests._legacy_template_snapshots import (
    LEGACY_COMPACT,
    LEGACY_DESCRIBE,
    LEGACY_FEEDBACK,
)


# -----------------------------------------------------------------------------
# Group A — packaged template loading at import time
# -----------------------------------------------------------------------------

def test_description_defaults_have_expected_keys():
    assert set(DEFAULT_DESCRIPTION_TEMPLATES) == set(_DESCRIPTION_PLACEHOLDERS)


def test_feedback_defaults_have_expected_keys():
    assert set(DEFAULT_FEEDBACK_TEMPLATES) == set(_FEEDBACK_PLACEHOLDERS)
    # 19 fix/intro/synthesis/empty/validation keys + 2 confidence keys
    # (feedback.op.pairing_ambiguous and feedback.diagnostics.intro) + 1
    # ref_fix_no_target + 8 semantic referential feedback keys.
    assert len(DEFAULT_FEEDBACK_TEMPLATES) == 30


def test_compact_overlay_only_overrides_known_keys():
    # Every key in the compact overlay must exist in the default set.
    assert set(_COMPACT_OVERRIDES) <= set(DEFAULT_FEEDBACK_TEMPLATES)
    # And it must be a strict overlay (not the full set).
    assert len(_COMPACT_OVERRIDES) < len(DEFAULT_FEEDBACK_TEMPLATES)


def test_packaged_description_defaults_pass_placeholder_validation():
    # Already runs at import; calling again as a regression net.
    validate_templates(
        DEFAULT_DESCRIPTION_TEMPLATES,
        DEFAULT_DESCRIPTION_TEMPLATES,
        _DESCRIPTION_PLACEHOLDERS,
        kind="description",
    )


def test_packaged_feedback_defaults_pass_placeholder_validation():
    validate_templates(
        DEFAULT_FEEDBACK_TEMPLATES,
        DEFAULT_FEEDBACK_TEMPLATES,
        _FEEDBACK_PLACEHOLDERS,
        kind="feedback",
    )


def test_packaged_compact_overlay_passes_placeholder_validation():
    validate_templates(
        _COMPACT_OVERRIDES,
        DEFAULT_FEEDBACK_TEMPLATES,
        _FEEDBACK_PLACEHOLDERS,
        kind="feedback",
    )


def test_load_packaged_template_handles_unknown_filename():
    # importlib.resources surfaces missing-file errors as FileNotFoundError
    # (via Traversable.read_text). We don't catch and re-raise — just
    # confirm the user sees a clear failure rather than a silent empty dict.
    with pytest.raises((FileNotFoundError, OSError)):
        _load_packaged_template("does_not_exist.toml")


# -----------------------------------------------------------------------------
# Group B — load_templates_from_toml (public helper)
# -----------------------------------------------------------------------------

def test_load_flat_style_toml(tmp_path):
    p = tmp_path / "flat.toml"
    p.write_text(
        '"feedback.op.primitive_replace" = "{rank}. flat {path}"\n'
        '"feedback.intro.perfect" = "All good"\n',
        encoding="utf-8",
    )
    loaded = load_templates_from_toml(p)
    assert loaded == {
        "feedback.op.primitive_replace": "{rank}. flat {path}",
        "feedback.intro.perfect": "All good",
    }


def test_load_nested_style_toml(tmp_path):
    p = tmp_path / "nested.toml"
    p.write_text(
        '[feedback.op]\n'
        'primitive_replace = "{rank}. nested {path}"\n'
        '[feedback.intro]\n'
        'perfect = "Yep"\n',
        encoding="utf-8",
    )
    loaded = load_templates_from_toml(p)
    assert loaded == {
        "feedback.op.primitive_replace": "{rank}. nested {path}",
        "feedback.intro.perfect": "Yep",
    }


def test_load_mixed_flat_and_nested_toml(tmp_path):
    # Documented forgiveness: both styles in the same file are flattened.
    p = tmp_path / "mixed.toml"
    p.write_text(
        '"feedback.intro.perfect" = "flat"\n'
        '\n'
        '[feedback.op]\n'
        'primitive_replace = "nested"\n',
        encoding="utf-8",
    )
    loaded = load_templates_from_toml(p)
    assert loaded == {
        "feedback.intro.perfect": "flat",
        "feedback.op.primitive_replace": "nested",
    }


def test_load_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_templates_from_toml(tmp_path / "nope.toml")


def test_load_malformed_toml_raises_decodeerror(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("this = is = not = valid TOML\n", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_templates_from_toml(p)


def test_load_non_string_value_raises_typeerror(tmp_path):
    p = tmp_path / "bad_value.toml"
    p.write_text('"feedback.intro.perfect" = 42\n', encoding="utf-8")
    with pytest.raises(TypeError, match="must be a string"):
        load_templates_from_toml(p)


def test_load_non_string_value_in_nested_table_raises_typeerror(tmp_path):
    p = tmp_path / "bad_nested.toml"
    p.write_text(
        '[feedback.op]\n'
        'primitive_replace = ["not", "a", "string"]\n',
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="must be a string"):
        load_templates_from_toml(p)


def test_loaded_dict_round_trips_through_objectaligner(tmp_path):
    # Build a minimal override TOML and feed it to ObjectAligner.
    p = tmp_path / "overrides.toml"
    p.write_text(
        '"feedback.op.primitive_replace" = "TOML-LOADED {rank}: {path}"\n',
        encoding="utf-8",
    )
    overrides = load_templates_from_toml(p)
    schema = {"type": "string", "score": "jaro"}
    aligner = ObjectAligner(schema, feedback_templates=overrides)
    fb = aligner.feedback("hello", "hallo")
    assert "TOML-LOADED" in fb.text


def test_loaded_dict_works_for_description_overrides(tmp_path):
    p = tmp_path / "describe.toml"
    p.write_text(
        '"describe.intro.perfect" = "Description override fires"\n',
        encoding="utf-8",
    )
    overrides = load_templates_from_toml(p)
    aligner = ObjectAligner(
        {"type": "string", "score": "exact"},
        description_templates=overrides,
        generate_description=True,
    )
    r = aligner.metric("a", "a")
    assert r["description"] == "Description override fires"


def test_load_supports_pathlib_path_and_str(tmp_path):
    p = tmp_path / "either.toml"
    p.write_text('"feedback.intro.perfect" = "ok"\n', encoding="utf-8")
    via_path = load_templates_from_toml(p)
    via_str = load_templates_from_toml(str(p))
    assert via_path == via_str


# -----------------------------------------------------------------------------
# Group C — snapshot equivalence with pre-refactor in-source defaults
# -----------------------------------------------------------------------------

def test_description_defaults_byte_identical_to_legacy_snapshot():
    assert DEFAULT_DESCRIPTION_TEMPLATES == LEGACY_DESCRIBE


def test_feedback_defaults_byte_identical_to_legacy_snapshot():
    assert DEFAULT_FEEDBACK_TEMPLATES == LEGACY_FEEDBACK


def test_compact_overrides_byte_identical_to_legacy_snapshot():
    assert _COMPACT_OVERRIDES == LEGACY_COMPACT


# -----------------------------------------------------------------------------
# Group D — _coerce_to_string_dict helper
# -----------------------------------------------------------------------------

def test_coerce_flat_dict_is_passthrough():
    src = {"a": "x", "b": "y"}
    assert _coerce_to_string_dict(src, source="test") == src


def test_coerce_flattens_nested_dict():
    src = {"feedback": {"op": {"primitive_replace": "x"}}}
    assert _coerce_to_string_dict(src, source="test") == {
        "feedback.op.primitive_replace": "x",
    }


def test_coerce_rejects_non_string_leaf():
    src = {"feedback.intro.perfect": 42}
    with pytest.raises(TypeError, match="must be a string"):
        _coerce_to_string_dict(src, source="test")


def test_coerce_error_message_includes_source():
    src = {"some.key": 99}
    with pytest.raises(TypeError, match="test-source"):
        _coerce_to_string_dict(src, source="test-source")

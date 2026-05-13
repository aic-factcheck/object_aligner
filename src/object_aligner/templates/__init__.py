"""Packaged TOML template files for reasoning and feedback rendering.

Marker module so the directory is a real sub-package; ``importlib.resources``
treats the ``.toml`` files alongside it as package data and ships them
through ``uv build`` automatically.

The loader is ``object_aligner._templates._load_packaged_template``; the
placeholder allowlists that gate user overrides live in
``object_aligner.object_aligner._TEMPLATE_PLACEHOLDERS`` (reasoning) and
``object_aligner.feedback._FEEDBACK_PLACEHOLDERS`` (feedback).
"""

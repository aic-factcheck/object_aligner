"""Frozen pre-refactor template snapshots.

This module is the golden reference for the TOML externalization refactor:
the three dicts below were copy-pasted *verbatim* from the pre-refactor
Python sources (``object_aligner.py`` and ``feedback.py``) before any
edits were made. The equivalence tests in ``tests/test_templates.py``
assert that the post-refactor TOML-loaded dicts are byte-identical to
these snapshots, catching any accidental drift during the refactor.

DO NOT modify this file when changing the live templates — if a template
genuinely needs to change, update *both* the live TOML file and this
snapshot in the same commit, on purpose.
"""

# Captured from src/object_aligner/object_aligner.py:75-96
LEGACY_REASONING = {
    "metric.perfect": "The predicted output perfectly matches the gold.",
    "metric.imperfect_intro": "The predicted output scores overall {score_pct}, let us align the predicted output to the gold and analyze the differences:\n",
    "item.match": '{indent}The predicted value "{pred}" exactly matches the gold.\n',
    "item.mismatch": '{indent}The predicted value "{pred}" does not match the gold "{gold}" (score={score_pct}).\n',
    "ref.match": '{indent}The predicted reference "{pred}" matches the gold reference "{gold}" under the inferred id mapping.\n',
    "ref.mismatch": '{indent}The predicted reference "{pred}" does not match the gold reference "{gold}" under the inferred id mapping (score={score_pct}).\n',
    "id.match": "",
    "id.mismatch": "",
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

# Captured from src/object_aligner/feedback.py DEFAULT_FEEDBACK_TEMPLATES
LEGACY_FEEDBACK = {
    "feedback.intro.perfect":
        "The prediction perfectly matches the gold (score 1.00).",
    "feedback.intro.imperfect":
        "The prediction scored {score:.2f} (deficit {deficit:.2f}). "
        "Top {n_shown} of {n_total} fix locations:\n",
    "feedback.op.primitive_replace":
        "{rank}. {path}: expected {gold}, got {pred}. "
        "Fixing this recovers +{score_delta:.3f}.",
    "feedback.op.primitive_replace_reorder":
        "{rank}. inside list {list_path}: replace item {pred} with {gold}. "
        "Fixing this recovers +{score_delta:.3f}.",
    "feedback.op.key_add":
        "{rank}. {path}: missing key \"{key}\" with value {gold}. "
        "Adding it recovers +{score_delta:.3f}.",
    "feedback.op.key_remove":
        "{rank}. {path}: extraneous key \"{key}\" (value {pred}). "
        "Removing it recovers +{score_delta:.3f}.",
    "feedback.op.key_rename_add":
        "{rank}. rename key \"{pred_key}\" -> \"{gold_key}\" at "
        "{gold_path} (value {gold}). Fixing this recovers "
        "+{score_delta:.3f}.",
    "feedback.op.key_rename_remove": "",
    "feedback.op.list_item_add":
        "{rank}. {path}: missing list item {gold}. "
        "Adding it recovers +{score_delta:.3f}.",
    "feedback.op.list_item_remove":
        "{rank}. {path}: extraneous list item {pred}. "
        "Removing it recovers +{score_delta:.3f}.",
    "feedback.op.list_item_missing":
        "{rank}. {list_path}: list is missing item {gold}. "
        "Adding it recovers +{score_delta:.3f}.",
    "feedback.op.list_item_excess":
        "{rank}. {list_path}: list has extraneous item {pred}. "
        "Removing it recovers +{score_delta:.3f}.",
    "feedback.op.ref_fix":
        "{rank}. {path}: wrong reference (expected {gold}, got {pred}). "
        "Fixing this recovers +{score_delta:.3f}.",
    "feedback.op.subtree_replace":
        "{rank}. {path}: subtree differs. "
        "Replacing it recovers +{score_delta:.3f}.",
    "feedback.synthesis.single_dominant":
        "\nFocus on {dominant_kind_human} errors — they account for "
        "{dominant_fraction_pct:.0f}% of the deficit shown.",
    "feedback.synthesis.mixed":
        "\nThe deficit is spread across multiple issue types ({top_kinds}).",
    "feedback.empty":
        "The prediction scored {score:.2f}; no individually significant "
        "fix locations under the current filter.",
    "feedback.validation_error":
        "Prediction failed schema validation at {path}: {message}",
}

# Captured from src/object_aligner/feedback.py _COMPACT_OVERRIDES
LEGACY_COMPACT = {
    "feedback.intro.imperfect":
        "Score {score:.2f}. Top {n_shown}/{n_total} fixes:\n",
    "feedback.op.primitive_replace":
        "{rank}. {path}: {gold} (got {pred}) [+{score_delta:.3f}]",
    "feedback.op.primitive_replace_reorder":
        "{rank}. {list_path}: replace {pred}->{gold} [+{score_delta:.3f}]",
    "feedback.op.key_add":
        "{rank}. {path}: add \"{key}\"={gold} [+{score_delta:.3f}]",
    "feedback.op.key_remove":
        "{rank}. {path}: remove \"{key}\" [+{score_delta:.3f}]",
    "feedback.op.key_rename_add":
        "{rank}. {gold_path}: rename \"{pred_key}\"->\"{gold_key}\""
        " [+{score_delta:.3f}]",
    "feedback.op.list_item_add":
        "{rank}. {path}: add item {gold} [+{score_delta:.3f}]",
    "feedback.op.list_item_remove":
        "{rank}. {path}: remove item {pred} [+{score_delta:.3f}]",
    "feedback.op.list_item_missing":
        "{rank}. {list_path}: add item {gold} [+{score_delta:.3f}]",
    "feedback.op.list_item_excess":
        "{rank}. {list_path}: remove item {pred} [+{score_delta:.3f}]",
    "feedback.op.ref_fix":
        "{rank}. {path}: ref {pred}->{gold} [+{score_delta:.3f}]",
    "feedback.op.subtree_replace":
        "{rank}. {path}: replace subtree [+{score_delta:.3f}]",
}

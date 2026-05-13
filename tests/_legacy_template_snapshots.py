"""Frozen template snapshots.

This module is the byte-for-byte reference for the live default-template
dicts. The equivalence tests in ``tests/test_templates.py`` assert that
the TOML-loaded defaults are byte-identical to these snapshots, catching
any accidental drift.

DO NOT modify this file when changing the live templates — if a template
genuinely needs to change, update *both* the live TOML file and this
snapshot in the same commit, on purpose.
"""

# Captured from src/object_aligner/templates/describe.toml at the time of
# the describe rename — every key is byte-identical to the legacy reasoning
# defaults modulo the "describe." prefix and the validation_error key.
LEGACY_DESCRIBE = {
    "describe.intro.perfect": "The predicted output perfectly matches the gold.",
    "describe.intro.imperfect": "The predicted output scores overall {score_pct}, let us align the predicted output to the gold and analyze the differences:\n",
    "describe.item.match": '{indent}The predicted value "{pred}" exactly matches the gold.\n',
    "describe.item.mismatch": '{indent}The predicted value "{pred}" does not match the gold "{gold}" (score={score_pct}).\n',
    "describe.ref.match": '{indent}The predicted reference "{pred}" matches the gold reference "{gold}" under the inferred id mapping.\n',
    "describe.ref.mismatch": '{indent}The predicted reference "{pred}" does not match the gold reference "{gold}" under the inferred id mapping (score={score_pct}).\n',
    "describe.id.match": "",
    "describe.id.mismatch": "",
    "describe.list.match": "{indent}The predicted list perfectly matches the gold one:\n",
    "describe.list.mismatch": "{indent}The predicted list scores {score_pct}:\n",
    "describe.list.excess": '{indent}The predicted list item "{pred}" is excessive, it was not in the gold.\n',
    "describe.list.missing": '{indent}The predicted output misses the "{gold}" list item from the gold.\n',
    "describe.dict.match": "{indent}The predicted dictionary perfectly matches the gold one:\n",
    "describe.dict.mismatch": "{indent}The predicted dictionary scores {score_pct}:\n",
    "describe.dict.key.match": '{indent}KEY = The predicted key "{pred}" exactly matches the gold.\n',
    "describe.dict.key.mismatch": '{indent}KEY = The predicted key "{pred}" does not match the gold "{gold}" (score={score_pct}).\n',
    "describe.dict.value.prefix": "{indent}VALUE = ",
    "describe.validation_error": 'JSON Schema validation failed for path="{path}". Error message: {message}.',
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

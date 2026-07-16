from object_aligner._templates import load_templates_from_toml

# Importing ``object_aligner.object_aligner`` first defines the MatchItem /
# MatchList / MatchDict dataclasses before any sibling module tries to
# resolve them. The sibling modules below all import those names at their
# own top level; loading them first would force object_aligner.py to be
# entered while attribution/repair/etc. are mid-load, producing a circular
# ImportError. Keep this ordering.
from object_aligner.object_aligner import MatchDict, MatchItem, MatchList, ObjectAligner
from object_aligner._matchtypes import ScoreContext
from object_aligner._metrics import context_metric
from object_aligner.attribution import AttributionEntry, AttributionResult, tree_walk_attribution
from object_aligner.describe import (
    DEFAULT_DESCRIPTION_TEMPLATES,
    DescriptionEntry,
    DescriptionResult,
    render_description,
)
from object_aligner.feedback import (
    DEFAULT_FEEDBACK_TEMPLATES,
    FeedbackEntry,
    FeedbackResult,
    render_feedback,
)
from object_aligner.repair import RepairOp, RepairResult, generate_repairs

__all__ = [
    "AttributionEntry",
    "AttributionResult",
    "DEFAULT_DESCRIPTION_TEMPLATES",
    "DEFAULT_FEEDBACK_TEMPLATES",
    "DescriptionEntry",
    "DescriptionResult",
    "FeedbackEntry",
    "FeedbackResult",
    "MatchDict",
    "MatchItem",
    "MatchList",
    "ObjectAligner",
    "RepairOp",
    "RepairResult",
    "ScoreContext",
    "context_metric",
    "generate_repairs",
    "load_templates_from_toml",
    "render_description",
    "render_feedback",
    "tree_walk_attribution",
]

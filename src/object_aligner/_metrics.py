"""Primitive similarity metrics and small schema/path helpers.

The built-in string and number metric callables, their name-to-callable
registries (:data:`BUILTIN_STRING_METRICS`, :data:`BUILTIN_NUMBER_METRICS`),
the public :func:`context_metric` decorator (marks a custom metric as
context-aware), and the ``path2str`` / ``_schema_allows_type`` helpers used
across the aligner. Pure functions with no project-internal dependencies.
"""
from rapidfuzz.distance import DamerauLevenshtein, Indel, Jaro, JaroWinkler, LCSseq, Levenshtein, OSA


def path2str(p):
    return "/" + "/".join([str(d) for d in p])


def _schema_allows_type(schema_type, name):
    """Return True if ``schema_type`` (a JSON Schema ``type`` value, which
    may be a string or a list of strings) permits ``name``. Used by the
    alignment dispatcher to accept union types such as
    ``type: ["string", "null"]`` alongside plain ``type: "string"``."""
    if schema_type == name:
        return True
    if isinstance(schema_type, list) and name in schema_type:
        return True
    return False


def similarity_exact(a, b):
    return float(a == b)


def similarity_num_inv_diff(a, b):
    diff = abs(a - b)
    score = 1 / (1 + diff)
    return score


def similarity_num_relative(a, b):
    """Scale-invariant similarity: ``1 - min(1, |a-b| / max(|a|, |b|))``.

    Equal values (including ``0`` vs ``0``) score ``1.0``; values whose
    difference is at least as large as the larger magnitude score ``0.0``.
    Unlike ``invdiff`` the result does not depend on the unit of the field:
    ``relative(k*a, k*b) == relative(a, b)`` for any ``k != 0``.
    """
    if a == b:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / max(abs(a), abs(b)))


def similarity_string_jaro(a, b):
    return Jaro.normalized_similarity(a, b)


def similarity_string_jaro_winkler(a, b):
    return JaroWinkler.normalized_similarity(a, b)


def similarity_string_levenshtein(a, b):
    return Levenshtein.normalized_similarity(a, b)


def similarity_string_damerau_levenshtein(a, b):
    return DamerauLevenshtein.normalized_similarity(a, b)


def similarity_string_osa(a, b):
    return OSA.normalized_similarity(a, b)


def similarity_string_indel(a, b):
    return Indel.normalized_similarity(a, b)


def similarity_string_lcsseq(a, b):
    return LCSseq.normalized_similarity(a, b)


BUILTIN_STRING_METRICS = {
    "exact": similarity_exact,
    "jaro": similarity_string_jaro,
    "jaro_winkler": similarity_string_jaro_winkler,
    "levenshtein": similarity_string_levenshtein,
    "damerau_levenshtein": similarity_string_damerau_levenshtein,
    "osa": similarity_string_osa,
    "indel": similarity_string_indel,
    "lcsseq": similarity_string_lcsseq,
}
BUILTIN_NUMBER_METRICS = {
    "exact": similarity_exact,
    "invdiff": similarity_num_inv_diff,
    "relative": similarity_num_relative,
}
SUPPORTED_CUSTOM_METRIC_TYPES = frozenset({"string", "number", "integer"})


def context_metric(fn):
    """Mark a custom metric as context-aware and return it unchanged.

    A plain custom metric is called ``fn(gold, pred)``. Decorating it with
    ``@context_metric`` sets ``fn.wants_context = True``, which tells
    `ObjectAligner` to call it as ``fn(gold, pred, context)`` instead,
    where ``context`` is a
    [`ScoreContext`][object_aligner.ScoreContext] exposing the enclosing
    parent objects and the aligned roots. Use it when a leaf's correctness
    depends on a sibling field or on a value elsewhere in the object.

    The decorator mutates the callable in place and returns the same
    object, so any other attributes it carries (for example the ``.cache``
    attached by the semantic metrics) are preserved. Because it assigns an
    attribute, it works on any callable that accepts attribute assignment —
    plain functions, lambdas, `functools.partial`, and instances of
    callable classes all do. The rare exception is a C-level builtin (e.g.
    ``str.upper``), which cannot hold attributes; wrap it in a plain
    function first.

    Args:
        fn: The custom metric callable, with signature
            ``(gold, pred, context) -> float in [0, 1]``.

    Returns:
        The same callable, now carrying ``wants_context = True``.
    """
    fn.wants_context = True
    return fn

"""Primitive similarity metrics and small schema/path helpers.

The built-in string and number metric callables, their name-to-callable
registries (:data:`BUILTIN_STRING_METRICS`, :data:`BUILTIN_NUMBER_METRICS`),
and the ``path2str`` / ``_schema_allows_type`` helpers used across the aligner.
Pure functions with no project-internal dependencies.
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
}
SUPPORTED_CUSTOM_METRIC_TYPES = frozenset({"string", "number", "integer"})

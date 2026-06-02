"""Hungarian-assignment confidence helpers.

:func:`_hungarian_confidence` derives per-pair and node-level stability scores
from the similarity matrix fed to :func:`scipy.optimize.linear_sum_assignment`;
:func:`_with_confidence` re-stamps the (frozen) match dataclasses. Used by the
list and dict aligners when ``compute_confidence=True``.
"""
import numpy as np


def _with_confidence(match, confidence):
    """Return a copy of ``match`` with its top-level ``confidence`` set.

    Used at Hungarian sites to attach a per-pair confidence to an already
    fully-built recursive match object without re-aligning. All three
    Match dataclasses are frozen, so a plain field write is impossible;
    we lean on ``dataclasses.replace`` which the frozen API officially
    supports.
    """
    from dataclasses import replace
    return replace(match, confidence=float(confidence))


def _hungarian_confidence(
    similarity_matrix,
    row_ind,
    col_ind,
    n,
    m,
    *,
    method="margin",
    temperature=8.0,
):
    """Per-pair and node-level confidence from a Hungarian assignment.

    Reads the similarity matrix that was passed to
    :func:`scipy.optimize.linear_sum_assignment` and returns a stability
    score in ``[0, 1]`` for each chosen pair plus an aggregate scalar.
    Excess/missing pairs (one side is zero-padding, i.e. ``row >= n`` or
    ``col >= m``) score ``1.0`` — there is no ambiguity, the item is
    simply unmatched.

    The ``"margin"`` method computes the symmetric clipped margin against
    the row's and column's second-best entries; the ``"entropy"`` method
    softmaxes each row over its first ``m`` columns and returns
    ``1 - H / log m``.

    Args:
        similarity_matrix: ``(d, d)`` matrix used by the Hungarian site,
            with ``d = max(n, m)``.
        row_ind, col_ind: Output of ``linear_sum_assignment(-similarity_matrix)``.
        n, m: Real (unpadded) gold and pred sizes.
        method: ``"margin"`` or ``"entropy"``.
        temperature: Softmax temperature ``β`` (entropy method only).

    Returns:
        A tuple ``(pair_confidences, node_confidence)`` where
        ``pair_confidences`` is a 1-D ``np.ndarray`` of length
        ``len(row_ind)`` aligned with ``zip(row_ind, col_ind)`` and
        ``node_confidence`` is a Python ``float`` — the mean of the
        confidences over genuinely matched pairs (both sides in range),
        or ``1.0`` if no such pair exists.
    """
    k = len(row_ind)
    pair_conf = np.ones(k, dtype=np.float64)
    matched_confs = []
    if method == "margin":
        for idx in range(k):
            ri = int(row_ind[idx])
            ci = int(col_ind[idx])
            if ri >= n or ci >= m:
                continue
            row = similarity_matrix[ri, :m]
            col = similarity_matrix[:n, ci]
            chosen = float(similarity_matrix[ri, ci])
            if m > 1:
                row_others = np.delete(row, ci)
                m_row = chosen - float(np.max(row_others))
            else:
                m_row = chosen
            if n > 1:
                col_others = np.delete(col, ri)
                m_col = chosen - float(np.max(col_others))
            else:
                m_col = chosen
            c = 0.5 * (min(1.0, max(0.0, m_row)) + min(1.0, max(0.0, m_col)))
            pair_conf[idx] = c
            matched_confs.append(c)
    elif method == "entropy":
        beta = float(temperature)
        log_m = np.log(m) if m > 1 else 1.0
        for idx in range(k):
            ri = int(row_ind[idx])
            ci = int(col_ind[idx])
            if ri >= n or ci >= m:
                continue
            if m == 1:
                pair_conf[idx] = 1.0
                matched_confs.append(1.0)
                continue
            row = similarity_matrix[ri, :m].astype(np.float64)
            shifted = beta * (row - np.max(row))
            exp_row = np.exp(shifted)
            denom = float(np.sum(exp_row))
            if denom <= 0.0:
                pair_conf[idx] = 1.0
                matched_confs.append(1.0)
                continue
            p = exp_row / denom
            with np.errstate(divide="ignore", invalid="ignore"):
                ent_terms = np.where(p > 0.0, -p * np.log(p), 0.0)
            H = float(np.sum(ent_terms))
            c = 1.0 - H / log_m
            c = min(1.0, max(0.0, c))
            pair_conf[idx] = c
            matched_confs.append(c)
    else:
        raise ValueError(f"unknown confidence method: {method!r}")
    node_conf = float(np.mean(matched_confs)) if matched_confs else 1.0
    return pair_conf, node_conf

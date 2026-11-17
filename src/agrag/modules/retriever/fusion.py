"""Rank fusion, diversity, and provenance-preserving dedup for retrieval.

These are pure, dependency-light helpers (numpy only) shared by the standard
hybrid retrieval path (``RetrieverModule``) and the agentic multi-query path
(``MultiQueryRetrieveTool``). Keeping them here — separate from any retriever or
tool — means the same fusion logic is used everywhere and is trivial to unit
test in isolation.

The building blocks:

* ``reciprocal_rank_fusion`` fuses several ranked lists (dense, sparse, one per
  subquery) into a single ranking. It fuses on ranks, not raw scores, so
  incomparable score scales (FAISS L2 distance vs. BM25) never need
  normalization.
* ``mmr`` re-orders a candidate set for diversity (Maximal Marginal Relevance).
* ``dedup_records`` collapses duplicate hits of the same chunk while keeping
  every query/subgoal that surfaced it and which signal (dense/sparse) found it.
"""

import logging
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[Hashable, float]]:
    """Fuse several ranked lists into one ranking via Reciprocal Rank Fusion.

    For each list, an item at 1-indexed rank ``r`` contributes ``weight / (k + r)``
    to its fused score; contributions are summed across lists. Items are returned
    sorted by fused score descending. Fusing on ranks (not raw scores) means the
    dense and sparse score scales never have to be reconciled.

    Parameters
    ----------
    ranked_lists : sequence of sequences
        Each inner sequence is an ordered list of item keys, best-first. Keys must
        be hashable and comparable across lists (e.g. metadata row indices, or
        ``(doc_id, chunk_id)`` tuples).
    k : int
        The RRF constant. Larger values flatten the contribution of top ranks.
    weights : sequence of float, optional
        Per-list weights (same length as ``ranked_lists``). Defaults to 1.0 each.

    Returns
    -------
    list of (key, score)
        Fused ranking, best-first. Ties broken by first appearance order.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"weights length {len(weights)} != number of ranked lists {len(ranked_lists)}")

    scores: Dict[Hashable, float] = {}
    first_seen: Dict[Hashable, int] = {}
    order = 0
    for weight, ranked in zip(weights, ranked_lists):
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            if key not in first_seen:
                first_seen[key] = order
                order += 1

    fused = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return fused


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for a (n, d) matrix, guarding zero norms."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


def mmr(
    query_embedding: np.ndarray,
    candidate_keys: Sequence[Hashable],
    candidate_embeddings: Sequence[Sequence[float]],
    lambda_mult: float = 0.5,
    top_k: Optional[int] = None,
) -> List[Hashable]:
    """Re-order candidates for diversity with Maximal Marginal Relevance.

    MMR greedily selects the candidate maximizing
    ``lambda * sim(query, cand) - (1 - lambda) * max sim(cand, already_selected)``.
    ``lambda_mult = 1`` reduces to pure relevance ordering; ``0`` maximizes
    diversity.

    Parameters
    ----------
    query_embedding : np.ndarray
        The query vector, shape ``(d,)``.
    candidate_keys : sequence
        Keys identifying each candidate (returned in selected order).
    candidate_embeddings : sequence of sequence of float
        Embeddings aligned with ``candidate_keys``, shape ``(n, d)``.
    lambda_mult : float
        Relevance/diversity trade-off in ``[0, 1]``.
    top_k : int, optional
        Number of candidates to select. Defaults to all.

    Returns
    -------
    list
        ``candidate_keys`` re-ordered by MMR selection (length ``min(top_k, n)``).
    """
    n = len(candidate_keys)
    if n == 0:
        return []
    if top_k is None or top_k > n:
        top_k = n

    docs = np.asarray(candidate_embeddings, dtype=float)
    query = np.asarray(query_embedding, dtype=float).reshape(-1)

    q_norm = np.linalg.norm(query) or 1.0
    d_norms = np.linalg.norm(docs, axis=1)
    d_norms[d_norms == 0] = 1.0
    query_sim = (docs @ query) / (d_norms * q_norm)
    doc_sim = _cosine_similarity_matrix(docs)

    selected: List[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: query_sim[i])
        else:
            best = max(
                remaining,
                key=lambda i: lambda_mult * query_sim[i] - (1 - lambda_mult) * max(doc_sim[i][j] for j in selected),
            )
        selected.append(best)
        remaining.remove(best)

    return [candidate_keys[i] for i in selected]


def default_dedup_key(record: Dict[str, Any]) -> Hashable:
    """Identity used to dedup a retrieval record.

    Prefers ``(doc_id, chunk_id)`` when both are present (mirrors
    ``Evidence.dedup_key``), else falls back to the chunk text so identical text
    is not stored twice.
    """
    doc_id = record.get(DOC_ID_KEY)
    chunk_id = record.get(CHUNK_ID_KEY)
    if doc_id is not None and chunk_id is not None:
        return (doc_id, chunk_id)
    return ("text", record.get(DOC_TEXT_KEY, ""))


def dedup_records(
    records: Sequence[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], Hashable] = default_dedup_key,
) -> List[Dict[str, Any]]:
    """Deduplicate retrieval records while preserving all provenance.

    Unlike a plain first-wins dedup, this merges every duplicate hit of the same
    chunk into one record that accumulates:

    * ``retrieval_queries`` — every query/subgoal that surfaced the chunk (fixes
      the previous "first-subquery-only" attribution loss), and
    * ``source_signals`` — which signals found it (e.g. ``"dense"``, ``"sparse"``,
      or a subquery label), taken from each record's ``signal`` field when set.

    The kept record is the first occurrence, augmented with the merged provenance
    and the best (minimum) ``rank`` seen across duplicates. Order of first
    appearance is preserved.

    Parameters
    ----------
    records : sequence of dict
        Retrieval records. Each may carry ``retrieval_query`` and/or ``signal``.
    key_fn : callable
        Maps a record to its dedup identity. Defaults to ``default_dedup_key``.

    Returns
    -------
    list of dict
        Deduplicated records with merged ``retrieval_queries`` / ``source_signals``.
    """
    merged: Dict[Hashable, Dict[str, Any]] = {}
    order: List[Hashable] = []

    for record in records:
        key = key_fn(record)
        query = record.get("retrieval_query")
        signal = record.get("signal")
        if key not in merged:
            kept = dict(record)
            kept["retrieval_queries"] = [query] if query is not None else []
            kept["source_signals"] = [signal] if signal is not None else []
            merged[key] = kept
            order.append(key)
        else:
            kept = merged[key]
            if query is not None and query not in kept["retrieval_queries"]:
                kept["retrieval_queries"].append(query)
            if signal is not None and signal not in kept["source_signals"]:
                kept["source_signals"].append(signal)
            # Keep the strongest position seen for this chunk.
            incoming_rank = record.get("rank")
            kept_rank = kept.get("rank")
            if incoming_rank is not None and (kept_rank is None or incoming_rank < kept_rank):
                kept["rank"] = incoming_rank

    logger.debug("dedup_records: %d records -> %d unique", len(records), len(merged))
    return [merged[key] for key in order]

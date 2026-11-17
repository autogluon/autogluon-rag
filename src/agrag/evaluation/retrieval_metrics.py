"""Retrieval-quality metrics for RAG evaluation (Hit@k, MRR, evidence coverage).

Answer exact-match tells you whether the final text was right; it does not tell
you whether the retriever surfaced the passages needed to get there. On a
multi-hop benchmark that distinction is the whole point: the static-vs-agentic
gap shows up first in retrieval, because the agentic path issues multiple
sub-queries and accumulates evidence across rounds. These metrics score the
retrieved passages against the benchmark's gold supporting facts.

Matching heuristic
------------------
Gold relevance in MultiHop-RAG is a set of ``fact`` snippets (sentences taken
from corpus articles). The pipeline retrieves fixed-size chunks (e.g. 128
tokens), so a gold fact almost never equals a retrieved chunk verbatim. A gold
fact is therefore counted as "retrieved" by a chunk when most of the fact's
content tokens appear in that chunk -- token containment above a threshold
(default 0.6). This is deliberately lenient on chunk boundaries but strict on
content, which is the right trade-off for chunked retrieval.

Sentence-level gold units
-------------------------
MuSiQue's gold "facts" are frequently whole paragraphs, not single sentences.
A 294-token gold paragraph can never place 60% of its content tokens inside a
single ~131-token chunk, so paragraph-level containment structurally under-counts
long facts -- the supporting sentence is retrieved, yet the fact scores as a miss.
To fix that lower bound, each gold fact is split into sentence units and the
fact is counted as retrieved when any one of its sentences clears the
containment bar against a retrieved chunk. A sentence fits in a chunk; a
paragraph does not. A fact with no sentence boundary is treated as a single unit,
so short single-sentence facts behave exactly as before.

The metrics are corpus-agnostic: they take the ranked list of retrieved chunk
texts and the list of gold fact strings, so they work for any benchmark that
provides gold evidence snippets, not just MultiHop-RAG.
"""

import re
from typing import Dict, List

# Minimal stopword set: dropping these keeps containment focused on the content
# words that actually distinguish one fact from another.
_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "by",
    "at",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "s",
    "t",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Sentence boundary: terminal punctuation (. ! ?) followed by whitespace. Kept
# deliberately simple (stdlib regex, no NLP dependency) -- over-splitting is
# harmless here because each unit is matched independently and the fact counts as
# retrieved if any unit clears the bar.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _content_tokens(text: str) -> List[str]:
    """Lowercase, split to alphanumeric tokens, drop stopwords."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _gold_units(fact: str) -> List[str]:
    """Split a gold fact into sentence units for containment matching.

    MuSiQue gold facts are often multi-sentence paragraphs too long to fit the
    60% containment bar inside a single chunk. Splitting into sentences lets a
    fact count as retrieved when any one supporting sentence is present. A fact
    with a single sentence (or none) yields one unit -- identical to the previous
    whole-fact behavior. Units with no content tokens are dropped.
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(fact) if p and p.strip()]
    units = [p for p in parts if _content_tokens(p)]
    if units:
        return units
    # No usable sentence split (e.g. a bare phrase): fall back to the whole fact.
    return [fact] if _content_tokens(fact) else []


def _unit_is_retrieved(unit: str, chunk_text: str, threshold: float) -> bool:
    """True if ``chunk_text`` contains enough of a single gold ``unit``'s tokens.

    Containment = (# of unit content-tokens present in the chunk's token set) /
    (# of unit content-tokens). Uses set membership so repetition does not skew
    the ratio. A unit with no content tokens can never be matched (returns False).
    """
    unit_tokens = _content_tokens(unit)
    if not unit_tokens:
        return False
    chunk_tokens = set(_content_tokens(chunk_text))
    present = sum(1 for tok in set(unit_tokens) if tok in chunk_tokens)
    return (present / len(set(unit_tokens))) >= threshold


def fact_is_retrieved(fact: str, chunk_text: str, threshold: float = 0.6) -> bool:
    """True if ``chunk_text`` contains enough of any of ``fact``'s sentence units.

    The fact is split into sentence units (``_gold_units``); it counts as
    retrieved when any single unit clears the containment threshold against the
    chunk. This keeps chunk-boundary leniency while no longer penalizing long
    multi-sentence gold paragraphs that cannot fit a chunk in full. A fact with no
    content tokens can never be matched (returns False).
    """
    return any(_unit_is_retrieved(unit, chunk_text, threshold) for unit in _gold_units(fact))


def _first_hit_rank(retrieved_texts: List[str], gold_facts: List[str], threshold: float) -> int:
    """1-indexed rank of the first retrieved chunk matching any gold fact, else 0."""
    for rank, chunk in enumerate(retrieved_texts, start=1):
        if any(fact_is_retrieved(fact, chunk, threshold) for fact in gold_facts):
            return rank
    return 0


def _evidence_coverage(retrieved_texts: List[str], gold_facts: List[str], threshold: float) -> float:
    """Fraction of distinct gold facts matched by at least one retrieved chunk.

    This is the multi-hop-specific signal: a single-shot retriever may nail one
    hop (coverage ~= 1/n) while missing the others; a multi-round agentic
    retriever should cover more of the required facts.
    """
    if not gold_facts:
        return 0.0
    covered = sum(
        1 for fact in gold_facts if any(fact_is_retrieved(fact, chunk, threshold) for chunk in retrieved_texts)
    )
    return covered / len(gold_facts)


def _recall_at_k(retrieved_texts: List[str], gold_facts: List[str], k: int, threshold: float) -> float:
    """Fraction of distinct gold facts covered within the top-``k`` retrieved chunks.

    This is ``_evidence_coverage`` restricted to a cutoff: unlike ``hit@k`` (which
    is 1.0 as soon as any single gold fact appears), recall@k rewards covering
    more of the required facts within the first ``k`` results -- the signal that
    separates a multi-hop retriever from a single-shot one at a fixed budget.
    """
    return _evidence_coverage(retrieved_texts[:k], gold_facts, threshold)


def retrieval_metrics_for_query(
    retrieved_texts: List[str],
    gold_facts: List[str],
    k_values=(1, 3, 5, 10),
    threshold: float = 0.6,
) -> Dict[str, float]:
    """Compute per-query retrieval metrics against gold evidence facts.

    Parameters
    ----------
    retrieved_texts : List[str]
        Retrieved chunk texts, best-first (rank 1 = index 0).
    gold_facts : List[str]
        Gold supporting-fact snippets for this query.
    k_values : tuple of int
        Cutoffs for Hit@k.
    threshold : float
        Token-containment threshold for counting a fact as retrieved.

    Returns
    -------
    Dict[str, float]
        ``hit@k`` for each k (1.0 if any gold fact appears in the top k, else 0),
        ``recall@k`` for each k (fraction of distinct gold facts covered within
        the top k), ``mrr`` (reciprocal of the first hit's rank, 0 if none), and
        ``evidence_coverage`` (fraction of distinct gold facts retrieved anywhere
        in the returned list).
    """
    first_rank = _first_hit_rank(retrieved_texts, gold_facts, threshold)
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = 1.0 if (first_rank and first_rank <= k) else 0.0
        metrics[f"recall@{k}"] = _recall_at_k(retrieved_texts, gold_facts, k, threshold)
    metrics["mrr"] = (1.0 / first_rank) if first_rank else 0.0
    metrics["evidence_coverage"] = _evidence_coverage(retrieved_texts, gold_facts, threshold)
    return metrics


def aggregate_retrieval_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean each metric across queries. Returns zeros-free empty dict if no queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: round(sum(q[key] for q in per_query) / n, 4) for key in keys}

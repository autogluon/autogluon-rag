"""Structured evidence for the agentic RAG path.

The standard retriever returns plain text chunks. The agentic path needs
structured evidence so it can support citation, verification, trace export, and
debugging. ``Evidence`` wraps a single retrieved chunk with its provenance and
scoring metadata; ``EvidenceStore`` collects evidence for one query and
deduplicates it by document/chunk identity.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def _max_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Max of two optional (higher-is-better) scores, ignoring ``None``."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _min_opt(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """Min of two optional (lower-is-better) positions, ignoring ``None``."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _min_rank(a: int, b: int) -> int:
    """Best (min) of two ranks. ``-1`` is the "unset" sentinel and loses to any
    real (>= 0) rank; when both are unset the sentinel is preserved."""
    if a is None or a < 0:
        return b if b is not None else a
    if b is None or b < 0:
        return a
    return min(a, b)


@dataclass
class Evidence:
    """A single piece of retrieved evidence.

    Attributes:
    ----------
    evidence_id : str
        Stable reference for citations and trace (assigned by ``EvidenceStore``).
    text : str
        Retrieved chunk content.
    retrieval_query : str
        The query that surfaced this evidence (may differ from the user query
        when subqueries or rewrites are used).
    rank : int
        Position of this chunk within the results for ``retrieval_query``.
        Enables ordering and debugging.
    doc_id : Optional[int]
        Source document identifier. Used for deduplication and provenance.
    chunk_id : Optional[int]
        Source chunk identifier within the document.
    source : Optional[str]
        Document path, URL, or file name. May be ``None`` until ingest-time
        metadata is extended to carry provenance.
    retrieval_score : Optional[float]
        Similarity score from the vector database, if exposed.
    rerank_score : Optional[float]
        Score from the reranker, if a reranker was used.
    rrf_score : Optional[float]
        Fused Reciprocal Rank Fusion score, when the evidence came from fused
        (hybrid / multi-query) retrieval.
    fusion_rank : Optional[int]
        Position of this evidence in the fused ranking (0-indexed).
    retrieval_queries : List[str]
        Every query/subgoal that surfaced this chunk. Preserves full multi-query
        provenance even after dedup (``retrieval_query`` holds the first).
    tool_name : Optional[str]
        Name of the tool that produced this evidence. Supports traceability.
    used_in_answer : bool
        Whether this evidence was cited in the final answer. Supports citation
        precision and verifier output.
    metadata : Dict[str, Any]
        Any additional raw metadata carried from the vector database record.
    """

    text: str
    retrieval_query: str = ""
    rank: int = -1
    doc_id: Optional[int] = None
    chunk_id: Optional[int] = None
    source: Optional[str] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    fusion_rank: Optional[int] = None
    retrieval_queries: List[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    used_in_answer: bool = False
    evidence_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> Any:
        """Return the identity used for deduplication.

        Prefers ``(doc_id, chunk_id)`` when both are present; otherwise falls
        back to the chunk text so that identical text is not stored twice.
        """
        if self.doc_id is not None and self.chunk_id is not None:
            return (self.doc_id, self.chunk_id)
        return ("text", self.text)

    def citation(self) -> str:
        """Return a short human-readable citation reference.

        Uses ``source`` when available, otherwise falls back to
        ``doc_id``/``chunk_id`` (the MVP citation scheme).
        """
        if self.source:
            return self.source
        if self.doc_id is not None and self.chunk_id is not None:
            return f"doc {self.doc_id}, chunk {self.chunk_id}"
        return self.evidence_id or "unknown"

    def merge_provenance(self, other: "Evidence") -> None:
        """Fold another duplicate's query provenance and best score into this item.

        Unions the set of queries that surfaced the chunk into ``retrieval_queries``
        (order of first appearance preserved) so subgoal-coverage signals credit
        every subquery that found it. In addition, keeps the best score seen across
        both occurrences rather than the first: the higher-is-better similarity
        scores (``retrieval_score`` under the normalized inner-product metric,
        ``rerank_score``, ``rrf_score``) take the ``max``, and the lower-is-better
        positions (``rank``, ``fusion_rank``) take the ``min``. A chunk re-retrieved
        by a stronger query then ranks by its strongest evidence rather than by
        whichever query happened to surface it first.
        """
        merged = list(self.retrieval_queries)
        for query in [self.retrieval_query] + [other.retrieval_query] + list(other.retrieval_queries):
            if query and query not in merged:
                merged.append(query)
        self.retrieval_queries = merged
        # Higher-is-better similarity scores: keep the max seen (ignoring None).
        self.retrieval_score = _max_opt(self.retrieval_score, other.retrieval_score)
        self.rerank_score = _max_opt(self.rerank_score, other.rerank_score)
        self.rrf_score = _max_opt(self.rrf_score, other.rrf_score)
        # Lower-is-better positions: keep the best (min) non-negative rank seen.
        self.rank = _min_rank(self.rank, other.rank)
        self.fusion_rank = _min_opt(self.fusion_rank, other.fusion_rank)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the evidence (used for trace export)."""
        return asdict(self)

    @classmethod
    def from_retrieval_record(
        cls,
        record: Dict[str, Any],
        retrieval_query: str,
        rank: int,
        tool_name: Optional[str] = None,
    ) -> "Evidence":
        """Build an ``Evidence`` object from a structured retrieval record.

        The record is expected to carry the keys returned by the retriever when
        ``return_metadata=True`` (``text``, and optionally ``doc_id``,
        ``chunk_id``, ``retrieval_score``, ``rerank_score``, ``source``). Any
        extra keys are preserved under ``metadata``.
        """
        known = {
            "text",
            "rank",
            "doc_id",
            "chunk_id",
            "source",
            "retrieval_score",
            "rerank_score",
            "rrf_score",
            "fusion_rank",
            "retrieval_queries",
        }
        extra = {k: v for k, v in record.items() if k not in known}
        return cls(
            text=record.get("text", ""),
            retrieval_query=retrieval_query,
            rank=rank,
            doc_id=record.get("doc_id"),
            chunk_id=record.get("chunk_id"),
            source=record.get("source"),
            retrieval_score=record.get("retrieval_score"),
            rerank_score=record.get("rerank_score"),
            rrf_score=record.get("rrf_score"),
            fusion_rank=record.get("fusion_rank"),
            retrieval_queries=list(record.get("retrieval_queries", [])),
            tool_name=tool_name,
            metadata=extra,
        )


class EvidenceStore:
    """Holds structured evidence for a single query.

    Deduplicates on ``Evidence.dedup_key()`` so the same chunk retrieved by
    multiple subqueries is stored once. The first occurrence is kept (its
    rank/score are authoritative), but the query provenance of later duplicates
    is merged into it so ``retrieval_queries`` records every subquery that
    surfaced the chunk — subgoal-coverage signals depend on this.
    """

    def __init__(self) -> None:
        self._evidence: List[Evidence] = []
        self._by_key: Dict[Any, Evidence] = {}
        self._counter: int = 0

    def __len__(self) -> int:
        return len(self._evidence)

    def __iter__(self):
        return iter(self._evidence)

    def add(self, evidence: Evidence) -> bool:
        """Add one piece of evidence.

        Returns ``True`` if it was stored, ``False`` if it was a duplicate. On a
        duplicate, the incoming item's query provenance (``retrieval_query`` and
        ``retrieval_queries``) is merged into the stored copy so coverage signals
        still credit every subquery that found the chunk. Assigns a stable
        ``evidence_id`` when the item is first stored.
        """
        key = evidence.dedup_key()
        existing = self._by_key.get(key)
        if existing is not None:
            logger.debug("Merging duplicate evidence provenance for key %s", key)
            existing.merge_provenance(evidence)
            return False
        if evidence.evidence_id is None:
            evidence.evidence_id = f"e{self._counter}"
        self._counter += 1
        self._evidence.append(evidence)
        self._by_key[key] = evidence
        return True

    def add_many(self, evidence_items: List[Evidence]) -> int:
        """Add multiple items; return the count actually stored (non-duplicate)."""
        return sum(1 for item in evidence_items if self.add(item))

    def all(self) -> List[Evidence]:
        """Return all stored evidence in insertion order."""
        return list(self._evidence)

    def get(self, evidence_id: str) -> Optional[Evidence]:
        """Return the evidence with the given id, or ``None``."""
        for item in self._evidence:
            if item.evidence_id == evidence_id:
                return item
        return None

    def texts(self) -> List[str]:
        """Return the raw text of all stored evidence (for prompt building)."""
        return [item.text for item in self._evidence]

    def mark_used(self, evidence_ids: List[str]) -> None:
        """Mark the given evidence ids as used in the answer."""
        wanted = set(evidence_ids)
        for item in self._evidence:
            if item.evidence_id in wanted:
                item.used_in_answer = True

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize all evidence to plain dicts (for trace export)."""
        return [item.to_dict() for item in self._evidence]

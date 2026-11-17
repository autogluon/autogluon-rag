"""Evidence signals for the agentic policy.

The ``DecisionPolicy`` used to decide purely on evidence count. These helpers
derive richer signals from the evidence already collected so the policy (and the
LLM/Strands prompt) can reason about quality, not just quantity:

* Subgoal coverage — the fraction of the planned subqueries that surfaced at
  least one evidence item. Low coverage means the plan was only partly answered.
* Relevance — the best and mean relevance score across evidence, taken from
  the strongest comparable score each item exposes (rerank > rrf). ``None`` when
  no item carries one (e.g. a retriever that returns text only, or a raw vector
  score whose direction depends on the index metric — see ``relevance_score``).
* Contradiction — whether the current verification flagged the evidence as
  ``conflicting_evidence``.

Every field degrades gracefully: coverage is ``1.0`` when there are no subqueries,
and the relevance fields are ``None`` when scores are unavailable, so a caller can
treat "signal absent" as "do not block".
"""

from dataclasses import dataclass
from typing import List, Optional

from agrag.modules.agentic.evidence import Evidence, EvidenceStore
from agrag.modules.agentic.state import AgentState
from agrag.modules.agentic.verifier import VerificationLabel


def relevance_score(ev: Evidence) -> Optional[float]:
    """Return the strongest higher-is-better relevance score an item exposes.

    Priority mirrors the retrieval pipeline's own precedence: a cross-encoder
    ``rerank_score`` is the most trustworthy relevance signal, then the fused
    ``rrf_score``. Both are oriented so that a larger value means more relevant.

    ``retrieval_score`` is deliberately excluded: it is the raw vector-DB score,
    whose direction depends on the index metric (an ``IndexFlatL2`` distance is
    lower-is-better, while an inner-product index is higher-is-better), and the
    evidence layer cannot tell which. Sorting or thresholding it as if higher were
    better would surface the worst chunks first, so it is not used for relevance.
    Returns ``None`` when the item carries neither a rerank nor an rrf score.
    """
    for score in (ev.rerank_score, ev.rrf_score):
        if score is not None:
            return score
    return None


@dataclass
class EvidenceAssessment:
    """Derived, quality-oriented view of the evidence collected so far.

    Attributes:
    ----------
    count : int
        Number of evidence items.
    subgoal_coverage : float
        Fraction of planned subqueries with at least one matching evidence item.
        ``1.0`` when there are no subqueries (nothing to cover).
    best_relevance : Optional[float]
        Highest per-item relevance score (see ``relevance_score``: rerank/rrf
        only, higher-is-better), or ``None`` when no item is scored.
    mean_relevance : Optional[float]
        Mean of the per-item relevance scores, or ``None`` when none are scored.
    has_contradiction : bool
        Whether the current verification labelled the evidence
        ``conflicting_evidence``.
    """

    count: int
    subgoal_coverage: float
    best_relevance: Optional[float]
    mean_relevance: Optional[float]
    has_contradiction: bool

    @property
    def relevance_available(self) -> bool:
        return self.best_relevance is not None


def _subgoal_covered(subquery: str, evidence_store: EvidenceStore) -> bool:
    """Whether any evidence item was surfaced by ``subquery``.

    Matches against both ``retrieval_query`` (the representative query) and
    ``retrieval_queries`` (full multi-query provenance kept through dedup), so an
    item found by several subqueries counts for each of them.
    """
    for ev in evidence_store:
        if ev.retrieval_query == subquery or subquery in ev.retrieval_queries:
            return True
    return False


def assess_evidence(state: AgentState, evidence_store: EvidenceStore) -> EvidenceAssessment:
    """Compute the evidence signals for the current state.

    Reuses the scores and provenance already carried on each ``Evidence`` object;
    it never re-retrieves or calls a model.
    """
    scores: List[float] = [s for s in (relevance_score(ev) for ev in evidence_store) if s is not None]

    subqueries = state.subqueries or []
    if subqueries:
        covered = sum(1 for sq in subqueries if _subgoal_covered(sq, evidence_store))
        coverage = covered / len(subqueries)
    else:
        coverage = 1.0

    label = (state.verification or {}).get("label")
    has_contradiction = label == VerificationLabel.CONFLICTING_EVIDENCE.value

    return EvidenceAssessment(
        count=len(evidence_store),
        subgoal_coverage=coverage,
        best_relevance=max(scores) if scores else None,
        mean_relevance=(sum(scores) / len(scores)) if scores else None,
        has_contradiction=has_contradiction,
    )

"""Per-query runtime state for the agentic RAG path.

``AgentState`` holds everything the agent loop needs for a single query: the
original and current query text, the retrieval plan and subqueries, the running
log of actions/observations, evidence ids, the draft answer, the verification
result, the iteration count, and the final status. It is intentionally
short-lived — one instance per ``generate_response`` call — and must not be used
as long-term or cross-query memory.
"""

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Reason(str, Enum):
    """Machine-readable reason a hop or a final answer is unresolved/unverified.

    Attached to ``hop_answers`` entries (per hop) and to ``AgentState.unverified_reason``
    (the run's final disposition) so the trace/benchmark can distinguish why a hop
    grounded to UNKNOWN or why a returned answer is ``ANSWERED_UNVERIFIED`` -- rather
    than collapsing every failure into one opaque state. Reasons are informational
    for now; they do not yet steer differentiated recovery routing.
    """

    NO_EVIDENCE = "no_evidence"  # retrieval returned nothing usable for the hop
    RETRIEVAL_FAILED = "retrieval_failed"  # the retrieval tool raised/failed
    AMBIGUOUS = "ambiguous"  # multiple conflicting candidates (reserved)
    DEPENDENCY_UNRESOLVED = "dependency_unresolved"  # a prerequisite hop was UNKNOWN / a #n ref was invalid
    BUDGET_EXHAUSTED = "budget_exhausted"  # a time/call budget tripped before support was reached
    UNSUPPORTED = "unsupported"  # a draft was produced but verification rejected it


class AgentStatus(str, Enum):
    """Terminal or in-progress status of an agentic run."""

    IN_PROGRESS = "in_progress"
    ANSWERED = "answered"
    # A best-effort answer returned when the run could not obtain an accepted
    # verification (never-refuse mode: iteration/recovery was exhausted and the
    # last verification was rejected or absent). Distinguished from ``ANSWERED``
    # so the true verified-accept rate is observable in the trace/benchmark. Both
    # count as "answered" for the answered-vs-abstained distinction -- use a
    # membership check, not ``== ANSWERED``.
    ANSWERED_UNVERIFIED = "answered_unverified"
    ABSTAINED = "abstained"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class ActionRecord:
    """One (action, observation) step in the agent loop.

    Attributes:
    ----------
    iteration : int
        The loop iteration in which this step occurred.
    action_type : str
        The action requested by the policy (e.g. "retrieve", "draft_answer").
    tool_name : Optional[str]
        The tool invoked for this action, if any.
    args : Dict[str, Any]
        Arguments passed to the tool/action.
    observation_summary : str
        A short, serializable summary of the observation (not the full payload).
    evidence_added : int
        How many new (non-duplicate) evidence items this step contributed.
    query : str
        The working query in effect when this step ran. Lets the policy tell
        whether retrieval has already happened for the current query (so a
        rewrite triggers fresh retrieval rather than dead-ending).
    """

    iteration: int
    action_type: str
    tool_name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    observation_summary: str = ""
    evidence_added: int = 0
    query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """Runtime state for answering one query.

    Attributes:
    ----------
    original_query : str
        The user's query, never mutated after construction.
    current_query : str
        The query currently being acted on (may be rewritten during the loop).
    plan : List[str]
        Retrieval queries produced by the planner.
    subqueries : List[str]
        Subqueries generated for multi-query retrieval.
    history : List[ActionRecord]
        Ordered log of actions and observation summaries.
    evidence_ids : List[str]
        Ids of evidence collected so far (mirrors the EvidenceStore contents).
    draft_answer : Optional[str]
        The most recent draft answer, if one has been synthesized.
    verification : Optional[Dict[str, Any]]
        The most recent verification result (label + details) for the current
        query. Reset to None on a rewrite so a stale rejection does not keep
        forcing rewrites. Use ``last_verification`` for debugging/trace instead.
    last_verification : Optional[Dict[str, Any]]
        The most recent verification result of the whole run, preserved across
        rewrites. Purely informational (trace/debugging); the policy never reads
        it. Lets the trace show what the final draft's verification was even when
        the run ended just after a rewrite cleared ``verification``.
    compressed_context : Optional[str]
        Context produced by the context-compression tool, if any. Reset to None on
        a rewrite (via ``set_current_query``); compression is chosen just before
        drafting for the current query, so it is regenerated per query rather than
        per retrieval.
    hop_answers : List[Dict[str, str]]
        Ordered chain of resolved intermediate answers, one per hop, populated only
        by the sequential-hop executor (``use_iterative_planner``). Each entry is
        ``{"subquery", "hop_query", "answer"}`` where ``hop_query`` is the subquery
        after prior hops' answers were substituted in. Empty on the parallel path.
    iteration : int
        Current loop iteration (0-based).
    recovery_attempts : int
        Number of whole-plan re-decomposition recovery passes taken by the
        sequential-hop executor (``use_replan_recovery``). Tracked separately from
        ``iteration`` because a recovery pass re-runs the hop loop and must not be
        blocked by (nor consume) the ``max_iterations`` budget, which also gates the
        default bounded loop. 0 on every other path.
    hop_recovery_attempts : int
        Number of local (in-plan) hop-recovery passes taken by the sequential-hop
        executor (``use_hop_recovery``) -- the extra hypothesized-seed retrieval
        issued when a hop grounds to UNKNOWN. Tracked separately from ``iteration``
        so local recovery (execution-failure repair) does not draw down the budget
        that gates verify-retry (evidence/support repair); the two fix different
        failure modes and must not compete for one counter. Local recovery is
        naturally bounded to one pass per UNKNOWN hop. 0 on every other path.
    retrieval_failures : int
        Number of retrieval-tool exceptions swallowed during this run (e.g. a hop
        or rewrite retrieval that threw). The first such failure per run is logged
        at WARNING so a systematic retriever error is distinguishable from the
        benign "retrieved but found nothing" case; subsequent failures stay at
        DEBUG to avoid log spam.
    llm_calls : int
        Number of generator (LLM) calls made during this run, counted for the
        ``max_llm_calls`` budget. Incremented by the executor around each
        synthesis/verification/rewrite/hop generation.
    retrieval_calls : int
        Number of retrieval-tool invocations made during this run, counted for the
        ``max_retrieval_calls`` budget (both successful and failed attempts).
    unverified_reason : Optional[str]
        When the run returns a best-effort answer tagged ``ANSWERED_UNVERIFIED``,
        the ``Reason`` value explaining why (e.g. ``unsupported``,
        ``budget_exhausted``, ``dependency_unresolved``). ``None`` for a verified
        ``ANSWERED`` run or an abstention.
    status : AgentStatus
        Current status of the run.
    """

    original_query: str
    current_query: str = ""
    plan: List[str] = field(default_factory=list)
    subqueries: List[str] = field(default_factory=list)
    history: List[ActionRecord] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    draft_answer: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    last_verification: Optional[Dict[str, Any]] = None
    compressed_context: Optional[str] = None
    hop_answers: List[Dict[str, str]] = field(default_factory=list)
    iteration: int = 0
    recovery_attempts: int = 0
    hop_recovery_attempts: int = 0
    retrieval_failures: int = 0
    llm_calls: int = 0
    retrieval_calls: int = 0
    unverified_reason: Optional[str] = None
    status: AgentStatus = AgentStatus.IN_PROGRESS

    def __post_init__(self) -> None:
        if not self.current_query:
            self.current_query = self.original_query

    def record_action(
        self,
        action_type: str,
        tool_name: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        observation_summary: str = "",
        evidence_added: int = 0,
    ) -> ActionRecord:
        """Append an (action, observation) step to the history and return it."""
        record = ActionRecord(
            iteration=self.iteration,
            action_type=action_type,
            tool_name=tool_name,
            args=args or {},
            observation_summary=observation_summary,
            evidence_added=evidence_added,
            query=self.current_query,
        )
        self.history.append(record)
        return record

    def add_evidence_ids(self, evidence_ids: List[str]) -> None:
        """Track newly stored evidence ids (dedup against existing)."""
        for eid in evidence_ids:
            if eid not in self.evidence_ids:
                self.evidence_ids.append(eid)

    def set_verification(self, verification: Optional[Dict[str, Any]]) -> None:
        """Record a verification result.

        Sets both the current-query ``verification`` (which the policy reads and
        which is cleared on a rewrite) and ``last_verification`` (which persists
        across rewrites for the trace).
        """
        self.verification = verification
        if verification is not None:
            self.last_verification = verification

    def set_current_query(self, query: str) -> None:
        """Update the working query (e.g. after a rewrite).

        Resets the per-query working artifacts that pertain to the previous
        query: the compressed context, the draft answer, and its verification.
        Otherwise a stale "unsupported" verdict would keep forcing rewrites even
        after fresh evidence has been retrieved for the new query.
        """
        self.current_query = query
        self.compressed_context = None
        self.draft_answer = None
        self.verification = None

    def retrieved_for_current_query(self) -> bool:
        """Whether a retrieval step has already run for the current query.

        A rewrite changes ``current_query``; because each ``ActionRecord`` stores
        the query in effect when it ran, this returns False again after a rewrite,
        which lets the policy retrieve afresh with the rewritten query.
        """
        retrieval_actions = ("retrieve", "multi_retrieve")
        return any(
            record.action_type in retrieval_actions and record.query == self.current_query
            for record in self.history
        )

    def draft_attempts(self) -> int:
        """Number of draft-answer steps recorded so far."""
        return sum(1 for record in self.history if record.action_type == "draft_answer")

    @property
    def tool_call_count(self) -> int:
        """Number of steps in history that invoked a tool."""
        return sum(1 for record in self.history if record.tool_name is not None)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_ids)

    @property
    def is_terminal(self) -> bool:
        return self.status != AgentStatus.IN_PROGRESS

    def finish(self, status: AgentStatus) -> None:
        """Mark the run terminal with the given status."""
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full state (used for trace export)."""
        return {
            "original_query": self.original_query,
            "current_query": self.current_query,
            "plan": list(self.plan),
            "subqueries": list(self.subqueries),
            "history": [record.to_dict() for record in self.history],
            "evidence_ids": list(self.evidence_ids),
            "draft_answer": self.draft_answer,
            "verification": self.verification,
            "last_verification": self.last_verification,
            "compressed_context": self.compressed_context,
            "hop_answers": [dict(h) for h in self.hop_answers],
            "iteration": self.iteration,
            "recovery_attempts": self.recovery_attempts,
            "hop_recovery_attempts": self.hop_recovery_attempts,
            "retrieval_failures": self.retrieval_failures,
            "llm_calls": self.llm_calls,
            "retrieval_calls": self.retrieval_calls,
            "unverified_reason": self.unverified_reason,
            "status": self.status.value,
        }

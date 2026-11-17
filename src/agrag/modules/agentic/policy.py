"""Decision policy for the agentic RAG loop.

The ``DecisionPolicy`` decides the next action: retrieve more evidence, rewrite
the query, compress context, draft an answer, or abstain. It is driven by the
current ``AgentState`` and the collected evidence.

Unlike a fixed pipeline, the policy reacts to observations: it retrieves again
after a query rewrite, and it changes course when a draft fails verification
(rewriting to gather better evidence rather than re-drafting the same answer).

Three modes are supported, in precedence order Strands > LLM > rule-based:

* Rule-based (default): a deterministic priority ordering picks the next
  action from the set of currently-legal actions.
* LLM-backed (opt-in via ``use_llm`` + a ``generator_module``): the LLM
  chooses among the legal actions the deterministic guardrails allow. The LLM
  returns only a validated action enum; the arguments for each action are
  assembled deterministically here, so the model can never emit a query string
  or tool argument that breaks the executor.
* Strands-backed (opt-in via a ``strands_backend``): a Strands agent driving
  Bedrock Sonnet 4.6 chooses among the legal actions, constrained to that exact
  set by a JSON-schema enum (Pydantic structured output). It too returns only an
  action value; ``_build_args`` still assembles arguments deterministically.

In every mode the guardrails (never draft before retrieval, forced abstain when
out of evidence and budget, the bounded loop) stay structural — the LLM only
reorders among options that are already safe, and any malformed/illegal choice
(or backend failure) falls back to the deterministic pick, ``legal_actions[0]``.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.signals import assess_evidence, relevance_score
from agrag.modules.agentic.state import AgentState
from agrag.modules.agentic.verifier import VerificationLabel

logger = logging.getLogger(LOGGER_NAME)


class ActionType(str, Enum):
    """Actions the policy can choose."""

    RETRIEVE = "retrieve"
    MULTI_RETRIEVE = "multi_retrieve"
    REWRITE_QUERY = "rewrite_query"
    COMPRESS_CONTEXT = "compress_context"
    DRAFT_ANSWER = "draft_answer"
    ABSTAIN = "abstain"


@dataclass
class Action:
    """A chosen action plus its arguments."""

    type: ActionType
    args: Dict[str, Any] = field(default_factory=dict)


# Short, model-facing description of what each action does, so the LLM can choose
# meaningfully among the legal options rather than by name alone.
_ACTION_DESCRIPTIONS = {
    ActionType.RETRIEVE: "run a single retrieval for the current query",
    ActionType.MULTI_RETRIEVE: "run retrieval over all planned subqueries and merge results",
    ActionType.REWRITE_QUERY: "reformulate the query to gather better/more evidence, then retrieve again",
    ActionType.COMPRESS_CONTEXT: "compress the oversized evidence into a smaller grounded context before drafting",
    ActionType.DRAFT_ANSWER: "draft the final answer now from the evidence already collected",
    ActionType.ABSTAIN: "give up and abstain because the question cannot be answered from the corpus",
}

# Prompt for the LLM policy. The model only picks among the legal action values
# it is given and must emit strict JSON so the choice can be validated without a
# schema library.
_POLICY_INSTRUCTION = (
    "You are the decision policy for an agentic RAG loop. Given the current "
    "state, choose the single best next action from the ALLOWED ACTIONS list. "
    "Choose only from that list. Judge sufficiency from the evidence itself, not "
    "just its count: prefer drafting when the subgoals are well covered, the "
    "evidence is relevant, and there is no contradiction; rewrite when coverage "
    "or relevance is weak (or the evidence conflicts) and better evidence is "
    "plausibly retrievable; abstain only when the question is not answerable from "
    "the corpus.\n\n"
    'Return ONLY a JSON object of the form {"action": "<one of the allowed '
    'action values>"} with no markdown, no code fences, and no explanation.\n\n'
)


class DecisionPolicy:
    """Next-action policy for the agentic loop.

    Attributes:
    ----------
    min_evidence_count : int
        Minimum evidence required before drafting an answer.
    use_query_rewrite : bool
        Whether the policy may choose to rewrite the query (after weak retrieval
        or a failed verification).
    use_context_compression : bool
        Whether the policy may choose to compress oversized context before
        drafting.
    max_context_tokens : int
        Approximate context-token budget; when the collected evidence exceeds it
        and compression is enabled, the policy compresses before drafting.
    min_subgoal_coverage : float
        Minimum fraction of planned subqueries that must have supporting evidence
        before drafting is the preferred action. When coverage is below this and
        a rewrite is still in budget, the policy prefers a rewrite to fill the gap.
        Only bites when the plan has subqueries; drafting is never removed, so the
        loop still terminates.
    min_relevance : Optional[float]
        Minimum best-evidence relevance score below which the policy prefers a
        rewrite (when in budget) over drafting. ``None`` disables the relevance
        gate. Scores are compared only when evidence actually carries them, so a
        score-less retriever never blocks drafting.
    max_rewrites : int
        Upper bound on query rewrites within a single run. This is the primary
        knob for how much the agent may iterate to surface new evidence: because
        re-retrieving the same query is deduplicated to nothing, only a rewrite (a
        new query) can add evidence. Defaults to 2.
    allow_abstention : bool
        Opt-in switch for refusing to answer. When False (the default) ABSTAIN is
        never offered as one of several fork options — so the LLM/Strands policy
        can never choose to give up over iterating — and survives only as a
        forced sole terminal (no rewrite budget and no new evidence), which the
        executor converts into a best-effort answer. When True the original
        behavior is restored: ABSTAIN is offered as a fallback alongside
        rewrite/draft and as a terminal that leads to the canned abstention.
    max_iterations : Optional[int]
        The executor's loop budget. When set, the policy will not choose to
        rewrite the query on the final iteration: a rewrite is only useful if
        there is budget left to re-retrieve and re-draft afterwards. Rewriting
        with no remaining budget guarantees an abstention and wastes an LLM call.
        When None, the policy assumes budget is always available (legacy
        behavior).
    generator_module : Optional[GeneratorModule]
        The shared generator used when ``use_llm`` is enabled. When ``None`` the
        policy is always rule-based regardless of ``use_llm``.
    use_llm : bool
        Whether the LLM chooses among the legal actions. Effective only when a
        ``generator_module`` is also provided, and only consulted at genuine
        forks (iterations where more than one action is legal). Single-legal-
        action iterations are decided structurally without an LLM call.
    strands_backend : Optional[StrandsReasoner]
        A Strands/Bedrock reasoning helper. When provided it takes precedence
        over the raw-LLM path at forks: it chooses one action from the legal set
        via an enum-constrained schema. ``None`` disables the Strands path. Like
        the LLM path it is consulted only at genuine forks.
    """

    # A rewrite must be followed by at least a re-retrieve and a re-draft to be
    # worth doing, so it needs this many iterations left after the current one.
    _REWRITE_FOLLOWUP_ITERATIONS = 2

    def __init__(
        self,
        min_evidence_count: int = 2,
        use_query_rewrite: bool = True,
        use_context_compression: bool = False,
        max_context_tokens: int = 6000,
        min_subgoal_coverage: float = 0.5,
        min_relevance: Optional[float] = None,
        max_rewrites: int = 2,
        max_iterations: int = None,
        generator_module=None,
        use_llm: bool = False,
        strands_backend=None,
        allow_abstention: bool = False,
    ):
        self.min_evidence_count = min_evidence_count
        self.use_query_rewrite = use_query_rewrite
        self.use_context_compression = use_context_compression
        self.max_context_tokens = max_context_tokens
        self.min_subgoal_coverage = min_subgoal_coverage
        self.min_relevance = min_relevance
        self.max_rewrites = max_rewrites
        self.max_iterations = max_iterations
        self.generator_module = generator_module
        self.use_llm = use_llm
        self.strands_backend = strands_backend
        self.allow_abstention = allow_abstention

    @property
    def _llm_enabled(self) -> bool:
        return self.use_llm and self.generator_module is not None

    @property
    def _fork_decider_enabled(self) -> bool:
        """Whether some model backend can choose among legal actions at a fork."""
        return self.strands_backend is not None or self._llm_enabled

    def _rewrite_count(self, state: AgentState) -> int:
        return sum(1 for r in state.history if r.action_type == ActionType.REWRITE_QUERY.value)

    def _has_retrieved(self, state: AgentState) -> bool:
        """Whether any retrieval has run this run (regardless of query)."""
        return any(
            r.action_type in (ActionType.RETRIEVE.value, ActionType.MULTI_RETRIEVE.value) for r in state.history
        )

    def _rewrite_has_budget(self, state: AgentState) -> bool:
        """Whether enough loop iterations remain to act on a rewrite.

        A rewrite consumes the current iteration and only pays off if the
        rewritten query can be retrieved and re-drafted before the loop ends.
        When ``max_iterations`` is unknown, assume budget is available.
        """
        if self.max_iterations is None:
            return True
        remaining_after_rewrite = self.max_iterations - (state.iteration + 1)
        return remaining_after_rewrite >= self._REWRITE_FOLLOWUP_ITERATIONS

    def _can_rewrite(self, state: AgentState) -> bool:
        return (
            self.use_query_rewrite
            and self._rewrite_count(state) < self.max_rewrites
            and self._rewrite_has_budget(state)
        )

    def _can_rewrite_posthoc(self, state: AgentState) -> bool:
        """Whether another rewrite is allowed for a post-loop verify-retry.

        Unlike ``_can_rewrite``, this drops the ``_rewrite_has_budget`` iteration
        gate: it is used by the sequential-hop path's verify-retry, which runs after
        the hop loop has finished, so the number of main-loop iterations left is
        irrelevant. The retry is still bounded by ``max_rewrites`` (each pass
        records a rewrite action), so it always terminates.
        """
        return self.use_query_rewrite and self._rewrite_count(state) < self.max_rewrites

    def _last_draft_failed(self, state: AgentState) -> bool:
        """True when the most recent draft was verified and rejected.

        ``set_current_query`` clears ``verification`` on a rewrite, so a stale
        rejection from a previous query does not keep forcing rewrites.
        """
        if state.draft_answer is None or state.verification is None:
            return False
        return not self.accept_verification(state.verification)

    def _new_evidence_since_last_draft(self, state: AgentState) -> bool:
        """Whether any retrieval added evidence after the most recent draft.

        A failed draft only justifies re-drafting if the evidence has actually
        changed since — otherwise re-drafting would reproduce the same rejected
        answer. Scans history back to the last ``draft_answer`` step and reports
        whether a retrieval recorded ``evidence_added > 0`` after it. Returns False
        when no draft has been recorded yet.
        """
        retrieve_actions = (ActionType.RETRIEVE.value, ActionType.MULTI_RETRIEVE.value)
        added_after_draft = 0
        seen_draft = False
        for record in state.history:
            if record.action_type == ActionType.DRAFT_ANSWER.value:
                seen_draft = True
                added_after_draft = 0  # reset: only count evidence after the latest draft
            elif seen_draft and record.action_type in retrieve_actions:
                added_after_draft += record.evidence_added
        return seen_draft and added_after_draft > 0

    def _compression_attempted(self, state: AgentState) -> bool:
        """Whether compression has already been tried for the current query.

        Prevents re-compressing every iteration when the tool returns nothing
        usable (which leaves ``compressed_context`` None).
        """
        return any(
            r.action_type == ActionType.COMPRESS_CONTEXT.value and r.query == state.current_query
            for r in state.history
        )

    def _context_too_large(self, evidence_store: EvidenceStore) -> bool:
        """Approximate whether collected evidence exceeds the token budget."""
        approx_tokens = sum(len(ev.text.split()) for ev in evidence_store)
        return approx_tokens > self.max_context_tokens

    def _evidence_is_weak(self, state: AgentState, evidence_store: EvidenceStore) -> bool:
        """Whether the collected evidence looks too weak to draft confidently.

        Weak means any of: incomplete subgoal coverage (plan subqueries left
        unsupported), best relevance below ``min_relevance`` when a score is
        available, or a detected contradiction. Signals that are unavailable
        (no subqueries, no scores) never count as weak, so a score-less retriever
        or a single-query plan behaves exactly as before.

        Note: in the normal loop a contradiction (``conflicting_evidence``) also
        sets ``is_supported=False`` on a draft, which routes through the
        draft-failed guardrail before this branch-5 check is reached. The
        contradiction term is kept as a defensive guard so this predicate stays
        correct if it is ever consulted from a state without a failed draft.
        """
        assessment = assess_evidence(state, evidence_store)
        if state.subqueries and assessment.subgoal_coverage < self.min_subgoal_coverage:
            return True
        if (
            self.min_relevance is not None
            and assessment.best_relevance is not None
            and assessment.best_relevance < self.min_relevance
        ):
            return True
        return assessment.has_contradiction

    def next_action(self, state: AgentState, evidence_store: EvidenceStore) -> Action:
        """Choose the next action given the current state and evidence.

        The legal actions for the current state are computed by ``_legal_actions``
        (the guardrails). When exactly one action is legal it is taken directly.
        When more than one is legal — the genuine fork — the LLM chooses among
        them if enabled, otherwise the deterministic priority order (first legal)
        is used. The chosen action's arguments are always assembled by
        ``_build_args`` so the executor's ``action.args`` contract is preserved
        regardless of mode.
        """
        legal = self._legal_actions(state, evidence_store)

        if len(legal) == 1:
            chosen = legal[0]
        elif self._fork_decider_enabled:
            chosen = self._llm_choose(state, evidence_store, legal)
        else:
            # Deterministic priority: first legal action (preserves prior behavior,
            # e.g. compress before draft).
            chosen = legal[0]

        return Action(chosen, self._build_args(chosen, state, evidence_store))

    def _legal_actions(self, state: AgentState, evidence_store: EvidenceStore) -> List[ActionType]:
        """Return the actions that are legal in the current state (guardrails).

        The guardrails mirror the original rule cascade so the LLM can never pick
        an illegal or wasteful action; within those bounds, most branches expose a
        genuine fork so the LLM policy has real agency. The first element of
        every returned list is the deterministic (rule-based) choice, so a
        rule-based policy (``use_llm`` off) reproduces the original behavior
        exactly.

        1. If nothing has been retrieved for the current query, retrieval is
           forced (multi on the first retrieval when the plan has subqueries, else
           single). No fork: drafting before any evidence is never allowed, and
           choosing single over multi at the start has little upside for a
           guaranteed extra LLM call.
        2. If evidence is below the minimum: rewrite to gather better evidence when
           a rewrite is allowed and in budget; otherwise abstain (give up as
           unanswerable). When ``allow_abstention`` is set, abstain is additionally
           offered as a discretionary fallback alongside the rewrite. Drafting
           below the evidence floor is never legal (the verifier would
           short-circuit to insufficient_evidence).
        3. If the latest draft failed verification: when new evidence has arrived
           since that draft, re-drafting can now succeed, so draft or rewrite
           (draft first, the default). When no new evidence has arrived, re-drafting
           would only reproduce the rejected answer, so rewrite (when allowed);
           otherwise abstain. When ``allow_abstention`` is set, abstain is also
           offered as a discretionary fallback in each of these forks (as before).
        4. If context compression is enabled and the evidence exceeds the token
           budget: compress oversized context first or draft now.
        5. Otherwise (enough evidence, nothing failing): draft the answer or, when
           a rewrite is still in budget, rewrite to seek stronger evidence before
           answering. Draft is the default, but when the evidence looks weak
           (incomplete subgoal coverage, low relevance, or a contradiction) the
           order flips so a rewrite is preferred. Drafting is never removed, so
           the loop still terminates.
        """
        if not state.retrieved_for_current_query():
            # Multi-query retrieval applies to the initial plan only; a rewritten
            # query is a single reformulation. Retrieval is forced (no fork).
            if not self._has_retrieved(state) and len(state.plan) > 1:
                return [ActionType.MULTI_RETRIEVE]
            return [ActionType.RETRIEVE]

        if len(evidence_store) < self.min_evidence_count:
            # Below the evidence floor. Rewrite to gather more; abstain only when a
            # rewrite is no longer possible. In never-refuse mode ABSTAIN is not
            # offered as a discretionary alternative to the rewrite — it appears
            # only as the forced sole terminal the executor turns into a best
            # effort. In refuse mode it is offered as the fallback, as before.
            if self._can_rewrite(state):
                if self.allow_abstention:
                    return [ActionType.REWRITE_QUERY, ActionType.ABSTAIN]
                return [ActionType.REWRITE_QUERY]
            return [ActionType.ABSTAIN]

        if self._last_draft_failed(state):
            # A single failed verification is not terminal: if new evidence has
            # arrived since that draft, re-drafting can now succeed, so offer it
            # again (first, so it is the default). Without new evidence, re-drafting
            # would only reproduce the rejected answer, so fall back to a rewrite to
            # gather better evidence. ABSTAIN keeps the loop terminating: in
            # never-refuse mode it is only the forced sole terminal (no new evidence
            # and no rewrite budget), never a discretionary option; in refuse mode
            # it is offered as the fallback alongside draft/rewrite, as before.
            if self._new_evidence_since_last_draft(state):
                if self._can_rewrite(state):
                    if self.allow_abstention:
                        return [ActionType.DRAFT_ANSWER, ActionType.REWRITE_QUERY, ActionType.ABSTAIN]
                    return [ActionType.DRAFT_ANSWER, ActionType.REWRITE_QUERY]
                if self.allow_abstention:
                    return [ActionType.DRAFT_ANSWER, ActionType.ABSTAIN]
                return [ActionType.DRAFT_ANSWER]
            if self._can_rewrite(state):
                if self.allow_abstention:
                    return [ActionType.REWRITE_QUERY, ActionType.ABSTAIN]
                return [ActionType.REWRITE_QUERY]
            return [ActionType.ABSTAIN]

        if (
            self.use_context_compression
            and state.compressed_context is None
            and not self._compression_attempted(state)
            and self._context_too_large(evidence_store)
        ):
            # Genuine choice point: compress oversized context first, or draft now.
            return [ActionType.COMPRESS_CONTEXT, ActionType.DRAFT_ANSWER]

        # Enough evidence and nothing failing: draft by default, but the LLM may
        # instead rewrite to seek stronger evidence when a rewrite is in budget.
        # When the evidence looks weak, prefer the rewrite by putting it first.
        if self._can_rewrite(state):
            if self._evidence_is_weak(state, evidence_store):
                return [ActionType.REWRITE_QUERY, ActionType.DRAFT_ANSWER]
            return [ActionType.DRAFT_ANSWER, ActionType.REWRITE_QUERY]
        return [ActionType.DRAFT_ANSWER]

    def _build_args(self, action_type: ActionType, state: AgentState, evidence_store: EvidenceStore) -> Dict[str, Any]:
        """Assemble the arguments for a chosen action deterministically.

        This is the single source of ``Action.args``; the LLM never supplies
        arguments, so the executor's reads (``args["query"]`` / ``["queries"]`` /
        ``["texts"]``) can never be broken by model output.
        """
        if action_type == ActionType.RETRIEVE:
            return {"query": state.current_query}
        if action_type == ActionType.MULTI_RETRIEVE:
            return {"queries": state.plan}
        if action_type == ActionType.REWRITE_QUERY:
            return {"query": state.current_query}
        if action_type == ActionType.COMPRESS_CONTEXT:
            return {"query": state.current_query, "texts": evidence_store.texts()}
        # DRAFT_ANSWER and ABSTAIN take no arguments.
        return {}

    def _llm_choose(
        self, state: AgentState, evidence_store: EvidenceStore, legal_actions: List[ActionType]
    ) -> ActionType:
        """Let a model backend choose among the legal actions; validate and guard.

        Precedence is Strands > raw-LLM. Both are given the same state summary and
        the legal-action set; both return only an action value. Falls back to
        ``legal_actions[0]`` (the deterministic priority choice) on any backend
        failure or if a backend picks an action outside the legal set.
        """
        prompt = self._build_choice_prompt(state, evidence_store, legal_actions)

        if self.strands_backend is not None:
            chosen = self._strands_choose(prompt, legal_actions)
            if chosen is not None:
                return chosen
            # Strands failed; fall through to the raw-LLM path if configured.

        if self._llm_enabled:
            try:
                raw = self.generator_module.generate_response(prompt)
            except Exception as exc:  # generator/backend failure must not break the loop
                logger.debug("LLM policy generation failed (%s); using deterministic choice", exc)
                return legal_actions[0]
            return self._parse_action(raw, legal_actions)

        return legal_actions[0]

    def _strands_choose(self, prompt: str, legal_actions: List[ActionType]) -> Optional[ActionType]:
        """Choose an action via the Strands backend's enum-constrained schema.

        The backend is restricted to exactly the legal action values, so its
        return is either one of them or ``None`` (on failure). Returns the mapped
        ``ActionType`` or ``None`` so the caller can fall back.
        """
        legal_values = [a.value for a in legal_actions]
        try:
            value = self.strands_backend.choose_action(prompt, legal_values)
        except Exception as exc:  # backend must never break the loop
            logger.debug("Strands policy call failed (%s); falling back", exc)
            return None
        if value is None:
            return None
        try:
            chosen = ActionType(value)
        except ValueError:
            logger.debug("Strands policy returned unmappable action %r", value)
            return None
        if chosen not in legal_actions:  # defensive; enum already constrains this
            return None
        return chosen

    def _build_choice_prompt(
        self, state: AgentState, evidence_store: EvidenceStore, legal_actions: List[ActionType]
    ) -> str:
        """Build a compact state summary + described allowed-action list for the LLM.

        The summary surfaces evidence quality (subgoal coverage, best/mean
        relevance, contradiction) and short snippets of the strongest evidence, so
        the model reasons over the actual context rather than a bare count.
        """
        allowed_lines = "\n".join(f"- {a.value}: {_ACTION_DESCRIPTIONS.get(a, '')}" for a in legal_actions)
        drafted = state.draft_answer is not None
        last_label = (state.verification or {}).get("label") if drafted else None
        assessment = assess_evidence(state, evidence_store)
        summary = (
            f"ORIGINAL QUESTION: {state.original_query}\n"
            f"CURRENT QUERY: {state.current_query}\n"
            f"EVIDENCE COUNT: {assessment.count}\n"
            f"SUBGOAL COVERAGE: {assessment.subgoal_coverage:.2f}"
            + (f" ({len(state.subqueries)} subqueries)\n" if state.subqueries else " (no subqueries)\n")
            + (f"BEST RELEVANCE: {assessment.best_relevance:.4f}\n" if assessment.best_relevance is not None else "")
            + (f"MEAN RELEVANCE: {assessment.mean_relevance:.4f}\n" if assessment.mean_relevance is not None else "")
            + ("CONTRADICTION DETECTED: yes\n" if assessment.has_contradiction else "")
            + f"ITERATION: {state.iteration}"
            + (f" of {self.max_iterations}" if self.max_iterations else "")
            + "\n"
            f"DRAFT ATTEMPTED: {drafted}\n"
            + (f"LAST VERIFICATION: {last_label}\n" if last_label else "")
            + self._evidence_digest(evidence_store)
            + f"ALLOWED ACTIONS:\n{allowed_lines}\n"
        )
        return f"{_POLICY_INSTRUCTION}{summary}"

    @staticmethod
    def _evidence_digest(evidence_store: EvidenceStore, top_n: int = 3, snippet_chars: int = 200) -> str:
        """Render short snippets of the strongest evidence for the policy prompt.

        Orders by each item's higher-is-better relevance score (rerank/rrf; scored
        items first), preserving the retriever's own rank order among items without
        such a score rather than sorting a raw vector distance. Includes the leading
        ``snippet_chars`` of the top ``top_n`` items with their citation and score.
        Returns an empty string when there is no evidence.
        """
        items = list(evidence_store)
        if not items:
            return ""
        # Stable sort keeps original (retriever rank) order for ties, so scoreless
        # items stay in retrieval order instead of being reordered by a distance.
        ranked = sorted(
            items,
            key=lambda ev: (relevance_score(ev) is not None, relevance_score(ev) or 0.0),
            reverse=True,
        )
        lines = ["TOP EVIDENCE:"]
        for ev in ranked[:top_n]:
            snippet = " ".join(ev.text.split())[:snippet_chars]
            score = relevance_score(ev)
            score_str = f"{score:.4f}" if score is not None else "n/a"
            lines.append(f"- [{ev.citation()}] (score={score_str}) {snippet}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Best-effort parse of a JSON object embedded in model output.

        Slices from the first ``{`` to the last ``}`` so leading/trailing prose or
        code fences do not defeat parsing. Returns the parsed dict, or ``None`` if
        no JSON object can be recovered.
        """
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None

    def _parse_action(self, raw: str, legal_actions: List[ActionType]) -> ActionType:
        """Validate the model output into a legal ``ActionType``.

        Returns the chosen action only when it parses to a known ``ActionType``
        that is in ``legal_actions``; otherwise returns ``legal_actions[0]``.
        """
        obj = self._extract_json(raw)
        if obj is None:
            logger.debug("LLM policy output was not valid JSON: %r", raw)
            return legal_actions[0]
        value = obj.get("action")
        if not isinstance(value, str):
            logger.debug("LLM policy JSON missing an 'action' string: %r", obj)
            return legal_actions[0]
        try:
            chosen = ActionType(value.strip().lower())
        except ValueError:
            logger.debug("LLM policy returned unknown action %r", value)
            return legal_actions[0]
        if chosen not in legal_actions:
            logger.debug("LLM policy chose illegal action %s; using deterministic choice", chosen.value)
            return legal_actions[0]
        return chosen

    def accept_verification(self, verification: Dict[str, Any]) -> bool:
        """Whether a verification result is good enough to return the answer.

        Only ``supported`` is accepted. ``partially_supported`` now joins
        ``conflicting_evidence``, ``unsupported`` and ``insufficient_evidence`` in
        being rejected, so it routes through rewrite/recovery + re-verification
        rather than being returned as verified. The executor's "unverified" path
        sets ``is_supported`` True (with no label), so verification-off runs still
        accept.
        """
        verification = verification or {}
        if verification.get("is_supported", False):
            return True
        return verification.get("label") == VerificationLabel.SUPPORTED.value

"""Answer verification for the agentic RAG path.

The ``AnswerVerifier`` checks whether a draft answer is supported by the retrieved
evidence and returns a structured label. It reuses the single configured
``GeneratorModule`` as an LLM judge, with robust parsing of the model's label
output.
"""

import logging
import re
from enum import Enum
from typing import Any, Dict

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.hop_utils import (
    hop_answer_anchors,
    is_non_answer,
    rank_evidence,
    render_hop_chain_with_gaps,
)

logger = logging.getLogger(LOGGER_NAME)


class VerificationLabel(str, Enum):
    """Structured verification labels."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Labels good enough to return the answer as verified. Only ``supported``
# qualifies; ``partially_supported`` (a minor detail ungrounded), along with
# ``conflicting_evidence``, ``unsupported`` and ``insufficient_evidence``, are
# rejected so they route through recovery + re-verification rather than being
# accepted. A run that still cannot reach ``supported`` returns its best-effort
# draft tagged ``ANSWERED_UNVERIFIED`` (never-refuse behavior preserved). This
# mirrors ``AgenticPolicy.accept_verification``; keeping ``is_supported`` consistent
# with the accept set matters because the executor's max-iterations fallback gates
# the best-draft return on ``is_supported`` directly.
_ACCEPTABLE_LABELS = frozenset({VerificationLabel.SUPPORTED})

_VERIFY_INSTRUCTION = (
    "You are a careful, skeptical verifier. Given a QUESTION, a draft ANSWER, the "
    "RESOLVED REASONING CHAIN the answer was built from (when one is provided), and "
    "the EVIDENCE, judge whether the evidence actually grounds the answer. This is "
    "often a multi-hop question. When a reasoning chain is provided, verify it LINK "
    "BY LINK: check that each intermediate fact in the chain is supported by the "
    "evidence and that the links compose to the final answer. Accept when every link "
    "is grounded even if the supporting facts are SCATTERED across different evidence "
    "items -- you do not need a single passage that states the whole answer. Reject "
    "if any link is contradicted by the evidence or absent from it. The final answer "
    "must be entailed by the evidence, not by outside/parametric knowledge.\n"
    "Be strict about SPECIFICITY: when the question asks for a specific date, year, "
    "number, name, or place, the answer must give exactly that value. A less precise "
    "or near-miss value (a decade instead of the year, or an adjacent/off-by-one "
    "value) is NOT supported. Do not accept an answer just because it is plausible. "
    "If the draft ANSWER does not actually answer the question -- it is empty, a "
    "refusal, or states the question cannot be answered -- reply 'unsupported'.\n\n"
    "Reply with exactly one of these labels and nothing else:\n"
    "- supported: the evidence, taken together (across items if needed), entails the "
    "specific answer.\n"
    "- partially_supported: the evidence establishes the specific answer entity but "
    "leaves a minor detail ungrounded. Use ONLY when the core answer is genuinely "
    "backed by the evidence; NEVER use it to hedge a guess or a near-miss value.\n"
    "- unsupported: the evidence does not establish the specific answer (including "
    "when the answer relies on knowledge not present in the evidence, is a near-miss "
    "on a specific value, or is a non-answer/refusal).\n"
    "- conflicting_evidence: the evidence directly contradicts the answer.\n"
    "- insufficient_evidence: there is essentially no relevant evidence for the "
    "question.\n\n"
)

class AnswerVerifier:
    """LLM-backed verifier returning a structured label.

    Attributes:
    ----------
    generator_module : GeneratorModule
        The generator used as the verification judge.
    min_evidence_count : int
        If fewer than this many evidence items are available, the verifier short
        circuits to ``insufficient_evidence`` without calling the model. Defaults to
        1 so the guard only trips on genuinely zero evidence: a single passage that
        fully grounds the answer must reach the LLM judge rather than being
        auto-rejected (which was a false-negative source, especially in never-refuse
        mode where the draft is then returned anyway).
    max_context_tokens : int
        Approximate cap on the size of the evidence block placed in the verifier
        prompt (approximated by whitespace token count, matching the
        synthesizer). Prevents concatenating unbounded evidence into a single
        prompt, which would overflow the model's context window. At least the
        first evidence item is always included.
    """

    def __init__(self, generator_module, min_evidence_count: int = 1, max_context_tokens: int = 6000):
        self.generator_module = generator_module
        self.min_evidence_count = min_evidence_count
        self.max_context_tokens = max_context_tokens

    @staticmethod
    def _parse_label(text: str) -> VerificationLabel:
        """Map raw model output to a label; default to unsupported if unclear.

        A reply with no recognizable label token is treated as a rejection: a
        parse miss is not evidence of support, so defaulting to ``unsupported``
        keeps the verifier honest. Callers that must always return an answer use
        the executor's ``always_answer`` mode, which returns the draft regardless
        of this label rather than relaxing what the verifier reports.
        """
        lowered = (text or "").strip().lower()
        # Match the earliest-occurring label token, not the first in some fixed
        # order: ``"unsupported"`` contains ``"supported"``, so a plain
        # "longest-first, substring-anywhere" scan flips a reply like
        # "...not unsupported -- it is clearly supported" to unsupported. Pick the
        # label whose value appears first in the reply; on a tie at the same index
        # prefer the longer value (so "unsupported" wins over "supported" at that
        # position, and "partially_supported" over "supported").
        best_label = None
        best_pos = None
        best_len = -1
        for label in VerificationLabel:
            pos = lowered.find(label.value)
            if pos == -1:
                continue
            if best_pos is None or pos < best_pos or (pos == best_pos and len(label.value) > best_len):
                best_label, best_pos, best_len = label, pos, len(label.value)
        if best_label is not None:
            return best_label
        logger.debug("Verifier could not parse label from %r; defaulting to unsupported", text)
        return VerificationLabel.UNSUPPORTED

    @staticmethod
    def _anchors(draft_answer: str, hop_answers) -> list:
        """Answer strings the supporting evidence should mention.

        The final draft answer plus every resolved (non-UNKNOWN) hop answer. Used
        to pull the evidence that actually grounds the answer into the bounded
        verifier window even when it was retrieved late (e.g. by a recovery pass)
        and would otherwise be truncated out under an insertion-order cut.
        """
        anchors = []
        cleaned = (draft_answer or "").strip()
        # Require a couple of chars so a stray token does not match half the corpus.
        if len(cleaned) >= 2 and cleaned.upper() != "UNKNOWN":
            anchors.append(cleaned.lower())
        # Resolved (non-UNKNOWN) hop answers, via the shared helper so anchor
        # selection stays consistent with the rendered hop chain.
        anchors.extend(hop_answer_anchors(hop_answers))
        return anchors

    def _build_evidence_block(self, evidence_store: EvidenceStore, anchors=None) -> str:
        """Join evidence text into a bounded block for the verifier prompt.

        Evidence is selected by relevance, not insertion order: items whose text
        contains an answer anchor (the draft answer or a resolved hop answer) come
        first, then items with a higher reranker/retrieval score, then original
        order as a stable tiebreak. This keeps the chunk that actually grounds
        the answer inside the ``max_context_tokens`` window even after a
        recovery pass has appended dozens of later, lower-relevance items, which
        an insertion-order cut would otherwise truncate out. Always keeps at least
        the top-ranked item.
        """
        # Shared ranking with the synthesizer (``rank_evidence``) so the answer is
        # drafted from and judged against the same best evidence.
        ordered = rank_evidence(evidence_store, anchors=anchors)

        lines = []
        used = 0
        for ev in ordered:
            approx_tokens = len(ev.text.split())
            if lines and used + approx_tokens > self.max_context_tokens:
                break
            lines.append(f"- {ev.text}")
            used += approx_tokens
        return "\n".join(lines)

    @staticmethod
    def _hop_chain_block(hop_answers) -> str:
        """Render the multi-hop chain for the verifier prompt, including gaps.

        The verifier sees the same links the answer was composed from, but unlike
        the synthesizer it also sees hops the executor could not resolve, marked
        ``(NOT RESOLVED from evidence)``. Hiding those (the old behavior) let the
        verifier judge the answer against a curated, self-consistent chain and
        rubber-stamp a final value the model had filled in from parametric
        knowledge -- the dominant false positive. Returns an empty string when
        there are no hops (e.g. the default loop path), leaving the prompt unchanged.
        """
        return render_hop_chain_with_gaps(hop_answers, "REASONING CHAIN:", trailer="\n\n")

    # Digit runs of 3+ (years like "1943", large counts) that a grounded answer
    # must be able to point at in the evidence. Short runs ("6th", "2 cities") are
    # excluded: they are too common to gate on without false rejections.
    _SALIENT_NUM_RE = re.compile(r"\d{3,}")

    @classmethod
    def _has_ungrounded_number(cls, draft_answer: str, evidence_store: EvidenceStore) -> bool:
        """True when the answer asserts a salient number absent from all evidence.

        A deterministic grounding gate that fires only on high-signal numeric
        claims (a year/large number in the answer that appears nowhere in the
        retrieved evidence). Catches the ``"1943"``-style hallucinations a same-model
        LLM judge waves through, while staying silent on non-numeric answers (where
        it would risk false rejections). Commas are stripped so ``"1,000"`` matches
        ``"1000"``.
        """
        nums = set(cls._SALIENT_NUM_RE.findall((draft_answer or "").replace(",", "")))
        if not nums:
            return False
        haystack = " ".join((ev.text or "") for ev in evidence_store).replace(",", "")
        return any(n not in haystack for n in nums)

    def verify(
        self, query: str, draft_answer: str, evidence_store: EvidenceStore, hop_answers=None
    ) -> Dict[str, Any]:
        """Return a verification result dict with a structured label.

        ``hop_answers`` (optional) is the executor's resolved multi-hop chain
        (``state.hop_answers``); when present it is shown to the verifier so it can
        check the chain link by link and its answers anchor evidence selection.

        Returns:
        -------
        Dict[str, Any]
            ``{"label": <str>, "is_supported": <bool>, "evidence_count": <int>}``.
            ``is_supported`` is True only for the ``supported`` label (see
            ``_ACCEPTABLE_LABELS``).
        """
        evidence_count = len(evidence_store)
        # A non-answer / refusal is never supported, regardless of evidence.
        if is_non_answer(draft_answer):
            logger.debug("Verifier: draft is a non-answer/refusal; labeling unsupported")
            return self._result(VerificationLabel.UNSUPPORTED, evidence_count)
        if evidence_count < self.min_evidence_count:
            label = VerificationLabel.INSUFFICIENT_EVIDENCE
            return self._result(label, evidence_count)

        anchors = self._anchors(draft_answer, hop_answers)
        evidence_block = self._build_evidence_block(evidence_store, anchors=anchors)
        chain_block = self._hop_chain_block(hop_answers)
        prompt = (
            f"{_VERIFY_INSTRUCTION}QUESTION: {query}\n\n"
            f"ANSWER: {draft_answer}\n\n"
            f"{chain_block}"
            f"EVIDENCE:\n{evidence_block}"
        )
        raw = self.generator_module.generate_response(prompt)
        label = self._parse_label(raw)
        # Deterministic grounding gate: never accept an answer whose salient number
        # (a year/large count) is absent from every evidence item, regardless of what
        # the LLM judge said. This only ever downgrades an accept -> unsupported, so
        # it cannot manufacture a false positive.
        if label in _ACCEPTABLE_LABELS and self._has_ungrounded_number(draft_answer, evidence_store):
            logger.debug("Verifier: answer asserts a number absent from evidence; downgrading to unsupported")
            label = VerificationLabel.UNSUPPORTED
        logger.debug("Verifier label: %s", label.value)
        return self._result(label, evidence_count)

    @staticmethod
    def _result(label: VerificationLabel, evidence_count: int) -> Dict[str, Any]:
        return {
            "label": label.value,
            "is_supported": label in _ACCEPTABLE_LABELS,
            "evidence_count": evidence_count,
        }

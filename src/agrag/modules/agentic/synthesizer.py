"""Answer synthesis for the agentic RAG path.

The ``AnswerSynthesizer`` builds a grounded prompt from the collected evidence and
calls the existing ``GeneratorModule``. It reuses ``format_query`` so prompt
formatting stays consistent with the standard RAG path (per-model templates).
"""

import logging
import re
from typing import List, Optional, Tuple

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import Evidence, EvidenceStore
from agrag.modules.agentic.hop_utils import (
    hop_answer_anchors,
    rank_evidence,
    render_hop_chain_with_gaps,
)
from agrag.modules.generator.utils import format_query

logger = logging.getLogger(LOGGER_NAME)

# Cues that mark a draft as a reasoning narration rather than a terse answer span
# (lowercased, matched against the draft's stripped start). Used to decide when the
# short-answer extraction pass is worth an extra generator call.
_REASONING_LEAD_CUES = (
    "let me",
    "let's",
    "to answer",
    "the question asks",
    "the question is asking",
    "first,",
    "step 1",
    "step 1:",
    "1.",
    "based on the",
    "here is",
    "here's",
)

# One-shot instruction that distills a verbose/reasoning draft into the shortest
# answer span. No evidence is passed -- it only compresses the draft the synthesizer
# already produced, so it is cheap and cannot introduce ungrounded content beyond
# what the draft asserted.
_EXTRACT_INSTRUCTION = (
    "Given a QUESTION and a DRAFT answer, output ONLY the final answer as the "
    "shortest possible span -- a name, number, date, or short phrase. Do not "
    "restate the question, show reasoning, or add any explanation or punctuation "
    "beyond the answer itself. Keep any quantity, unit, or qualifier that is part "
    "of the answer (e.g. '75% of the world's teak', not 'teak'; 'regrouped and "
    "defeated the Portuguese', not 'defeated'). Copy the answer span from the DRAFT "
    "verbatim -- do NOT add information the draft does not state and do NOT make it "
    "more specific than the draft (e.g. if the draft says 'the 1970s', answer "
    "'1970s', never a specific year like '1978').\n\n"
    "QUESTION: {query}\n\nDRAFT: {draft}\n\nFinal answer:"
)

# Salient qualifier tokens whose loss would silently change the answer's meaning:
# any run of digits (covers numbers, years, dates like "1947"), a percentage, or a
# money amount. Used as a content-preservation guard on short-answer extraction --
# if the draft carried such tokens and the extracted span dropped *all* of them, the
# extraction over-compressed (e.g. "75% of the world's teak" -> "teak") and the
# draft is kept instead. Deliberately narrow so terse answers are never affected.
_SALIENT_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*\s*%?|[$€£]\s*\d")


def _salient_tokens(text: str) -> set:
    """Set of normalized salient qualifier tokens present in ``text``."""
    return {m.group(0).replace(" ", "") for m in _SALIENT_TOKEN_RE.finditer(text or "")}


class AnswerSynthesizer:
    """Generates a grounded answer from evidence using the configured generator.

    Attributes:
    ----------
    generator_module : GeneratorModule
        The generator used to produce the answer.
    max_context_tokens : int
        Approximate cap on context size. Evidence is included in order until the
        budget is reached (approximated by whitespace token count).
    query_prefix : str
        Optional instruction prepended to the query before formatting, mirroring
        the standard RAG path (``AutoGluonRAG.generate_response``) so answer
        formatting is consistent across standard and agentic modes. Applied only
        to answer synthesis, not to verification or query-rewriting.
    extract_answer : bool
        When True, a verbose/reasoning-style draft is distilled into the shortest
        answer span via one extra generator call (see ``extract_short_answer``).
        Only fires when the draft looks verbose, so latency impact is bounded.
        Default False -> the raw draft is returned unchanged (prior behavior).
    """

    def __init__(
        self,
        generator_module,
        max_context_tokens: int = 6000,
        query_prefix: str = "",
        extract_answer: bool = False,
    ):
        self.generator_module = generator_module
        self.max_context_tokens = max_context_tokens
        self.query_prefix = query_prefix or ""
        self.extract_answer = extract_answer

    def _select_evidence(self, evidence_store: EvidenceStore, anchors=None) -> List[Evidence]:
        """Select evidence up to the (approximate) context-token budget.

        Evidence is ranked by (anchor hit, relevance) via the shared
        ``rank_evidence`` -- the same ordering the verifier uses -- so the answer is
        drafted from, and later judged against, the same best evidence rather than
        raw insertion order. ``anchors`` are the resolved hop answers (the draft
        does not exist yet at synthesis time).
        """
        ranked = rank_evidence(evidence_store, anchors=anchors)
        selected: List[Evidence] = []
        used = 0
        for ev in ranked:
            approx_tokens = len(ev.text.split())
            if selected and used + approx_tokens > self.max_context_tokens:
                break
            selected.append(ev)
            used += approx_tokens
        return selected

    def build_context(self, evidence_store: EvidenceStore, anchors=None) -> Tuple[List[str], List[str]]:
        """Return (context_texts, evidence_ids) for the selected evidence."""
        selected = self._select_evidence(evidence_store, anchors=anchors)
        texts = [f"[{ev.citation()}] {ev.text}" for ev in selected]
        ids = [ev.evidence_id for ev in selected if ev.evidence_id is not None]
        return texts, ids

    @staticmethod
    def _hop_answer_block(hop_answers: Optional[List[dict]]) -> str:
        """Render the multi-hop chain as a compact prompt block, gaps included.

        Returns an empty string when there are no hops. Each entry is the
        intermediate ``{subquery -> answer}`` the executor resolved, so the final
        synthesis can compose the answer from the chain instead of re-deriving it
        from raw evidence. Unlike the resolved-only rendering, a hop the executor
        could not ground is shown explicitly as ``(NOT RESOLVED from evidence)``
        (via the shared ``render_hop_chain_with_gaps`` helper the verifier also
        uses). Previously these gaps were hidden, leaving the synthesizer a curated,
        self-consistent chain that invited it to fill a missing bridge fact from
        parametric knowledge (the confident-wrong-answer failure mode on deep
        multi-hop questions). Showing the gap lets ``_FINAL_ANSWER_INSTRUCTION``'s
        no-fabrication clause actually bind.
        """
        return render_hop_chain_with_gaps(hop_answers, "Resolved intermediate facts:")

    def _looks_verbose(self, draft: str) -> bool:
        """Heuristic: is this draft a narration rather than a terse answer span?

        True when the draft is long/multi-sentence or opens with a reasoning cue.
        Used to gate the extra extraction call so short answers pay no penalty.
        """
        text = (draft or "").strip()
        if not text:
            return False
        low = text.lower()
        if any(low.startswith(cue) for cue in _REASONING_LEAD_CUES):
            return True
        # Multi-sentence and non-trivially long -> likely an explanation.
        return text.count(".") >= 2 and len(text) > 120

    def extract_short_answer(self, query: str, draft: str) -> str:
        """Distill a verbose draft into the shortest answer span (one LLM call).

        Falls back to the original draft on empty output or any generation error,
        so extraction can never blank out an answer.
        """
        prompt = _EXTRACT_INSTRUCTION.format(query=query, draft=draft)
        try:
            extracted = self.generator_module.generate_response(prompt)
        except Exception as exc:  # extraction must never break answering
            logger.debug("Short-answer extraction failed (%s); keeping draft", exc)
            return draft
        extracted = (extracted or "").strip()
        if not extracted:
            return draft
        # Content-preservation guard: if the draft carried salient qualifier tokens
        # (numbers, dates, percentages, money) and the extracted span dropped all
        # of them, the extraction over-compressed and would silently change the
        # answer's meaning -- keep the draft instead. Only fires when every salient
        # token was lost, so terse answers and partial keeps are unaffected.
        #
        # But only apply this to terse drafts. A verbose reasoning narration
        # (multi-sentence "Step 1... Step 2..." output) is riddled with
        # incidental salient tokens -- ordinal list markers ("1.", "2."), dates and
        # tallies cited mid-reasoning ("95 wins to 91", "founded 1902") -- none of
        # which are the answer, while the correct short span usually carries no
        # digit at all. There the guard almost always mis-fires and returns the whole
        # narration, which tanks F1/strict-EM/ROUGE (measured: 6 of 7 verbose
        # answers kept their full narration this way). For those drafts we trust the
        # extraction, whose instruction already preserves answer-bearing qualifiers.
        # Terse drafts (e.g. "75% of the world's teak") still get the guard, so the
        # genuine over-compression case it was built for is unaffected.
        if not self._looks_verbose(draft):
            draft_salient = _salient_tokens(draft)
            if draft_salient and not (_salient_tokens(extracted) & draft_salient):
                logger.debug(
                    "Short-answer extraction dropped all salient tokens %s; keeping draft",
                    draft_salient,
                )
                return draft
        return extracted

    def generate(
        self,
        query: str,
        evidence_store: EvidenceStore,
        compressed_context: str = None,
        hop_answers: Optional[List[dict]] = None,
        answer_instruction: Optional[str] = None,
        drop_query_prefix: bool = False,
        temperature: Optional[float] = None,
    ) -> Tuple[str, List[str]]:
        """Generate an answer grounded in the evidence.

        When ``compressed_context`` is provided (produced by the context
        compression tool), it is used as the context instead of the raw evidence
        chunks. All collected evidence ids are still returned as "used" so
        citation/traceability reflect what fed the compressed summary.

        When ``hop_answers`` is provided (the resolved multi-hop chain), it is
        prepended to the query so the model composes the final answer from the
        already-derived intermediate facts rather than re-reasoning from scratch.

        When ``answer_instruction`` is provided, it is prepended to the query
        (before ``query_prefix``) as a per-call directive — used by intermediate
        hop answering to force strict grounding in the retrieved context (the base
        ``format_query`` template gives no such instruction, so hop answers can
        otherwise leak to the model's parametric knowledge). Default ``None`` ->
        unchanged behavior.

        When ``drop_query_prefix`` is True, the configured ``query_prefix`` is not
        prepended for this call. The shared prefix pushes single-word answers (good
        for single-hop exact match) but pulls a multi-hop final answer to shed
        answer-bearing qualifiers; dropping it on the final synthesis lets
        ``answer_instruction`` (the full-answer directive) govern uncontested.
        Default False -> unchanged behavior (prefix applied).

        Returns the answer text and the list of evidence ids that were placed in
        the prompt (the caller can mark these as used).
        """
        if compressed_context:
            context_texts = [compressed_context]
            evidence_ids = [ev.evidence_id for ev in evidence_store if ev.evidence_id is not None]
        else:
            # Anchor evidence selection on the resolved hop answers so the drafter
            # sees the same relevance/anchor-ranked window the verifier judges.
            context_texts, evidence_ids = self.build_context(
                evidence_store, anchors=hop_answer_anchors(hop_answers)
            )
        # Compose from the resolved hop chain when available.
        hop_block = self._hop_answer_block(hop_answers)
        if hop_block:
            query = f"{hop_block}\n\n{query}"
        # Prepend the configured query prefix (e.g. answer-format instructions),
        # matching the standard path's behavior so both modes format answers alike.
        # Skipped when ``drop_query_prefix`` is set (final multi-hop synthesis), so
        # the single-word push does not fight the full-answer directive.
        if self.query_prefix and not drop_query_prefix:
            query = f"{self.query_prefix}\n{query}"
        # Prepend a per-call answer instruction (e.g. strict-grounding directive
        # for hop answering) closest to the query so it takes precedence.
        if answer_instruction:
            query = f"{answer_instruction}\n{query}"
        formatted = format_query(
            model_name=self.generator_module.model_name,
            query=query,
            context=context_texts,
        )
        # Only thread ``temperature`` through when set, so generators/fakes whose
        # ``generate_response`` takes just ``query`` keep working (single-draft path).
        if temperature is None:
            answer = self.generator_module.generate_response(formatted)
        else:
            answer = self.generator_module.generate_response(formatted, temperature=temperature)
        # Distill verbose/reasoning drafts into a short answer span when enabled.
        if self.extract_answer and self._looks_verbose(answer):
            answer = self.extract_short_answer(query, answer)
        logger.debug("Synthesized answer using %d evidence items", len(evidence_ids))
        return answer, evidence_ids

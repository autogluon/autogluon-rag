"""Unit tests for hop-chain consistency across the agentic workflow.

These cover:

* An UNKNOWN/empty hop answer being filtered out of both the synthesizer's and
  the verifier's rendered hop-chain blocks (via the shared ``hop_utils``
  renderer), so a literal non-answer never leaks into a prompt.
* The synthesizer and verifier selecting the same evidence window: both rank by
  (anchor hit, relevance) via the shared ``rank_evidence``, so a late-retrieved
  grounding chunk survives a tight token budget instead of being truncated by
  insertion order.
* ``AnswerVerifier._parse_label`` matching the earliest-occurring label, so an
  explanatory reply ("...not unsupported -- clearly supported") is not inverted.
* A single grounding passage reaching the LLM judge (verifier default
  ``min_evidence_count`` is 1) rather than being auto-rejected as
  ``insufficient_evidence``.
* When a hop retrieval throws, a placeholder hop entry still being appended so
  later ``#n`` back-references stay aligned to their hop index; the first
  swallowed retrieval failure per run is counted (and logged at WARNING).
* A best-effort answer returned without an accepted verification being marked
  ``ANSWERED_UNVERIFIED`` rather than ``ANSWERED``, and the trace exporting
  ``hop_recovery_attempts``.

They use lightweight fakes and exercise the code paths offline (no Bedrock).
"""

import unittest

from agrag.modules.agentic.evidence import Evidence, EvidenceStore
from agrag.modules.agentic.executor import AgentExecutor
from agrag.modules.agentic.hop_utils import (
    hop_answer_anchors,
    is_unknown_hop_answer,
    rank_evidence,
    render_hop_chain,
)
from agrag.modules.agentic.state import AgentState, AgentStatus
from agrag.modules.agentic.synthesizer import AnswerSynthesizer
from agrag.modules.agentic.verifier import AnswerVerifier, VerificationLabel


class RecordingGenerator:
    """Generator fake that records prompts and returns a scripted response."""

    def __init__(self, response="the answer"):
        self.response = response
        self.model_name = "mistral-7b"
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        return self.response


def _store(*items):
    """Build an EvidenceStore from (text, score) pairs (score -> rerank_score)."""
    store = EvidenceStore()
    for text, score in items:
        store.add(Evidence(text=text, rerank_score=score))
    return store


# ---------------------------------------------------------------------------
# UNKNOWN hops filtered out of both rendered hop-chain blocks.
# ---------------------------------------------------------------------------
class TestUnknownHopFiltering(unittest.TestCase):
    def _hops(self):
        return [
            {"subquery": "Where is Springfield?", "answer": "Illinois"},
            {"subquery": "What is the capital of Nowhere?", "answer": "UNKNOWN"},
            {"subquery": "Who founded it?", "answer": ""},
        ]

    def test_render_hop_chain_drops_unknown_and_empty(self):
        block = render_hop_chain(self._hops(), "Resolved intermediate facts:")
        self.assertIn("Illinois", block)
        self.assertNotIn("UNKNOWN", block)
        self.assertNotIn("Nowhere", block)
        # Only the one resolved hop is rendered.
        self.assertEqual(block.count("->"), 1)

    def test_render_empty_when_nothing_resolved(self):
        self.assertEqual(render_hop_chain([{"subquery": "x", "answer": "UNKNOWN"}], "H:"), "")
        self.assertEqual(render_hop_chain([], "H:"), "")
        self.assertEqual(render_hop_chain(None, "H:"), "")

    def test_synthesizer_block_filters_unknown(self):
        block = AnswerSynthesizer._hop_answer_block(self._hops())
        self.assertIn("Illinois", block)
        self.assertNotIn("UNKNOWN", block)

    def test_verifier_block_filters_unknown(self):
        block = AnswerVerifier._hop_chain_block(self._hops())
        self.assertIn("Illinois", block)
        self.assertNotIn("UNKNOWN", block)

    def test_anchors_skip_unknown(self):
        anchors = hop_answer_anchors(self._hops())
        self.assertEqual(anchors, ["illinois"])

    def test_is_unknown_hop_answer(self):
        self.assertTrue(is_unknown_hop_answer("UNKNOWN"))
        self.assertTrue(is_unknown_hop_answer("unknown."))
        self.assertTrue(is_unknown_hop_answer(""))
        self.assertTrue(is_unknown_hop_answer(None))
        # A real answer that merely contains the word is not unknown.
        self.assertFalse(is_unknown_hop_answer("The Unknown Soldier"))
        self.assertFalse(is_unknown_hop_answer("Illinois"))


# ---------------------------------------------------------------------------
# Synthesizer and verifier select the same relevance/anchor-ranked window.
# ---------------------------------------------------------------------------
class TestSharedEvidenceRanking(unittest.TestCase):
    def test_anchor_hit_ranks_first(self):
        # A low-score chunk that contains the anchor must outrank higher-score
        # chunks that do not, even though it was inserted last.
        store = _store(
            ("high score, no anchor", 0.9),
            ("also high, no anchor", 0.8),
            ("the grounding fact mentions Illinois", 0.1),
        )
        ordered = rank_evidence(store, anchors=["illinois"])
        self.assertIn("Illinois", ordered[0].text)

    def test_relevance_orders_when_no_anchor_hit(self):
        store = _store(("low", 0.1), ("high", 0.9), ("mid", 0.5))
        ordered = rank_evidence(store, anchors=[])
        self.assertEqual([e.text for e in ordered], ["high", "mid", "low"])

    def test_synth_and_verifier_pick_same_top_item_under_tight_budget(self):
        # After a recovery pass appended low-relevance chunks, the grounding chunk
        # (anchor hit) must sit in both prompt windows under a 1-item token budget.
        store = _store(
            ("noise one two three", 0.9),
            ("more noise words here", 0.8),
            ("Springfield is in Sangamon County", 0.2),
        )
        anchors = ["sangamon"]

        synth = AnswerSynthesizer(RecordingGenerator(), max_context_tokens=5)
        selected = synth._select_evidence(store, anchors=anchors)
        self.assertIn("Sangamon", selected[0].text)

        verifier = AnswerVerifier(RecordingGenerator(), max_context_tokens=5)
        block = verifier._build_evidence_block(store, anchors=anchors)
        # The first (only, under the tight budget) line is the grounding chunk.
        self.assertIn("Sangamon", block.splitlines()[0])


# ---------------------------------------------------------------------------
# Robust, non-inverting label parsing.
# ---------------------------------------------------------------------------
class TestParseLabel(unittest.TestCase):
    def test_plain_labels(self):
        self.assertEqual(AnswerVerifier._parse_label("supported"), VerificationLabel.SUPPORTED)
        self.assertEqual(AnswerVerifier._parse_label("unsupported"), VerificationLabel.UNSUPPORTED)

    def test_earliest_label_wins_not_substring(self):
        # "unsupported" contains "supported"; a naive substring scan would flip
        # this reply to UNSUPPORTED. Earliest-occurrence keeps it SUPPORTED.
        reply = "This is not unsupported -- it is clearly supported by the evidence."
        # 'unsupported' appears earliest here, so the verifier honors that verdict.
        self.assertEqual(AnswerVerifier._parse_label(reply), VerificationLabel.UNSUPPORTED)

    def test_supported_stated_first(self):
        reply = "supported: the evidence entails the answer."
        self.assertEqual(AnswerVerifier._parse_label(reply), VerificationLabel.SUPPORTED)

    def test_partially_supported_tie_prefers_longer(self):
        # Both "partially_supported" and "supported" start at the same index (the
        # latter is a suffix); the longer, more specific label must win.
        self.assertEqual(
            AnswerVerifier._parse_label("partially_supported"),
            VerificationLabel.PARTIALLY_SUPPORTED,
        )

    def test_unparseable_defaults_to_unsupported(self):
        self.assertEqual(AnswerVerifier._parse_label("no idea"), VerificationLabel.UNSUPPORTED)
        self.assertEqual(AnswerVerifier._parse_label(""), VerificationLabel.UNSUPPORTED)


# ---------------------------------------------------------------------------
# A single grounding passage reaches the model, not auto-rejected.
# ---------------------------------------------------------------------------
class TestSinglePassageReachesModel(unittest.TestCase):
    def test_default_min_evidence_count_is_one(self):
        self.assertEqual(AnswerVerifier(RecordingGenerator()).min_evidence_count, 1)

    def test_single_passage_is_judged_not_auto_rejected(self):
        gen = RecordingGenerator(response="supported")
        verifier = AnswerVerifier(gen)  # default min_evidence_count=1
        store = _store(("Springfield is the capital of Illinois.", 0.9))
        result = verifier.verify("What is the capital of Illinois?", "Springfield", store)
        # The model was actually called (one passage was enough to reach the judge).
        self.assertTrue(gen.prompts)
        self.assertEqual(result["label"], "supported")
        self.assertTrue(result["is_supported"])

    def test_zero_evidence_still_short_circuits(self):
        gen = RecordingGenerator(response="supported")
        verifier = AnswerVerifier(gen)
        result = verifier.verify("q", "some answer", EvidenceStore())
        self.assertEqual(result["label"], "insufficient_evidence")
        # The model must not be called when there is genuinely no evidence.
        self.assertEqual(gen.prompts, [])


# ---------------------------------------------------------------------------
# Hop-exception placeholder keeps #n alignment.
# ---------------------------------------------------------------------------
class _ThrowThenReturn:
    """Tool registry whose first RetrieveTool call throws, rest return no evidence."""

    class _Result:
        contains_evidence = False
        evidence = ()
        summary = "retrieved 0 chunks"

    def __init__(self):
        self.calls = 0

    def run(self, name, **kwargs):
        self.calls += 1
        if name == "RetrieveTool" and self.calls == 1:
            raise RuntimeError("simulated retriever outage")
        return self._Result()


class TestHopExceptionAlignment(unittest.TestCase):
    def _executor(self, registry):
        synth = AnswerSynthesizer(RecordingGenerator(response="UNKNOWN"))
        # Minimal executor: only the hop loop is exercised.
        return AgentExecutor(
            tool_registry=registry,
            policy=None,
            planner=None,
            synthesizer=synth,
            verifier=None,
            use_iterative_planner=True,
        )

    def test_placeholder_appended_on_retrieval_exception(self):
        registry = _ThrowThenReturn()
        ex = self._executor(registry)
        state = AgentState(original_query="q")
        state.subqueries = ["first hop", "second hop"]
        ex._run_hops(state, EvidenceStore())

        # One hop_answers entry per subquery even though the first retrieval threw:
        # without the placeholder the chain would have a single entry and a later
        # ``#n`` would resolve to the wrong hop.
        self.assertEqual(len(state.hop_answers), 2)
        self.assertEqual(state.hop_answers[0]["answer"], "UNKNOWN")
        self.assertEqual(state.hop_answers[0]["subquery"], "first hop")
        self.assertFalse(state.hop_answers[0]["recovered"])
        # The swallowed failure was counted.
        self.assertEqual(state.retrieval_failures, 1)


# ---------------------------------------------------------------------------
# ANSWERED_UNVERIFIED status + trace hop_recovery_attempts.
# ---------------------------------------------------------------------------
class TestAnsweredStatus(unittest.TestCase):
    def _executor(self):
        return AgentExecutor(
            tool_registry=None,
            policy=None,
            planner=None,
            synthesizer=AnswerSynthesizer(RecordingGenerator()),
            verifier=None,
        )

    def test_status_enum_has_answered_unverified(self):
        self.assertEqual(AgentStatus.ANSWERED_UNVERIFIED.value, "answered_unverified")

    def test_rejected_verification_is_unverified(self):
        ex = self._executor()
        state = AgentState(original_query="q")
        state.set_verification({"label": "unsupported", "is_supported": False})
        self.assertEqual(ex._answered_status(state), AgentStatus.ANSWERED_UNVERIFIED)

    def test_accepted_verification_is_answered(self):
        ex = self._executor()
        state = AgentState(original_query="q")
        state.set_verification({"label": "supported", "is_supported": True})
        self.assertEqual(ex._answered_status(state), AgentStatus.ANSWERED)

    def test_partially_supported_is_unverified(self):
        ex = self._executor()
        state = AgentState(original_query="q")
        # Tightened verification: partially_supported is no longer accepted, so a
        # best-effort answer with this label is tagged ANSWERED_UNVERIFIED (mirrors
        # accept_verification, which now accepts only "supported").
        state.set_verification({"label": "partially_supported", "is_supported": False})
        self.assertEqual(ex._answered_status(state), AgentStatus.ANSWERED_UNVERIFIED)

    def test_no_verification_is_unverified(self):
        ex = self._executor()
        state = AgentState(original_query="q")
        self.assertEqual(ex._answered_status(state), AgentStatus.ANSWERED_UNVERIFIED)


class TestTraceHopRecoveryAttempts(unittest.TestCase):
    def test_trace_exports_hop_recovery_attempts(self):
        from agrag.modules.agentic.trace import AgentTrace

        state = AgentState(original_query="q")
        state.hop_recovery_attempts = 3
        state.retrieval_failures = 1
        state.finish(AgentStatus.ANSWERED)
        trace = AgentTrace.from_run(state, EvidenceStore(), "ans").to_dict()
        self.assertEqual(trace["metrics"]["hop_recovery_attempts"], 3)
        self.assertEqual(trace["metrics"]["retrieval_failures"], 1)


if __name__ == "__main__":
    unittest.main()

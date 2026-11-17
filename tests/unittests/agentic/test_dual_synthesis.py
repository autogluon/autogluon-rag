

import unittest

from agrag.modules.agentic.agentic_module import AgenticRAGModule
from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.executor import AgentExecutor
from agrag.modules.agentic.state import AgentState


class _RecordingGenerator:
    """Generator fake: returns a fixed (or queued) response, records prompts."""

    def __init__(self, responses="A", model_name="fake-model"):
        # ``responses`` may be a single string or an iterable of strings.
        self._responses = responses
        self._iter = None if isinstance(responses, str) else iter(responses)
        self.model_name = model_name
        self.prompts = []

    def generate_response(self, prompt, temperature=None):
        self.prompts.append((prompt, temperature))
        if self._iter is not None:
            return next(self._iter)
        return self._responses


class _QueueSynthesizer:
    """Synthesizer fake for ``_synthesize_final``: yields queued answers in order.

    Mirrors ``AnswerSynthesizer.generate`` -- accepts the same keyword arguments the
    executor passes (including ``temperature``) and returns ``(answer, used_ids)``.
    """

    def __init__(self, answers):
        self._iter = iter(answers)
        self.calls = []

    def generate(self, query, evidence_store, compressed_context=None, hop_answers=None,
                 answer_instruction=None, drop_query_prefix=False, temperature=None):
        self.calls.append(temperature)
        return next(self._iter), ["e1"]


class _FakeVerifier:
    """Verifier stand-in exposing just what ``_arbitrate`` touches."""

    def __init__(self, reply):
        self.generator_module = _RecordingGenerator(responses=reply)

    def _build_evidence_block(self, evidence_store, anchors=None):
        return "[E1] some evidence"


def _hop(answer):
    return {"subquery": "h", "hop_query": "h", "answer": answer, "recovered": False}


class TestSelfConsistencyVote(unittest.TestCase):
    """``_synthesize_final`` majority-votes over K samples."""

    def _executor(self, answers, k):
        return AgentExecutor(
            tool_registry=None, policy=None, planner=None,
            synthesizer=_QueueSynthesizer(answers), self_consistency=k,
        )

    def test_single_draft_when_k_is_one(self):
        ex = self._executor(["Paris"], k=1)
        state = AgentState(original_query="capital?")
        answer, ids = ex._synthesize_final(state, EvidenceStore())
        self.assertEqual(answer, "Paris")
        self.assertEqual(ids, ["e1"])
        self.assertEqual(state.llm_calls, 1)  # exactly one synthesis call
        self.assertEqual(len(ex.synthesizer.calls), 1)

    def test_majority_answer_wins(self):
        # greedy + 2 samples; "Paris" has the majority over a one-off misread.
        ex = self._executor(["Paris", "Paris", "London"], k=3)
        state = AgentState(original_query="capital?")
        answer, _ = ex._synthesize_final(state, EvidenceStore())
        self.assertEqual(answer, "Paris")
        self.assertEqual(state.llm_calls, 3)  # one call per sample

    def test_greedy_sampled_at_none_rest_at_temperature(self):
        ex = self._executor(["Paris", "Paris", "Paris"], k=3)
        ex._synthesize_final(AgentState(original_query="q?"), EvidenceStore())
        # First (greedy) draft uses no temperature override; the rest diversify.
        self.assertEqual(ex.synthesizer.calls[0], None)
        self.assertEqual(ex.synthesizer.calls[1], ex._SELF_CONSISTENCY_TEMPERATURE)
        self.assertEqual(ex.synthesizer.calls[2], ex._SELF_CONSISTENCY_TEMPERATURE)

    def test_normalized_forms_are_pooled(self):
        # "the Beatles" and "The Beatles." normalize equal -> two votes, they win.
        ex = self._executor(["the Beatles", "The Beatles.", "Rolling Stones"], k=3)
        answer, _ = ex._synthesize_final(AgentState(original_query="band?"), EvidenceStore())
        self.assertIn(answer, ("the Beatles", "The Beatles."))

    def test_tie_breaks_toward_greedy_draft(self):
        # Two distinct answers, one vote each -> keep the greedy (first) draft.
        ex = self._executor(["Alpha", "Beta"], k=2)
        answer, _ = ex._synthesize_final(AgentState(original_query="q?"), EvidenceStore())
        self.assertEqual(answer, "Alpha")

    def test_all_non_answers_returns_greedy_draft(self):
        # No real answer to vote on -> fall back to the greedy draft so the
        # downstream dual/fallback path still fires.
        ex = self._executor(["INSUFFICIENT EVIDENCE", "I don't have enough"], k=2)
        answer, _ = ex._synthesize_final(AgentState(original_query="q?"), EvidenceStore())
        self.assertEqual(answer, "INSUFFICIENT EVIDENCE")


class TestArbitrateParsing(unittest.TestCase):
    """``_arbitrate`` maps a verifier reply to chain/free defensively."""

    def _executor(self, reply):
        ex = AgentExecutor(tool_registry=None, policy=None, planner=None, synthesizer=None)
        ex.verifier = _FakeVerifier(reply)
        return ex

    def _arbitrate(self, reply):
        ex = self._executor(reply)
        return ex._arbitrate("q?", "chain answer", "free answer", EvidenceStore())

    def test_reply_a_picks_chain(self):
        self.assertEqual(self._arbitrate("A"), "chain")

    def test_reply_b_picks_free(self):
        self.assertEqual(self._arbitrate("B"), "free")

    def test_empty_reply_defaults_to_free(self):
        self.assertEqual(self._arbitrate(""), "free")

    def test_no_ab_token_defaults_to_free(self):
        # No 'a'/'b' character anywhere -> safe default (free).
        self.assertEqual(self._arbitrate("???"), "free")


class TestDualCandidateDecision(unittest.TestCase):
    """``_dual_candidate_answer`` decision table.

    ``_direct_answer_fallback`` and ``_arbitrate`` are stubbed so the test isolates
    the reconciliation logic from generator/verifier wiring.
    """

    CHAIN_V = {"label": "chain-verif"}
    ALT_V = {"label": "alt-verif"}

    def _executor(self, alt_answer, alt_accepted, arbitrate="free"):
        ex = AgentExecutor(tool_registry=None, policy=None, planner=None, synthesizer=None)

        def fake_fallback(state, store):
            # Mirror the real method's side effect: it sets last_verification to the
            # chain-free candidate's verification before returning.
            state.set_verification(self.ALT_V)
            return alt_answer, alt_accepted

        ex._direct_answer_fallback = fake_fallback
        ex._arbitrate = lambda q, c, f, store: arbitrate
        return ex

    def _state(self, hop_answers=None):
        state = AgentState(original_query="q?")
        state.hop_answers = hop_answers or []
        # Chain verification is on the state before the fallback runs.
        state.set_verification(self.CHAIN_V)
        return state

    def test_both_verified_and_agree_returns_chain(self):
        ex = self._executor(alt_answer="Paris", alt_accepted=True)
        state = self._state()
        answer, accepted = ex._dual_candidate_answer(state, EvidenceStore(), "Paris", True)
        self.assertEqual(answer, "Paris")
        self.assertTrue(accepted)
        self.assertEqual(state.verification, self.CHAIN_V)

    def test_both_verified_differ_arbitrates_to_free(self):
        ex = self._executor(alt_answer="Lyon", alt_accepted=True, arbitrate="free")
        state = self._state()
        answer, accepted = ex._dual_candidate_answer(state, EvidenceStore(), "Paris", True)
        self.assertEqual(answer, "Lyon")
        self.assertTrue(accepted)
        self.assertEqual(state.verification, self.ALT_V)

    def test_both_verified_differ_arbitrates_to_chain(self):
        ex = self._executor(alt_answer="Lyon", alt_accepted=True, arbitrate="chain")
        state = self._state()
        answer, accepted = ex._dual_candidate_answer(state, EvidenceStore(), "Paris", True)
        self.assertEqual(answer, "Paris")
        self.assertTrue(accepted)

    def test_only_chain_verified_takes_chain(self):
        ex = self._executor(alt_answer="Lyon", alt_accepted=False)
        state = self._state()
        answer, accepted = ex._dual_candidate_answer(state, EvidenceStore(), "Paris", True)
        self.assertEqual(answer, "Paris")
        self.assertTrue(accepted)

    def test_only_alt_verified_takes_alt(self):
        ex = self._executor(alt_answer="Lyon", alt_accepted=True)
        state = self._state()
        answer, accepted = ex._dual_candidate_answer(state, EvidenceStore(), "Paris", False)
        self.assertEqual(answer, "Lyon")
        self.assertTrue(accepted)
        self.assertEqual(state.verification, self.ALT_V)

    def test_neither_verified_prefers_real_over_non_answer(self):
        # Chain draft abstained; the chain-free candidate is a real answer -> take it.
        ex = self._executor(alt_answer="Lyon", alt_accepted=False)
        state = self._state()
        answer, accepted = ex._dual_candidate_answer(
            state, EvidenceStore(), "INSUFFICIENT EVIDENCE", False
        )
        self.assertEqual(answer, "Lyon")
        self.assertFalse(accepted)  # best-effort, never a canned abstention

    def test_neither_verified_both_real_broken_chain_prefers_free(self):
        # Both candidates real, neither verified, last hop UNKNOWN -> prefer chain-free.
        ex = self._executor(alt_answer="Lyon", alt_accepted=False)
        state = self._state(hop_answers=[_hop("Paris"), _hop("UNKNOWN")])
        answer, accepted = ex._dual_candidate_answer(state, EvidenceStore(), "Paris", False)
        self.assertEqual(answer, "Lyon")
        self.assertFalse(accepted)

    def test_neither_verified_both_real_intact_chain_prefers_chain(self):
        # Both real, neither verified, chain intact -> keep the chain draft.
        ex = self._executor(alt_answer="Lyon", alt_accepted=False)
        state = self._state(hop_answers=[_hop("Paris"), _hop("France")])
        answer, accepted = ex._dual_candidate_answer(state, EvidenceStore(), "Paris", False)
        self.assertEqual(answer, "Paris")
        self.assertFalse(accepted)

    def test_both_non_answers_keeps_chain(self):
        ex = self._executor(alt_answer="INSUFFICIENT EVIDENCE", alt_accepted=False)
        state = self._state()
        answer, accepted = ex._dual_candidate_answer(
            state, EvidenceStore(), "INSUFFICIENT EVIDENCE", False
        )
        self.assertEqual(answer, "INSUFFICIENT EVIDENCE")
        self.assertFalse(accepted)


class TestIndependentVerifierWiring(unittest.TestCase):
    """The verifier can use its own generator, distinct from synthesis."""

    def _module(self, verifier_gen):
        gen = _RecordingGenerator(model_name="synth-model")
        return AgenticRAGModule(
            retriever_module=object(),
            generator_module=gen,
            config={"use_verification": True},
            verifier_generator_module=verifier_gen,
        ), gen

    def test_verifier_uses_independent_generator_when_provided(self):
        verifier_gen = _RecordingGenerator(model_name="verifier-model")
        module, gen = self._module(verifier_gen)
        self.assertIs(module.verifier.generator_module, verifier_gen)
        self.assertIsNot(module.verifier.generator_module, gen)
        # The synthesizer still uses the drafting generator.
        self.assertIs(module.synthesizer.generator_module, gen)

    def test_verifier_reuses_generator_when_none(self):
        module, gen = self._module(None)
        self.assertIs(module.verifier.generator_module, gen)


if __name__ == "__main__":
    unittest.main()

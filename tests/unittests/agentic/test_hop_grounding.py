"""Unit tests for hop-answer grounding gates in the agentic path.

These cover:

* ``is_unknown_hop_answer`` catching a trailing/embedded all-caps ``UNKNOWN``
  sentinel (a query-poisoning guard), and ``_is_substitutable`` only inlining a
  clean short span into a ``#n`` reference.
* Discriminative verification: showing the verifier the full chain with gaps
  marked (``render_hop_chain_with_gaps``), plus a deterministic numeric grounding
  gate that downgrades an accept when the answer asserts a salient number absent
  from every evidence item.
* The grounding clause in the final-answer instruction, and withholding
  acceptance when the final hop never resolved so recovery re-decomposes.
* Bounding the sequential hop loop with a global ``max_total_hops`` budget.
* Stripping articles in exact-match / token-F1 normalization.

They use lightweight fakes and run entirely offline (no Bedrock).
"""

import unittest

from agrag.evaluation.utils import inclusive_exact_match_metric, preprocess_text, token_f1
from agrag.modules.agentic.evidence import Evidence, EvidenceStore
from agrag.modules.agentic.executor import AgentExecutor
from agrag.modules.agentic.hop_utils import is_unknown_hop_answer, render_hop_chain_with_gaps
from agrag.modules.agentic.state import AgentState
from agrag.modules.agentic.synthesizer import AnswerSynthesizer
from agrag.modules.agentic.verifier import AnswerVerifier, VerificationLabel


class RecordingGenerator:
    """Generator fake that records prompts and returns a fixed response."""

    def __init__(self, response="Illinois"):
        self.response = response
        self.model_name = "mistral-7b"
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        return self.response


class _FakeResult:
    contains_evidence = False
    evidence = ()
    summary = "retrieved 0 chunks"


class _FakeToolRegistry:
    def __init__(self):
        self.calls = []

    def run(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return _FakeResult()


def _store(*texts):
    store = EvidenceStore()
    for t in texts:
        store.add(Evidence(text=t))
    return store


# --- Unresolved-hop detection ------------------------------------------------


class TestIsUnknownHopAnswer(unittest.TestCase):
    def test_bare_and_empty_are_unknown(self):
        self.assertTrue(is_unknown_hop_answer(""))
        self.assertTrue(is_unknown_hop_answer("   "))
        self.assertTrue(is_unknown_hop_answer("UNKNOWN"))
        self.assertTrue(is_unknown_hop_answer("unknown."))

    def test_embedded_allcaps_sentinel_is_unknown(self):
        # The query-poisoning shape: verbose prose that trails off into UNKNOWN.
        self.assertTrue(
            is_unknown_hop_answer("A Don is from Laos. The context does not name the tournament. UNKNOWN")
        )

    def test_titlecased_unknown_is_a_real_answer(self):
        # A title containing "Unknown" (not the all-caps sentinel) stays resolved.
        self.assertFalse(is_unknown_hop_answer("The Unknown Soldier"))
        self.assertFalse(is_unknown_hop_answer("Illinois"))


# --- Substitution guard -------------------------------------------------------


class TestIsSubstitutable(unittest.TestCase):
    def test_short_span_is_substitutable(self):
        self.assertTrue(AgentExecutor._is_substitutable("Berlin"))
        self.assertTrue(AgentExecutor._is_substitutable("January 2015"))
        self.assertTrue(AgentExecutor._is_substitutable("United States of America"))

    def test_empty_is_not_substitutable(self):
        self.assertFalse(AgentExecutor._is_substitutable(""))

    def test_prose_and_overlong_are_not_substitutable(self):
        self.assertFalse(AgentExecutor._is_substitutable("It is Laos. The tournament is unclear"))
        self.assertFalse(
            AgentExecutor._is_substitutable("one two three four five six seven eight nine")
        )

    def test_prose_hop_left_untouched_in_reference(self):
        ex = AgentExecutor(None, None, None, AnswerSynthesizer(RecordingGenerator()))
        state = AgentState(original_query="q?")
        state.hop_answers = [
            {"subquery": "h1", "hop_query": "h1", "answer": "A Don is from Laos. Unclear.", "recovered": False}
        ]
        # The verbose hop is not inlined -> #1 stays for the LLM fill-in to handle.
        self.assertEqual(ex._substitute_refs("what tournament used #1", state), "what tournament used #1")


# --- Verifier sees gaps + numeric grounding gate ------------------------------


class TestRenderHopChainWithGaps(unittest.TestCase):
    def test_unresolved_hop_marked(self):
        block = render_hop_chain_with_gaps(
            [
                {"subquery": "region north of Israel", "answer": "Lebanon"},
                {"subquery": "when established", "answer": "UNKNOWN"},
            ],
            "REASONING CHAIN:",
        )
        self.assertIn("region north of Israel -> Lebanon", block)
        self.assertIn("when established -> (NOT RESOLVED from evidence)", block)

    def test_empty_when_no_hops(self):
        self.assertEqual(render_hop_chain_with_gaps([], "H:"), "")


class TestVerifierNumericGate(unittest.TestCase):
    def test_ungrounded_number_downgrades_accept(self):
        # Model rubber-stamps "supported" but the year is absent from evidence.
        verifier = AnswerVerifier(RecordingGenerator(response="supported"))
        store = _store("The country was founded after a long struggle for independence.")
        result = verifier.verify("When was it founded?", "1943", store)
        self.assertEqual(result["label"], VerificationLabel.UNSUPPORTED.value)
        self.assertFalse(result["is_supported"])

    def test_grounded_number_survives(self):
        verifier = AnswerVerifier(RecordingGenerator(response="supported"))
        store = _store("The country was founded in 1943 after independence.")
        result = verifier.verify("When was it founded?", "1943", store)
        self.assertEqual(result["label"], VerificationLabel.SUPPORTED.value)
        self.assertTrue(result["is_supported"])

    def test_comma_grouped_number_matches(self):
        self.assertFalse(
            AnswerVerifier._has_ungrounded_number("1,000 troops", _store("a force of 1000 troops"))
        )

    def test_non_numeric_answer_not_gated(self):
        self.assertFalse(AnswerVerifier._has_ungrounded_number("Lebanon", _store("north of Israel")))

    def test_unknown_draft_is_non_answer(self):
        verifier = AnswerVerifier(RecordingGenerator(response="supported"))
        store = _store("some evidence", "more evidence")
        result = verifier.verify("q?", "A Don is from Laos. UNKNOWN", store)
        # Never calls the model for a non-answer draft.
        self.assertEqual(result["label"], VerificationLabel.UNSUPPORTED.value)
        self.assertEqual(verifier.generator_module.prompts, [])

    def test_verifier_prompt_includes_gap(self):
        gen = RecordingGenerator(response="supported")
        verifier = AnswerVerifier(gen)
        store = _store("Lebanon is north of Israel.")
        verifier.verify(
            "when established",
            "1932",
            store,
            hop_answers=[{"subquery": "region", "answer": "Lebanon"}, {"subquery": "founded", "answer": "UNKNOWN"}],
        )
        self.assertIn("(NOT RESOLVED from evidence)", gen.prompts[0])


# --- Grounding clause + force-replan on unresolved final hop ------------------


class TestFinalAnswerGrounding(unittest.TestCase):
    def test_final_instruction_has_grounding_clause(self):
        low = AgentExecutor._FINAL_ANSWER_INSTRUCTION.lower()
        self.assertIn("only on the provided context", low)
        self.assertIn("most specific value", low)


class _FakePlanner:
    def __init__(self):
        self.replan_calls = 0

    def create_plan(self, query):
        return [query, "sub-a", "sub-b"]

    def replan(self, query, previous_plan=None, feedback=None):
        self.replan_calls += 1
        return [query] + [f"r{self.replan_calls}-{i}" for i in range(2)]


class TestForceReplanOnUnresolvedFinalHop(unittest.TestCase):
    def test_final_hop_unresolved_helper(self):
        state = AgentState(original_query="q?")
        state.hop_answers = [{"subquery": "a", "answer": "X"}, {"subquery": "b", "answer": "UNKNOWN"}]
        self.assertTrue(AgentExecutor._final_hop_unresolved(state))
        state.hop_answers[-1]["answer"] = "Y"
        self.assertFalse(AgentExecutor._final_hop_unresolved(state))

    def test_unresolved_final_hop_triggers_replan_despite_supported(self):
        planner = _FakePlanner()
        ex = AgentExecutor(
            tool_registry=_FakeToolRegistry(),
            policy=None,
            planner=planner,
            synthesizer=AnswerSynthesizer(RecordingGenerator()),
            use_replan_recovery=True,
            allow_abstention=False,
            max_recovery_attempts=2,
        )
        # The verifier always accepts ("supported"); only the final-hop gate should
        # withhold acceptance and force one replan, after which the hop resolves.
        calls = {"n": 0}

        def fake_run_hops(state, store):
            calls["n"] += 1
            if calls["n"] == 1:
                state.hop_answers = [{"subquery": "a", "answer": "X"}, {"subquery": "b", "answer": "UNKNOWN"}]
            else:
                state.hop_answers = [{"subquery": "a", "answer": "X"}, {"subquery": "b", "answer": "Resolved"}]

        ex._run_hops = fake_run_hops
        ex._draft_and_verify = lambda state, store: ("ans", True)

        state = AgentState(original_query="q?")
        state.plan = ["q?", "sub-a", "sub-b"]
        state.subqueries = state.plan[1:]
        _, _, answer = ex._run_sequential(state, EvidenceStore())

        self.assertEqual(answer, "ans")
        self.assertEqual(planner.replan_calls, 1)  # one replan because hop1's final hop was UNKNOWN


# --- Global hop budget --------------------------------------------------------


class TestHopBudget(unittest.TestCase):
    def _executor(self, **kwargs):
        return AgentExecutor(
            tool_registry=_FakeToolRegistry(),
            policy=None,
            planner=None,
            synthesizer=AnswerSynthesizer(RecordingGenerator()),
            **kwargs,
        )

    def test_budget_caps_hop_expansion(self):
        ex = self._executor(max_total_hops=2)
        ex._extract_hop_answer = lambda q, store: "X"
        state = AgentState(original_query="q?")
        state.subqueries = ["hop one alpha", "hop two beta", "hop three gamma", "hop four delta"]
        ex._run_hops(state, EvidenceStore())
        # Only two hops execute; the budget stops expansion before the third.
        self.assertEqual(len(state.hop_answers), 2)
        retrievals = [c for c in ex.tool_registry.calls if c[0] == "RetrieveTool"]
        self.assertEqual(len(retrievals), 2)

    def test_no_budget_runs_all_hops(self):
        ex = self._executor()  # max_total_hops=None
        ex._extract_hop_answer = lambda q, store: "X"
        state = AgentState(original_query="q?")
        state.subqueries = ["hop one alpha", "hop two beta", "hop three gamma"]
        ex._run_hops(state, EvidenceStore())
        self.assertEqual(len(state.hop_answers), 3)


# --- Article-stripping normalization ------------------------------------------


class TestArticleStripping(unittest.TestCase):
    def test_preprocess_strips_articles(self):
        self.assertEqual(
            preprocess_text("The Beatles", ignore_case=True, ignore_articles=True), "beatles"
        )
        self.assertEqual(
            preprocess_text("a man and an idea", ignore_case=True, ignore_articles=True), "man and idea"
        )

    def test_preprocess_keeps_articles_by_default(self):
        self.assertEqual(preprocess_text("The Beatles", ignore_case=True), "the beatles")

    def test_inclusive_em_matches_across_articles(self):
        matches = inclusive_exact_match_metric(
            ["Beatles"], [["the Beatles"]], ignore_case=True, ignore_punctuation=True, substring=False
        )
        self.assertTrue(matches[0])

    def test_token_f1_ignores_articles(self):
        self.assertEqual(token_f1("the quick fox", ["quick fox"]), 1.0)


if __name__ == "__main__":
    unittest.main()

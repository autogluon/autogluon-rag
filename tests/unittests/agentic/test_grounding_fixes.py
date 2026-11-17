"""Unit tests for grounding in the agentic hop path.

These cover:

* The hop synthesizer passing a strict-grounding ``answer_instruction`` so
  intermediate hops answer from retrieved evidence, not parametric memory.
* ``_needs_resolution`` treating bracketed ``[placeholder]`` slots and ``#n``
  back-references as unresolved, so ``_resolve_hop_query`` fills them.
* The LLM planner prompt telling the model to keep every anchor in a conjunctive
  constraint.
* The short-answer extraction prompt keeping quantifiers/qualifiers.

They use lightweight fakes and exercise the code paths offline (no Bedrock).
"""

import unittest

from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.executor import AgentExecutor
from agrag.modules.agentic.planner import _PLAN_INSTRUCTION, _REPLAN_DIRECTIVE, QueryPlanner
from agrag.modules.agentic.policy import DecisionPolicy
from agrag.modules.agentic.state import AgentState
from agrag.modules.agentic.synthesizer import _EXTRACT_INSTRUCTION, AnswerSynthesizer


class RecordingGenerator:
    """Generator fake that records the prompts it is asked to complete."""

    def __init__(self, response="Illinois"):
        self.response = response
        self.model_name = "mistral-7b"
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        return self.response


class _FakeResult:
    """Minimal ToolResult stand-in with no evidence (ingest is a no-op)."""

    contains_evidence = False
    evidence = ()
    summary = "retrieved 0 chunks"


class _FakeToolRegistry:
    """Records tool calls and returns an evidence-free result."""

    def __init__(self):
        self.calls = []

    def run(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return _FakeResult()


class TestHopGroundingInstruction(unittest.TestCase):
    """Hop answering forces grounding in the retrieved context."""

    def test_generate_prepends_answer_instruction(self):
        gen = RecordingGenerator()
        synth = AnswerSynthesizer(gen)
        instruction = "Answer using ONLY the provided context."
        synth.generate("Where was X born?", EvidenceStore(), answer_instruction=instruction)
        self.assertEqual(len(gen.prompts), 1)
        self.assertIn(instruction, gen.prompts[0])

    def test_default_generate_has_no_grounding_instruction(self):
        gen = RecordingGenerator()
        synth = AnswerSynthesizer(gen)
        synth.generate("Where was X born?", EvidenceStore())
        self.assertNotIn("ONLY the provided context", gen.prompts[0])

    def test_extract_hop_answer_passes_grounding_instruction(self):
        gen = RecordingGenerator(response="Laos")
        synth = AnswerSynthesizer(gen)
        executor = AgentExecutor(None, None, None, synth)
        answer = executor._extract_hop_answer("Where is the A Don river?", EvidenceStore())
        self.assertEqual(answer, "Laos")
        # The strict-grounding directive reached the generator prompt.
        self.assertIn("UNKNOWN", gen.prompts[0])
        self.assertIn("ONLY the provided context", gen.prompts[0])


class TestPlaceholderResolution(unittest.TestCase):
    """Bracketed / numbered placeholders count as unresolved references."""

    def _executor(self):
        return AgentExecutor(None, None, None, AnswerSynthesizer(RecordingGenerator()))

    def test_bracketed_placeholder_needs_resolution(self):
        ex = self._executor()
        self.assertTrue(ex._needs_resolution("birthplace of [performer]"))

    def test_numbered_backreference_needs_resolution(self):
        ex = self._executor()
        self.assertTrue(ex._needs_resolution("what county contains #2"))

    def test_plain_selfcontained_query_needs_no_resolution(self):
        ex = self._executor()
        self.assertFalse(ex._needs_resolution("what county contains Springfield"))

    def test_pronoun_still_needs_resolution(self):
        ex = self._executor()
        self.assertTrue(ex._needs_resolution("which county is it in"))


class TestPromptDirectives(unittest.TestCase):
    """The planner and extraction prompt strings carry the grounding directives."""

    def test_planner_prompt_preserves_conjunctive_anchors(self):
        self.assertIn("keep ALL of them together", _PLAN_INSTRUCTION)

    def test_extract_prompt_keeps_qualifiers(self):
        low = _EXTRACT_INSTRUCTION.lower()
        self.assertIn("qualifier", low)
        self.assertIn("teak", low)


class TestHopRecovery(unittest.TestCase):
    """UNKNOWN-hop recovery: candidate is used only as a search seed, never stored."""

    def _executor(self, **kwargs):
        synth = AnswerSynthesizer(RecordingGenerator())
        kwargs.setdefault("use_hop_recovery", True)
        return AgentExecutor(
            tool_registry=_FakeToolRegistry(),
            policy=None,
            planner=None,
            synthesizer=synth,
            **kwargs,
        )

    def test_unknown_hop_recovered_via_reretrieval(self):
        ex = self._executor()
        # First extraction UNKNOWN; after the candidate-seeded re-retrieval it grounds.
        extractions = iter(["UNKNOWN", "Rowan County"])
        ex._extract_hop_answer = lambda q, store: next(extractions)
        ex._hypothesize_candidate = lambda q, state, store: "Rowan County"

        state = AgentState(original_query="what county contains Cleveland?")
        state.subqueries = ["what county contains Cleveland"]  # self-contained -> no fill-in
        ex._run_hops(state, EvidenceStore())

        self.assertEqual(len(state.hop_answers), 1)
        self.assertEqual(state.hop_answers[0]["answer"], "Rowan County")
        self.assertTrue(state.hop_answers[0]["recovered"])
        # The recovery issued a second RetrieveTool call seeded with the candidate.
        retrieval_calls = [c for c in ex.tool_registry.calls if c[0] == "RetrieveTool"]
        self.assertEqual(len(retrieval_calls), 2)
        self.assertIn("Rowan County", retrieval_calls[1][1]["query"])

    def test_still_unknown_hop_never_stores_the_guess(self):
        ex = self._executor()
        ex._extract_hop_answer = lambda q, store: "UNKNOWN"  # never grounds
        ex._hypothesize_candidate = lambda q, state, store: "Cuyahoga County"

        state = AgentState(original_query="what county contains Cleveland?")
        state.subqueries = ["what county contains Cleveland"]
        ex._run_hops(state, EvidenceStore())

        self.assertTrue(ex._is_unknown_hop_answer(state.hop_answers[0]["answer"]))
        self.assertFalse(state.hop_answers[0]["recovered"])
        # The parametric guess must not leak into the stored hop answer.
        self.assertNotIn("Cuyahoga", state.hop_answers[0]["answer"])

    def test_recovery_off_makes_no_extra_retrieval(self):
        ex = self._executor(use_hop_recovery=False)
        ex._extract_hop_answer = lambda q, store: "UNKNOWN"
        ex._hypothesize_candidate = lambda q, state, store: "should-not-be-called"

        state = AgentState(original_query="q?")
        state.subqueries = ["a self contained hop query"]
        ex._run_hops(state, EvidenceStore())

        retrieval_calls = [c for c in ex.tool_registry.calls if c[0] == "RetrieveTool"]
        self.assertEqual(len(retrieval_calls), 1)  # no recovery retrieval


class _FakePlanner:
    """Planner fake: create_plan is fixed; replan yields a distinct plan per call."""

    def __init__(self):
        self.replan_calls = 0

    def create_plan(self, query):
        return [query, "sub-a", "sub-b"]

    def replan(self, query, previous_plan=None, feedback=None):
        self.replan_calls += 1
        # A genuinely different, finer-grained plan each call so recovery proceeds.
        return [query] + [f"r{self.replan_calls}-{i}" for i in range(4)]


class TestReplanRecovery(unittest.TestCase):
    """Verifier-triggered whole-plan re-decomposition on a failed final answer."""

    def _executor(self, verify_verdicts, planner, **kwargs):
        ex = AgentExecutor(
            tool_registry=_FakeToolRegistry(),
            policy=None,
            planner=planner,
            synthesizer=AnswerSynthesizer(RecordingGenerator()),
            use_replan_recovery=True,
            allow_abstention=False,
            **kwargs,
        )
        # Isolate the recovery control flow from tool/generator wiring.
        ex._run_hops = lambda state, store: None
        verdicts = iter(verify_verdicts)

        def fake_draft(state, store):
            state.draft_answer = "ans"
            accepted = next(verdicts)
            state.set_verification({"label": "supported" if accepted else "unsupported"})
            return "ans", accepted

        ex._draft_and_verify = fake_draft
        return ex

    def test_replan_until_accepted(self):
        planner = _FakePlanner()
        # First draft rejected, replan, second draft accepted.
        ex = self._executor([False, True], planner, max_recovery_attempts=2)
        state = AgentState(original_query="q?")
        state.plan = ["q?", "sub-a", "sub-b"]
        state.subqueries = state.plan[1:]

        _, _, answer = ex._run_sequential(state, EvidenceStore())
        self.assertEqual(answer, "ans")
        self.assertEqual(planner.replan_calls, 1)
        self.assertEqual(state.recovery_attempts, 1)

    def test_replan_bounded_by_max_attempts(self):
        planner = _FakePlanner()
        # Always rejected -> exhaust the recovery budget, then best-effort answer.
        ex = self._executor([False, False, False, False], planner, max_recovery_attempts=2)
        state = AgentState(original_query="q?")
        state.plan = ["q?", "sub-a", "sub-b"]
        state.subqueries = state.plan[1:]

        _, _, answer = ex._run_sequential(state, EvidenceStore())
        self.assertEqual(planner.replan_calls, 2)  # capped
        self.assertEqual(state.recovery_attempts, 2)
        self.assertEqual(answer, "ans")  # never-refuse best effort


class TestReplanPlanner(unittest.TestCase):
    """QueryPlanner.replan produces a different, higher-cap decomposition."""

    def test_replan_uses_directive_and_raised_cap(self):
        gen = RecordingGenerator(response='{"subqueries": ["a", "b", "c", "d", "e", "f"]}')
        planner = QueryPlanner(max_subqueries=4, generator_module=gen, use_llm=True)
        plan = planner.replan("orig question?", previous_plan=["orig question?", "old-sub"])
        # cap is max_subqueries + 2 = 6, so all six subqueries survive (+ original).
        self.assertEqual(len(plan), 7)
        self.assertEqual(plan[0], "orig question?")
        self.assertIn("RETRY", gen.prompts[0])
        self.assertIn("old-sub", gen.prompts[0])


def _hop(subquery, answer):
    return {"subquery": subquery, "hop_query": subquery, "answer": answer, "recovered": False}


class TestMultiReferenceResolution(unittest.TestCase):
    """Deterministic ``#n`` substitution handles branching shapes (3hop2/4hop2/4hop3).

    A hop may reference two or more earlier hops; every reference must resolve, and
    the substitution must not be starved or corrupted by the model.
    """

    def _executor(self, gen=None):
        return AgentExecutor(None, None, None, AnswerSynthesizer(gen or RecordingGenerator()))

    def test_substitutes_multiple_references_in_one_hop(self):
        gen = RecordingGenerator()
        ex = self._executor(gen)
        state = AgentState(original_query="q?")
        state.hop_answers = [_hop("h1", "Alice"), _hop("h2", "Berlin"), _hop("h3", "1990")]
        resolved = ex._resolve_hop_query("What happened in #2 in #3 involving #1?", state)
        self.assertEqual(resolved, "What happened in Berlin in 1990 involving Alice?")
        # All refs resolved deterministically -> no LLM fill-in was needed.
        self.assertEqual(gen.prompts, [])

    def test_double_digit_reference_not_corrupted_by_single_digit(self):
        ex = self._executor()
        state = AgentState(original_query="q?")
        state.hop_answers = [_hop(f"h{i}", f"ANS{i}") for i in range(1, 11)]
        # ``#10`` must map to the tenth hop, not to ``#1`` + literal "0".
        self.assertEqual(ex._substitute_refs("compare #1 and #10", state), "compare ANS1 and ANS10")

    def test_unknown_referenced_hop_is_left_untouched(self):
        ex = self._executor()
        state = AgentState(original_query="q?")
        state.hop_answers = [_hop("h1", "Alice"), _hop("h2", "UNKNOWN")]
        # #1 resolves; #2 (UNKNOWN) is left as-is for the LLM fill-in to generalize.
        self.assertEqual(ex._substitute_refs("link #1 to #2", state), "link Alice to #2")

    def test_out_of_range_reference_is_left_untouched(self):
        ex = self._executor()
        state = AgentState(original_query="q?")
        state.hop_answers = [_hop("h1", "Alice")]
        self.assertEqual(ex._substitute_refs("who is #5", state), "who is #5")

    def test_pronoun_after_ref_substitution_still_triggers_fill_in(self):
        # #1 substitutes deterministically; the remaining pronoun forces the LLM pass.
        gen = RecordingGenerator(response="what year was Alice's band formed")
        ex = self._executor(gen)
        state = AgentState(original_query="q?")
        state.hop_answers = [_hop("h1", "Alice")]
        resolved = ex._resolve_hop_query("what year was #1's band and its label formed", state)
        self.assertEqual(len(gen.prompts), 1)  # fill-in ran for the leftover reference
        self.assertIn("Alice", gen.prompts[0])  # the ref-substituted query reached the prompt
        self.assertEqual(resolved, "what year was Alice's band formed")


class TestRecoveryBudgetSeparation(unittest.TestCase):
    """Each recovery mechanism draws down its own budget, not the shared iteration."""

    def test_local_recovery_uses_its_own_counter_not_iteration(self):
        synth = AnswerSynthesizer(RecordingGenerator())
        ex = AgentExecutor(
            tool_registry=_FakeToolRegistry(), policy=None, planner=None,
            synthesizer=synth, use_hop_recovery=True,
        )
        extractions = iter(["UNKNOWN", "Rowan County"])
        ex._extract_hop_answer = lambda q, store: next(extractions)
        ex._hypothesize_candidate = lambda q, state, store: "Rowan County"

        state = AgentState(original_query="what county contains Cleveland?")
        state.subqueries = ["what county contains Cleveland"]
        ex._run_hops(state, EvidenceStore())

        self.assertTrue(state.hop_answers[0]["recovered"])
        self.assertEqual(state.hop_recovery_attempts, 1)  # local recovery counted here
        self.assertEqual(state.iteration, 1)  # only the per-hop bump, not the recovery pass

    def test_verify_retry_gate_decoupled_from_iteration(self):
        policy = DecisionPolicy(use_query_rewrite=True, max_rewrites=2, max_iterations=5)
        state = AgentState(original_query="q?")
        state.iteration = 5  # hops exhausted the iteration budget
        # The iteration-gated gate refuses; the post-hoc gate (used by the sequential
        # verify-retry) still admits because rewrite budget remains.
        self.assertFalse(policy._can_rewrite(state))
        self.assertTrue(policy._can_rewrite_posthoc(state))


class TestMultiReferencePromptDirectives(unittest.TestCase):
    """The planner/replan prompts instruct the model to emit multiple back-references."""

    def test_plan_prompt_mentions_multiple_dependencies(self):
        low = _PLAN_INSTRUCTION.lower()
        self.assertIn("more than one earlier subquery", low)

    def test_replan_prompt_mentions_multiple_dependencies(self):
        self.assertIn("two or more earlier hops", _REPLAN_DIRECTIVE.lower())


if __name__ == "__main__":
    unittest.main()

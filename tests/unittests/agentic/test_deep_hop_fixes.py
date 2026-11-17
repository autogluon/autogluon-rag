"""Unit tests for deep-hop robustness in the agentic path.

Each behavior is independently flag-gated and default-off; these tests exercise
them offline (no Bedrock) with lightweight fakes and assert that with the flags
off the behavior is unchanged.

* The ``hop_answer_ungrounded`` entity-grounding gate on hop answers, wired into
  ``executor._run_hops`` (gate -> recovery -> UNKNOWN downgrade).
* ``RetrieveTool.run`` honoring a per-call ``top_k``, and ``executor._run_hops``
  requesting ``deep_hop_top_k`` for dependency-bearing / deep hops.
* ``synthesizer.generate(drop_query_prefix=...)`` skipping the single-word
  ``query_prefix`` on the final multi-hop synthesis.
"""

import unittest

from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.executor import AgentExecutor
from agrag.modules.agentic.hop_utils import hop_answer_ungrounded
from agrag.modules.agentic.state import AgentState
from agrag.modules.agentic.synthesizer import AnswerSynthesizer


class RecordingGenerator:
    """Generator fake that records the prompts it is asked to complete."""

    def __init__(self, response="Illinois"):
        self.response = response
        self.model_name = "mistral-7b"
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        return self.response


class _Ev:
    """Minimal Evidence stand-in exposing just the ``.text`` the gate reads."""

    def __init__(self, text):
        self.text = text


class _EvResult:
    """ToolResult stand-in carrying evidence texts but ingesting nothing.

    ``contains_evidence`` is False so ``_ingest_evidence`` is a no-op (keeps the
    EvidenceStore empty and the fakes simple), while ``evidence`` is still
    populated because the entity-grounding gate reads ``result.evidence`` directly.
    """

    contains_evidence = False
    summary = "retrieved chunks"

    def __init__(self, texts):
        self.evidence = [_Ev(t) for t in texts]


class _QueueRegistry:
    """Returns queued results per ``run`` call and records the calls (incl. top_k)."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def run(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self._results.pop(0) if self._results else _EvResult([])


# --------------------------------------------------------------------------- #
# hop_answer_ungrounded (pure function)
# --------------------------------------------------------------------------- #
class TestHopAnswerUngrounded(unittest.TestCase):
    def test_fires_when_every_content_token_absent(self):
        # Full parametric leak: the named entity appears nowhere in the evidence.
        self.assertTrue(
            hop_answer_ungrounded("Eric Carmen", ["The song was performed by Tom Jones."])
        )

    def test_grounded_when_a_content_token_is_present(self):
        # Sharing even one content token keeps the answer (paraphrase-tolerant).
        self.assertFalse(
            hop_answer_ungrounded("Fulton County", ["Atlanta is the seat of Fulton County."])
        )

    def test_partial_overlap_is_grounded(self):
        # "county" overlaps, so a wrong-but-plausible entity is not caught here --
        # that is the reranker's job, documented as out of scope for this gate.
        self.assertFalse(
            hop_answer_ungrounded("Montmorency County", ["Atlanta is in Fulton County."])
        )

    def test_unknown_answer_is_never_gated(self):
        self.assertFalse(hop_answer_ungrounded("UNKNOWN", ["irrelevant text"]))

    def test_empty_answer_is_never_gated(self):
        self.assertFalse(hop_answer_ungrounded("", ["irrelevant text"]))
        self.assertFalse(hop_answer_ungrounded("   ", ["irrelevant text"]))

    def test_numeric_or_stopword_only_answer_is_never_gated(self):
        # No content tokens survive stopword/number removal -> numeric gate owns it.
        self.assertFalse(hop_answer_ungrounded("1981", ["the year was 1975"]))
        self.assertFalse(hop_answer_ungrounded("the", ["a b c"]))

    def test_no_evidence_is_never_gated(self):
        # A retrieval fault (no evidence at all) is not a parametric leak.
        self.assertFalse(hop_answer_ungrounded("Eric Carmen", []))
        self.assertFalse(hop_answer_ungrounded("Eric Carmen", ["", "   "]))


# --------------------------------------------------------------------------- #
# RetrieveTool per-call top_k pass-through
# --------------------------------------------------------------------------- #
class _RecordingRetriever:
    def __init__(self):
        self.top_ks = []

    def retrieve(self, query, return_metadata=True, top_k=None):
        self.top_ks.append(top_k)
        return ["chunk a", "chunk b"]


class TestRetrieveToolTopK(unittest.TestCase):
    def test_per_call_top_k_overrides_default(self):
        from agrag.modules.agentic.tools.retrieve_tools import RetrieveTool

        retriever = _RecordingRetriever()
        tool = RetrieveTool(retriever, top_k=5)
        tool.run("q", top_k=25)
        self.assertEqual(retriever.top_ks, [25])

    def test_falls_back_to_tool_default_top_k(self):
        from agrag.modules.agentic.tools.retrieve_tools import RetrieveTool

        retriever = _RecordingRetriever()
        tool = RetrieveTool(retriever, top_k=5)
        tool.run("q")  # no per-call override
        self.assertEqual(retriever.top_ks, [5])


# --------------------------------------------------------------------------- #
# Executor deep-hop top_k selection
# --------------------------------------------------------------------------- #
class TestDeepHopTopKSelection(unittest.TestCase):
    def _executor(self, registry, **kwargs):
        ex = AgentExecutor(
            tool_registry=registry,
            policy=None,
            planner=None,
            synthesizer=AnswerSynthesizer(RecordingGenerator()),
            **kwargs,
        )
        return ex

    def _retrieve_top_ks(self, registry):
        return [c[1].get("top_k") for c in registry.calls if c[0] == "RetrieveTool"]

    def test_dependency_bearing_hop_gets_deep_top_k(self):
        # Threshold pushed out of the way to isolate the ``deps.prereqs`` path.
        registry = _QueueRegistry([_EvResult(["e1"]), _EvResult(["e2"])])
        ex = self._executor(registry, deep_hop_top_k=50, deep_hop_threshold=99)
        answers = iter(["Nirvana", "Kurt Cobain"])
        ex._extract_hop_answer = lambda q, store: next(answers)

        state = AgentState(original_query="who founded the band?")
        state.subqueries = ["what band", "who founded #1"]
        ex._run_hops(state, EvidenceStore())

        # Hop 0 is independent -> base top_k (None); hop 1 depends on #1 -> deep.
        self.assertEqual(self._retrieve_top_ks(registry), [None, 50])

    def test_threshold_index_gets_deep_top_k(self):
        registry = _QueueRegistry([_EvResult(["e1"]), _EvResult(["e2"]), _EvResult(["e3"])])
        ex = self._executor(registry, deep_hop_top_k=40, deep_hop_threshold=2)
        answers = iter(["A", "B", "C"])
        ex._extract_hop_answer = lambda q, store: next(answers)

        state = AgentState(original_query="q?")
        state.subqueries = ["hop zero", "hop one", "hop two"]  # all self-contained
        ex._run_hops(state, EvidenceStore())

        # Only the index-2 hop crosses the depth threshold.
        self.assertEqual(self._retrieve_top_ks(registry), [None, None, 40])

    def test_disabled_when_deep_hop_top_k_none(self):
        registry = _QueueRegistry([_EvResult(["e1"]), _EvResult(["e2"])])
        ex = self._executor(registry, deep_hop_top_k=None, deep_hop_threshold=1)
        answers = iter(["A", "B"])
        ex._extract_hop_answer = lambda q, store: next(answers)

        state = AgentState(original_query="q?")
        state.subqueries = ["hop zero", "hop one"]
        ex._run_hops(state, EvidenceStore())

        # No widening configured -> every hop uses the base top_k.
        self.assertEqual(self._retrieve_top_ks(registry), [None, None])


# --------------------------------------------------------------------------- #
# Entity-grounding gate wired into _run_hops
# --------------------------------------------------------------------------- #
class TestEntityGroundingGate(unittest.TestCase):
    def _executor(self, registry, **kwargs):
        return AgentExecutor(
            tool_registry=registry,
            policy=None,
            planner=None,
            synthesizer=AnswerSynthesizer(RecordingGenerator()),
            **kwargs,
        )

    def test_ungrounded_hop_downgraded_to_unknown_without_recovery(self):
        # Evidence never mentions the extracted answer -> gate fires -> UNKNOWN.
        registry = _QueueRegistry([_EvResult(["The song was performed by Tom Jones."])])
        ex = self._executor(registry, use_entity_grounding=True, use_hop_recovery=False)
        ex._extract_hop_answer = lambda q, store: "Eric Carmen"

        state = AgentState(original_query="who performed the song?")
        state.subqueries = ["who performed the song"]
        ex._run_hops(state, EvidenceStore())

        self.assertTrue(ex._is_unknown_hop_answer(state.hop_answers[0]["answer"]))
        self.assertFalse(state.hop_answers[0]["recovered"])
        # The parametric leak must not survive in the stored answer.
        self.assertNotIn("Eric Carmen", state.hop_answers[0]["answer"])

    def test_grounded_hop_passes_gate_unchanged(self):
        registry = _QueueRegistry([_EvResult(["Atlanta is the seat of Fulton County."])])
        ex = self._executor(registry, use_entity_grounding=True, use_hop_recovery=False)
        ex._extract_hop_answer = lambda q, store: "Fulton County"

        state = AgentState(original_query="what county is Atlanta in?")
        state.subqueries = ["what county is Atlanta in"]
        ex._run_hops(state, EvidenceStore())

        self.assertEqual(state.hop_answers[0]["answer"], "Fulton County")

    def test_gate_off_keeps_ungrounded_answer(self):
        # Same setup as the downgrade test but with the gate off: unchanged.
        registry = _QueueRegistry([_EvResult(["The song was performed by Tom Jones."])])
        ex = self._executor(registry, use_entity_grounding=False, use_hop_recovery=False)
        ex._extract_hop_answer = lambda q, store: "Eric Carmen"

        state = AgentState(original_query="who performed the song?")
        state.subqueries = ["who performed the song"]
        ex._run_hops(state, EvidenceStore())

        self.assertEqual(state.hop_answers[0]["answer"], "Eric Carmen")
        # No recovery retrieval either.
        retrieval_calls = [c for c in registry.calls if c[0] == "RetrieveTool"]
        self.assertEqual(len(retrieval_calls), 1)

    def test_ungrounded_hop_recovered_via_reretrieval(self):
        # First retrieval's evidence lacks the answer (gate fires); the candidate-
        # seeded second retrieval surfaces evidence that grounds the re-extraction.
        registry = _QueueRegistry(
            [
                _EvResult(["The song was performed by Tom Jones."]),
                _EvResult(["Hungry Eyes was performed by Eric Carmen."]),
            ]
        )
        ex = self._executor(registry, use_entity_grounding=True, use_hop_recovery=True)
        extractions = iter(["Eric Carmen", "Eric Carmen"])
        ex._extract_hop_answer = lambda q, store: next(extractions)
        ex._hypothesize_candidate = lambda q, state, store: "Eric Carmen"

        state = AgentState(original_query="who performed the song?")
        state.subqueries = ["who performed the song"]
        ex._run_hops(state, EvidenceStore())

        self.assertEqual(state.hop_answers[0]["answer"], "Eric Carmen")
        self.assertTrue(state.hop_answers[0]["recovered"])
        self.assertEqual(state.hop_recovery_attempts, 1)
        retrieval_calls = [c for c in registry.calls if c[0] == "RetrieveTool"]
        self.assertEqual(len(retrieval_calls), 2)


# --------------------------------------------------------------------------- #
# Synthesizer drop_query_prefix
# --------------------------------------------------------------------------- #
class TestSynthesizerDropPrefix(unittest.TestCase):
    def test_prefix_applied_by_default(self):
        gen = RecordingGenerator()
        synth = AnswerSynthesizer(gen, query_prefix="ANSWER IN ONE WORD.")
        synth.generate("Where was X born?", EvidenceStore())
        self.assertIn("ANSWER IN ONE WORD.", gen.prompts[0])

    def test_prefix_dropped_when_requested(self):
        gen = RecordingGenerator()
        synth = AnswerSynthesizer(gen, query_prefix="ANSWER IN ONE WORD.")
        synth.generate("Where was X born?", EvidenceStore(), drop_query_prefix=True)
        self.assertNotIn("ANSWER IN ONE WORD.", gen.prompts[0])

    def test_drop_flag_noop_when_no_prefix_configured(self):
        gen = RecordingGenerator()
        synth = AnswerSynthesizer(gen, query_prefix="")
        synth.generate("Where was X born?", EvidenceStore(), drop_query_prefix=True)
        self.assertEqual(len(gen.prompts), 1)


# --------------------------------------------------------------------------- #
# Config plumbing: agentic_module -> executor attributes
# --------------------------------------------------------------------------- #
class TestConfigPlumbing(unittest.TestCase):
    def _module(self, cfg):
        from agrag.modules.agentic.agentic_module import AgenticRAGModule

        class _FakeGen:
            model_name = "mistral-7b"

        return AgenticRAGModule(retriever_module=object(), generator_module=_FakeGen(), config=cfg)

    def test_new_knobs_default_off(self):
        mod = self._module({})
        self.assertIsNone(mod.executor.deep_hop_top_k)
        self.assertEqual(mod.executor.deep_hop_threshold, 2)
        self.assertFalse(mod.executor.use_entity_grounding)
        self.assertFalse(mod.executor.final_answer_drop_prefix)

    def test_new_knobs_flow_to_executor(self):
        mod = self._module(
            {
                "deep_hop_top_k": 40,
                "deep_hop_threshold": 3,
                "use_entity_grounding": True,
                "final_answer_drop_prefix": True,
            }
        )
        self.assertEqual(mod.executor.deep_hop_top_k, 40)
        self.assertEqual(mod.executor.deep_hop_threshold, 3)
        self.assertTrue(mod.executor.use_entity_grounding)
        self.assertTrue(mod.executor.final_answer_drop_prefix)


if __name__ == "__main__":
    unittest.main()

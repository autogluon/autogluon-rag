import unittest

from agrag.modules.agentic.evidence import Evidence, EvidenceStore
from agrag.modules.agentic.state import AgentState, AgentStatus
from agrag.modules.agentic.trace import AgentTrace


class TestAgentTrace(unittest.TestCase):
    def _build_run(self):
        state = AgentState(original_query="what is autogluon?")
        state.plan = ["what is autogluon?"]
        state.iteration = 1
        state.record_action("retrieve", tool_name="RetrieveTool", evidence_added=2)
        state.record_action("multi_retrieve", tool_name="MultiQueryRetrieveTool", evidence_added=1)
        state.record_action("draft_answer", tool_name=None)
        state.verification = {"label": "supported"}
        state.finish(AgentStatus.ANSWERED)

        store = EvidenceStore()
        store.add(Evidence(text="a", doc_id=0, chunk_id=0))
        store.add(Evidence(text="b", doc_id=0, chunk_id=1))
        store.add(Evidence(text="c", doc_id=1, chunk_id=0))
        store.mark_used(["e0", "e2"])
        return state, store

    def test_from_run_computes_metrics(self):
        state, store = self._build_run()
        trace = AgentTrace.from_run(state, store, final_answer="AutoGluon is an AutoML library.")
        m = trace.metrics
        self.assertEqual(m["iterations"], 1)
        self.assertEqual(m["tool_calls"], 2)  # draft_answer has no tool
        self.assertEqual(m["retrieval_calls"], 2)  # both retrieve tools
        self.assertEqual(m["evidence_count"], 3)
        self.assertEqual(m["cited_evidence_count"], 2)

    def test_from_run_carries_answer_status_and_verification(self):
        state, store = self._build_run()
        trace = AgentTrace.from_run(state, store, final_answer="ans")
        self.assertEqual(trace.final_answer, "ans")
        self.assertEqual(trace.status, "answered")
        self.assertEqual(trace.verification, {"label": "supported"})
        self.assertEqual(len(trace.steps), 3)
        self.assertEqual(len(trace.evidence), 3)

    def test_extra_metrics_merged(self):
        state, store = self._build_run()
        trace = AgentTrace.from_run(state, store, final_answer="ans", extra_metrics={"latency_s": 1.23, "tokens": 456})
        self.assertEqual(trace.metrics["latency_s"], 1.23)
        self.assertEqual(trace.metrics["tokens"], 456)
        # built-in metrics still present
        self.assertEqual(trace.metrics["evidence_count"], 3)

    def test_to_dict_is_serializable(self):
        state, store = self._build_run()
        trace = AgentTrace.from_run(state, store, final_answer="ans")
        data = trace.to_dict()
        self.assertEqual(data["original_query"], "what is autogluon?")
        self.assertEqual(data["status"], "answered")
        self.assertIn("metrics", data)
        self.assertEqual(data["evidence"][0]["evidence_id"], "e0")

    def test_abstain_run_has_none_answer(self):
        state = AgentState(original_query="q")
        state.finish(AgentStatus.ABSTAINED)
        store = EvidenceStore()
        trace = AgentTrace.from_run(state, store, final_answer=None)
        self.assertIsNone(trace.final_answer)
        self.assertEqual(trace.status, "abstained")
        self.assertEqual(trace.metrics["evidence_count"], 0)


if __name__ == "__main__":
    unittest.main()

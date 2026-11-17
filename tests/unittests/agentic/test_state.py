import unittest

from agrag.modules.agentic.state import ActionRecord, AgentState, AgentStatus


class TestAgentState(unittest.TestCase):
    def test_current_query_defaults_to_original(self):
        state = AgentState(original_query="what is autogluon?")
        self.assertEqual(state.current_query, "what is autogluon?")
        self.assertEqual(state.status, AgentStatus.IN_PROGRESS)
        self.assertFalse(state.is_terminal)

    def test_record_action_appends_history_with_current_iteration(self):
        state = AgentState(original_query="q")
        state.iteration = 2
        record = state.record_action(
            action_type="retrieve",
            tool_name="RetrieveTool",
            args={"query": "q"},
            observation_summary="3 chunks",
            evidence_added=3,
        )
        self.assertIsInstance(record, ActionRecord)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.history[0].iteration, 2)
        self.assertEqual(state.history[0].evidence_added, 3)

    def test_tool_call_count_ignores_non_tool_actions(self):
        state = AgentState(original_query="q")
        state.record_action("retrieve", tool_name="RetrieveTool")
        state.record_action("draft_answer", tool_name=None)
        state.record_action("retrieve", tool_name="MultiQueryRetrieveTool")
        self.assertEqual(state.tool_call_count, 2)

    def test_add_evidence_ids_dedups_and_counts(self):
        state = AgentState(original_query="q")
        state.add_evidence_ids(["e0", "e1"])
        state.add_evidence_ids(["e1", "e2"])
        self.assertEqual(state.evidence_ids, ["e0", "e1", "e2"])
        self.assertEqual(state.evidence_count, 3)

    def test_set_current_query_and_original_unchanged(self):
        state = AgentState(original_query="orig")
        state.set_current_query("rewritten")
        self.assertEqual(state.current_query, "rewritten")
        self.assertEqual(state.original_query, "orig")

    def test_finish_sets_terminal_status(self):
        state = AgentState(original_query="q")
        state.finish(AgentStatus.ABSTAINED)
        self.assertTrue(state.is_terminal)
        self.assertEqual(state.status, AgentStatus.ABSTAINED)

    def test_to_dict_serializes_status_as_value(self):
        state = AgentState(original_query="q")
        state.plan = ["q1", "q2"]
        state.record_action("retrieve", tool_name="RetrieveTool")
        state.finish(AgentStatus.ANSWERED)
        data = state.to_dict()
        self.assertEqual(data["status"], "answered")
        self.assertEqual(data["plan"], ["q1", "q2"])
        self.assertEqual(len(data["history"]), 1)
        self.assertEqual(data["history"][0]["tool_name"], "RetrieveTool")


if __name__ == "__main__":
    unittest.main()

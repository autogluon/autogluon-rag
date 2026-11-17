import os
import unittest
from unittest.mock import MagicMock

from agrag.agrag import PRESETS_CONFIG_DIRECTORY, AutoGluonRAG

PRESET_CONFIG = os.path.join(PRESETS_CONFIG_DIRECTORY, "medium_quality_config.yaml")


class TestGenerateResponseRouting(unittest.TestCase):
    def _build_agrag(self):
        # Pass the preset config file explicitly so the constructor does not fall
        # back to argparse (which would try to parse pytest's argv). data_dir
        # satisfies the constructor's requirement without initializing anything.
        agrag = AutoGluonRAG(config_file=PRESET_CONFIG, data_dir="/tmp/does_not_matter")

        # Inject lightweight fakes for the modules generate_response depends on.
        agrag.retriever_module = MagicMock()
        agrag.retriever_module.top_k = 5
        agrag.retriever_module.retrieve.return_value = ["chunk 1", "chunk 2"]

        agrag.generator_module = MagicMock()
        agrag.generator_module.model_name = "mistral-7b"
        agrag.generator_module.generate_response.return_value = "standard answer"

        # Data processing module must NEVER be touched at query time.
        agrag.data_processing_module = MagicMock()
        return agrag

    def test_default_uses_standard_path(self):
        agrag = self._build_agrag()
        response = agrag.generate_response("what is autogluon")
        self.assertEqual(response, "standard answer")
        # Standard path retrieves exactly once, via the plain (text) retrieve.
        agrag.retriever_module.retrieve.assert_called_once()
        # Agentic module was never constructed.
        self.assertIsNone(agrag.agentic_module)
        # No data processing at query time (design invariant).
        agrag.data_processing_module.process_data.assert_not_called()

    def test_standard_path_skips_retrieval_when_top_k_zero(self):
        agrag = self._build_agrag()
        agrag.retriever_module.top_k = 0
        agrag.generate_response("q")
        agrag.retriever_module.retrieve.assert_not_called()

    def test_agentic_mode_routes_to_agentic_module(self):
        agrag = self._build_agrag()
        # Structured retrieval for the agentic path.
        agrag.retriever_module.retrieve.return_value = [
            {"text": "a", "doc_id": 0, "chunk_id": 0, "rank": 0},
            {"text": "b", "doc_id": 0, "chunk_id": 1, "rank": 1},
        ]
        agrag.generator_module.generate_response.return_value = "supported"

        response = agrag.generate_response("q", mode="agentic")
        self.assertIsInstance(response, str)
        # Agentic module was lazily built.
        self.assertIsNotNone(agrag.agentic_module)
        agrag.data_processing_module.process_data.assert_not_called()

    def test_agentic_return_trace(self):
        agrag = self._build_agrag()
        agrag.retriever_module.retrieve.return_value = [
            {"text": "a", "doc_id": 0, "chunk_id": 0, "rank": 0},
            {"text": "b", "doc_id": 0, "chunk_id": 1, "rank": 1},
        ]
        agrag.generator_module.generate_response.return_value = "supported"

        result = agrag.generate_response("q", mode="agentic", return_trace=True)
        self.assertIsInstance(result, tuple)
        answer, trace = result
        self.assertIsInstance(trace, dict)
        self.assertIn("metrics", trace)

    def test_standard_return_trace_single_retrieval_and_evidence_consistency(self):
        agrag = self._build_agrag()
        # Structured records the standard trace must carry verbatim as evidence.
        records = [
            {"text": "alpha", "doc_id": 0, "chunk_id": 0, "rank": 0, "retrieval_score": 0.9},
            {"text": "beta", "doc_id": 1, "chunk_id": 0, "rank": 1, "retrieval_score": 0.8},
        ]
        agrag.retriever_module.retrieve.return_value = records
        agrag.generator_module.generate_response.return_value = "standard answer"

        result = agrag.generate_response("what is autogluon", mode="standard", return_trace=True)

        # Returns (answer, trace); default (no return_trace) still returns a str.
        self.assertIsInstance(result, tuple)
        answer, trace = result
        self.assertEqual(answer, "standard answer")
        # Exactly ONE retrieval for the whole standard run.
        agrag.retriever_module.retrieve.assert_called_once()
        # Trace mirrors the agentic schema and carries the exact evidence used.
        self.assertEqual(trace["mode"], "standard")
        self.assertEqual(trace["final_answer"], "standard answer")
        self.assertEqual([e["text"] for e in trace["evidence"]], ["alpha", "beta"])
        self.assertEqual(trace["evidence"][0]["retrieval_score"], 0.9)
        self.assertEqual(trace["metrics"]["retrieval_calls"], 1)
        self.assertEqual(trace["metrics"]["evidence_count"], 2)
        # original_query is the raw query, not the prefixed generator query.
        self.assertEqual(trace["original_query"], "what is autogluon")

    def test_standard_default_call_returns_string_unchanged(self):
        # Backward compatibility: the plain call still returns only a string and
        # uses the text-only retrieval path exactly once.
        agrag = self._build_agrag()
        out = agrag.generate_response("q")
        self.assertIsInstance(out, str)
        agrag.retriever_module.retrieve.assert_called_once()

    def test_standard_return_trace_handles_no_retrieval(self):
        agrag = self._build_agrag()
        agrag.retriever_module.top_k = 0
        answer, trace = agrag.generate_response("q", mode="standard", return_trace=True)
        agrag.retriever_module.retrieve.assert_not_called()
        self.assertEqual(trace["evidence"], [])
        self.assertEqual(trace["metrics"]["retrieval_calls"], 0)

    def test_resolve_mode_precedence(self):
        agrag = self._build_agrag()
        # Explicit arg wins.
        self.assertEqual(agrag._resolve_mode("agentic"), "agentic")
        # Default when nothing set.
        self.assertEqual(agrag._resolve_mode(None), "standard")
        # agent.enabled flips default to agentic.
        agrag.args.config.setdefault("agent", {})["enabled"] = True
        self.assertEqual(agrag._resolve_mode(None), "agentic")


if __name__ == "__main__":
    unittest.main()

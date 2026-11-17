import unittest

from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.tools import (
    ContextCompressionTool,
    MultiQueryRetrieveTool,
    QueryRewriteTool,
    RetrieveTool,
    Tool,
    ToolRegistry,
)


class FakeRetriever:
    """Returns canned structured records per query; records calls."""

    def __init__(self, responses, reranker=None):
        self.responses = responses
        self.calls = []
        self.reranker = reranker

    def retrieve(self, query, return_metadata=False, top_k=None):
        self.calls.append((query, return_metadata, top_k))
        return self.responses.get(query)


class FakeGenerator:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        return self.response


class TestRetrieveTool(unittest.TestCase):
    def test_produces_evidence_from_records(self):
        retriever = FakeRetriever(
            {
                "q": [
                    {"text": "chunk a", "doc_id": 0, "chunk_id": 0, "rank": 0, "retrieval_score": 0.9},
                    {"text": "chunk b", "doc_id": 0, "chunk_id": 1, "rank": 1, "retrieval_score": 0.8},
                ]
            }
        )
        tool = RetrieveTool(retriever)
        result = tool.run(query="q")
        self.assertTrue(result.contains_evidence)
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual(result.evidence[0].doc_id, 0)
        self.assertEqual(result.evidence[0].tool_name, "RetrieveTool")
        # Called the retriever with metadata enabled and default top_k.
        self.assertEqual(retriever.calls, [("q", True, None)])

    def test_passes_top_k_through(self):
        retriever = FakeRetriever({"q": [{"text": "a", "doc_id": 0, "chunk_id": 0}]})
        RetrieveTool(retriever, top_k=8).run(query="q")
        self.assertEqual(retriever.calls, [("q", True, 8)])

    def test_handles_none_result(self):
        tool = RetrieveTool(FakeRetriever({"q": None}))
        result = tool.run(query="q")
        self.assertFalse(result.contains_evidence)
        self.assertEqual(len(result.evidence), 0)

    def test_handles_text_only_records(self):
        tool = RetrieveTool(FakeRetriever({"q": ["plain chunk"]}))
        result = tool.run(query="q")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].text, "plain chunk")


class TestMultiQueryRetrieveTool(unittest.TestCase):
    def test_merges_across_subqueries(self):
        retriever = FakeRetriever(
            {
                "q1": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
                "q2": [{"text": "b", "doc_id": 0, "chunk_id": 1}],
            }
        )
        tool = MultiQueryRetrieveTool(retriever)
        result = tool.run(queries=["q1", "q2"])
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual({e.retrieval_query for e in result.evidence}, {"q1", "q2"})

    def test_dedup_left_to_store(self):
        # Same chunk returned by both subqueries; the store dedups it.
        retriever = FakeRetriever(
            {
                "q1": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
                "q2": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
            }
        )
        result = MultiQueryRetrieveTool(retriever).run(queries=["q1", "q2"])
        self.assertEqual(len(result.evidence), 2)  # tool returns both
        store = EvidenceStore()
        stored = store.add_many(result.evidence)
        self.assertEqual(stored, 1)  # store keeps one


class FakeFusedRetriever:
    """Retriever exposing ``retrieve_fused``; records how it was called.

    Fusion/rerank/MMR correctness lives in the retriever's own unit tests
    (test_hybrid_retriever.py). Here we only assert the tool delegates to
    ``retrieve_fused`` once and converts the returned records into Evidence,
    preserving provenance — the tool itself does no fusion or reranking.
    """

    def __init__(self, fused_records):
        self.fused_records = fused_records
        self.fused_calls = []
        self.retrieve_calls = []

    def retrieve_fused(self, subqueries, original_query=None, return_metadata=False, top_k=None, rrf_k=None):
        self.fused_calls.append(
            {
                "subqueries": subqueries,
                "original_query": original_query,
                "return_metadata": return_metadata,
                "top_k": top_k,
                "rrf_k": rrf_k,
            }
        )
        return list(self.fused_records) if self.fused_records is not None else None

    def retrieve(self, query, return_metadata=False, top_k=None):
        self.retrieve_calls.append((query, return_metadata, top_k))
        return None


class TestMultiQueryRetrieveToolFused(unittest.TestCase):
    def test_fused_mode_delegates_once_to_retriever(self):
        retriever = FakeFusedRetriever([{"text": "a", "doc_id": 0, "chunk_id": 0, "retrieval_queries": ["q1", "q2"]}])
        tool = MultiQueryRetrieveTool(retriever, top_k=7, use_fused_retrieval=True, rrf_k=42)
        tool.run(queries=["q1", "q2"], original_query="user question")

        # Exactly one global fused call; no per-subquery retrieve() calls (which
        # would be the double-processing / concatenation path).
        self.assertEqual(len(retriever.fused_calls), 1)
        self.assertEqual(retriever.retrieve_calls, [])
        call = retriever.fused_calls[0]
        self.assertEqual(call["subqueries"], ["q1", "q2"])
        self.assertEqual(call["original_query"], "user question")
        self.assertEqual(call["top_k"], 7)
        self.assertEqual(call["rrf_k"], 42)
        self.assertTrue(call["return_metadata"])

    def test_fused_mode_preserves_provenance_from_records(self):
        retriever = FakeFusedRetriever(
            [
                {
                    "text": "a",
                    "doc_id": 0,
                    "chunk_id": 0,
                    "rrf_score": 0.5,
                    "fusion_rank": 0,
                    "rerank_score": 9.0,
                    "retrieval_queries": ["q1", "q2"],
                    "source_signals": ["dense", "sparse"],
                },
                {"text": "b", "doc_id": 0, "chunk_id": 1, "retrieval_queries": ["q2"]},
            ]
        )
        result = MultiQueryRetrieveTool(retriever, use_fused_retrieval=True).run(queries=["q1", "q2"])
        by_text = {e.text: e for e in result.evidence}
        # All retrieval_queries survive onto the Evidence; representative query is
        # the first of the merged provenance.
        self.assertEqual(by_text["a"].retrieval_queries, ["q1", "q2"])
        self.assertEqual(by_text["a"].retrieval_query, "q1")
        self.assertEqual(by_text["b"].retrieval_queries, ["q2"])
        # source_signals is carried through as extra metadata.
        self.assertEqual(by_text["a"].metadata.get("source_signals"), ["dense", "sparse"])
        # Fusion + rerank scores threaded onto Evidence.
        self.assertEqual(by_text["a"].fusion_rank, 0)
        self.assertEqual(by_text["a"].rrf_score, 0.5)
        self.assertEqual(by_text["a"].rerank_score, 9.0)
        # Tool-assigned rank follows the retriever's global order.
        self.assertEqual([e.rank for e in result.evidence], [0, 1])

    def test_fused_mode_empty_results(self):
        retriever = FakeFusedRetriever(None)
        result = MultiQueryRetrieveTool(retriever, use_fused_retrieval=True).run(queries=["q1", "q2"])
        self.assertFalse(result.contains_evidence)
        self.assertEqual(len(result.evidence), 0)

    def test_fused_mode_falls_back_when_retriever_lacks_method(self):
        # An older retriever without retrieve_fused must not crash; the tool falls
        # back to plain concatenation.
        retriever = FakeRetriever(
            {
                "q1": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
                "q2": [{"text": "b", "doc_id": 0, "chunk_id": 1}],
            }
        )
        result = MultiQueryRetrieveTool(retriever, use_fused_retrieval=True).run(queries=["q1", "q2"])
        self.assertEqual(len(result.evidence), 2)


class TestLLMTools(unittest.TestCase):
    def test_query_rewrite(self):
        gen = FakeGenerator("  better query  ")
        result = QueryRewriteTool(gen).run(query="orig query")
        self.assertEqual(result.output, "better query")
        self.assertFalse(result.contains_evidence)
        self.assertIn("orig query", gen.prompts[0])

    def test_context_compression(self):
        gen = FakeGenerator("short summary")
        result = ContextCompressionTool(gen).run(query="q", texts=["c1", "c2"])
        self.assertEqual(result.output, "short summary")
        self.assertIn("c1", gen.prompts[0])
        self.assertIn("c2", gen.prompts[0])


class TestToolRegistry(unittest.TestCase):
    def test_register_run_and_names(self):
        retriever = FakeRetriever({"q": [{"text": "a", "doc_id": 0, "chunk_id": 0}]})
        registry = ToolRegistry([RetrieveTool(retriever)])
        self.assertTrue(registry.has("RetrieveTool"))
        self.assertEqual(registry.names(), ["RetrieveTool"])
        result = registry.run("RetrieveTool", query="q")
        self.assertEqual(len(result.evidence), 1)

    def test_unknown_tool_raises(self):
        registry = ToolRegistry([])
        with self.assertRaises(KeyError):
            registry.run("NopeTool", query="q")

    def test_duplicate_registration_raises(self):
        retriever = FakeRetriever({})
        registry = ToolRegistry([RetrieveTool(retriever)])
        with self.assertRaises(ValueError):
            registry.register(RetrieveTool(retriever))

    def test_nameless_tool_rejected(self):
        class Nameless(Tool):
            name = ""

        with self.assertRaises(ValueError):
            ToolRegistry([Nameless()])


if __name__ == "__main__":
    unittest.main()

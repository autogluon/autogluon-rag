import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, PARENT_ID_KEY
from agrag.modules.retriever.retrievers.retriever_base import RetrieverModule
from agrag.modules.vector_db.vector_database import VectorDatabaseModule


class FakeSparseRetriever:
    """Minimal stand-in for BM25Retriever returning canned (row_index, score)."""

    def __init__(self, hits):
        self._hits = hits
        self._built = True

    def build(self, documents):  # pragma: no cover - already "built"
        return self

    def search(self, query, top_k):
        return self._hits[:top_k]


def _metadata():
    return pd.DataFrame(
        [
            {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "alpha chunk"},
            {DOC_ID_KEY: 0, CHUNK_ID_KEY: 1, DOC_TEXT_KEY: "beta chunk"},
            {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "gamma chunk"},
        ]
    )


def _build_retriever(**kwargs):
    vector_db = MagicMock(VectorDatabaseModule)
    vector_db.metadata = _metadata()
    retriever = RetrieverModule(
        vector_database_module=vector_db,
        embedding_module=MagicMock(),
        top_k=5,
        use_reranker=False,
        **kwargs,
    )
    # Deterministic query embedding; no real model.
    retriever.encode_query = lambda query: np.zeros(4)
    return retriever, vector_db


class TestLegacyPathUnchanged(unittest.TestCase):
    def test_all_flags_off_uses_dense_only_path(self):
        retriever, vector_db = _build_retriever()
        self.assertFalse(retriever._uses_advanced_path)
        vector_db.search_vector_database.return_value = [0, 1]
        result = retriever.retrieve("q")
        # Legacy text-only path returns chunk texts in DB order.
        self.assertEqual(result, ["alpha chunk", "beta chunk"])


class TestHybridFusion(unittest.TestCase):
    def test_hybrid_fuses_dense_and_sparse(self):
        sparse = FakeSparseRetriever([(2, 3.0), (0, 1.0)])
        retriever, vector_db = _build_retriever(sparse_retriever=sparse, use_hybrid=True, use_rrf=True)
        self.assertTrue(retriever._uses_advanced_path)
        # Dense surfaces rows 0,1; sparse surfaces rows 2,0. Row 0 is shared.
        vector_db.search_vector_database.return_value = ([0, 1], [0.9, 0.8])

        records = retriever.retrieve("q", return_metadata=True)
        texts = [r["text"] for r in records]
        # Row 0 appears in both lists, so it fuses to the top.
        self.assertEqual(texts[0], "alpha chunk")
        # Every fused row is present exactly once (deduped).
        self.assertEqual(set(texts), {"alpha chunk", "beta chunk", "gamma chunk"})
        # Shared chunk keeps both signals as provenance.
        self.assertEqual(records[0]["source_signals"], ["dense", "sparse"])
        # rrf_score threaded through, ranks reassigned in fused order.
        self.assertIn("rrf_score", records[0])
        self.assertEqual([r["rank"] for r in records], [0, 1, 2])

    def test_rerank_keys_on_identity_not_text(self):
        # Two different chunks share identical text; the global rerank must not
        # collapse them into one record.
        vector_db = MagicMock(VectorDatabaseModule)
        vector_db.metadata = pd.DataFrame(
            [
                {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "same text"},
                {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "same text"},
            ]
        )
        reranker = MagicMock()
        reranker.rerank.return_value = [("same text", 5.0), ("same text", 4.0)]
        retriever = RetrieverModule(
            vector_database_module=vector_db,
            embedding_module=MagicMock(),
            top_k=5,
            use_reranker=False,
            use_rrf=True,
        )
        # Attach the mock reranker directly (the isinstance guard only runs when
        # use_reranker=True); the advanced path uses whatever ``reranker`` is set.
        retriever.reranker = reranker
        retriever.encode_query = lambda query: np.zeros(4)
        vector_db.search_vector_database.return_value = ([0, 1], [0.9, 0.8])

        records = retriever.retrieve("q", return_metadata=True)
        self.assertEqual(len(records), 2)
        # Both distinct (doc_id, chunk_id) identities survive.
        self.assertEqual({(r[DOC_ID_KEY], r[CHUNK_ID_KEY]) for r in records}, {(0, 0), (1, 0)})


class RecordingReranker:
    """Reranker that records how many times it was called and against which query.

    Sorts by a per-text score table (default: reverse of input order) so tests
    can assert on the resulting global order.
    """

    def __init__(self, score_by_text=None):
        self.calls = []
        self.score_by_text = score_by_text

    def rerank(self, query, texts, return_scores=False):
        self.calls.append({"query": query, "texts": list(texts)})
        if self.score_by_text is not None:
            scored = [(t, float(self.score_by_text.get(t, 0.0))) for t in texts]
        else:
            # Reverse of the given order, descending scores.
            reversed_texts = list(reversed(texts))
            scored = [(t, float(len(reversed_texts) - i)) for i, t in enumerate(reversed_texts)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored if return_scores else [t for t, _ in scored]


class TestFusedMultiQueryRetrieval(unittest.TestCase):
    """Global multi-query fusion pipeline (agentic MultiQueryRetrieveTool path)."""

    def _fused_retriever(self, per_query_rows, **kwargs):
        """Build a retriever whose dense search returns canned rows per query."""
        retriever, vector_db = _build_retriever(**kwargs)

        def fake_search(embedding=None, top_k=None, return_scores=False):
            key = fake_search.current
            if key not in per_query_rows and len(per_query_rows) == 1:
                # Single-subquery tests may replace encode_query (dropping the
                # routing hook); fall back to the only canned response.
                key = next(iter(per_query_rows))
            rows, scores = per_query_rows[key]
            if return_scores:
                return rows[:top_k], scores[:top_k]
            return rows[:top_k]

        fake_search.current = None
        # encode_query is stubbed; route each subquery to its canned rows via a
        # side-channel set on the search mock right before each dense call.
        original_encode = retriever.encode_query

        def encode(query):
            fake_search.current = query
            return original_encode(query)

        retriever.encode_query = encode
        vector_db.search_vector_database.side_effect = fake_search
        return retriever, vector_db

    def test_global_rrf_dedup_and_provenance(self):
        # q1 -> rows [0,1]; q2 -> rows [0,2]. Row 0 shared across both subqueries.
        retriever, _ = self._fused_retriever(
            {"q1": ([0, 1], [0.9, 0.8]), "q2": ([0, 2], [0.7, 0.6])},
            use_rrf=True,
        )
        records = retriever.retrieve_fused(["q1", "q2"], original_query="user q", return_metadata=True)

        texts = [r["text"] for r in records]
        # One record per distinct chunk (not 4 concatenated).
        self.assertEqual(len(records), 3)
        # Shared row 0 fuses to the top and keeps BOTH subqueries as provenance.
        self.assertEqual(texts[0], "alpha chunk")
        self.assertEqual(records[0]["retrieval_queries"], ["q1", "q2"])
        # Single-subquery chunks keep just their own subquery.
        by_text = {r["text"]: r for r in records}
        self.assertEqual(by_text["beta chunk"]["retrieval_queries"], ["q1"])
        self.assertEqual(by_text["gamma chunk"]["retrieval_queries"], ["q2"])
        # Ranks reassigned in fused order; fusion_rank + rrf_score present.
        self.assertEqual([r["rank"] for r in records], [0, 1, 2])
        self.assertIn("rrf_score", records[0])
        self.assertEqual(records[0]["fusion_rank"], 0)

    def test_reranker_called_exactly_once_against_original_query(self):
        reranker = RecordingReranker()
        retriever, _ = self._fused_retriever(
            {"q1": ([0, 1], [0.9, 0.8]), "q2": ([0, 2], [0.7, 0.6])},
            use_rrf=True,
        )
        retriever.reranker = reranker
        retriever.retrieve_fused(["q1", "q2"], original_query="the user question", return_metadata=True)

        # Exactly one global rerank call (not one per subquery).
        self.assertEqual(len(reranker.calls), 1)
        # Reranked against the ORIGINAL user query, not a subquery.
        self.assertEqual(reranker.calls[0]["query"], "the user question")
        # The single call saw the full deduped fused pool (all 3 distinct chunks).
        self.assertEqual(sorted(reranker.calls[0]["texts"]), ["alpha chunk", "beta chunk", "gamma chunk"])

    def test_reranker_orders_result_globally(self):
        # Force a specific global order via the score table.
        reranker = RecordingReranker(score_by_text={"alpha chunk": 1.0, "beta chunk": 5.0, "gamma chunk": 3.0})
        retriever, _ = self._fused_retriever(
            {"q1": ([0, 1], [0.9, 0.8]), "q2": ([0, 2], [0.7, 0.6])},
            use_rrf=True,
        )
        retriever.reranker = reranker
        records = retriever.retrieve_fused(["q1", "q2"], original_query="user q", return_metadata=True)
        self.assertEqual([r["text"] for r in records], ["beta chunk", "gamma chunk", "alpha chunk"])
        self.assertEqual([r["rank"] for r in records], [0, 1, 2])

    def test_final_top_k_truncates_after_rerank(self):
        retriever, _ = self._fused_retriever(
            {"q1": ([0, 1], [0.9, 0.8]), "q2": ([0, 2], [0.7, 0.6])},
            use_rrf=True,
        )
        records = retriever.retrieve_fused(["q1", "q2"], original_query="user q", return_metadata=True, top_k=2)
        # 3 distinct chunks fused, truncated to the final top-k of 2.
        self.assertEqual(len(records), 2)
        self.assertEqual([r["rank"] for r in records], [0, 1])

    def test_global_mmr_reorders_once(self):
        # Two near-identical vectors + one diverse; MMR should not keep both
        # similar chunks adjacent. We assert MMR runs on the global pool by
        # checking the order changes deterministically vs. relevance-only.
        retriever, _ = self._fused_retriever(
            {"q1": ([0, 1, 2], [0.9, 0.8, 0.7])},
            use_rrf=True,
            use_mmr=True,
            mmr_lambda=0.3,
        )
        # Distinct embeddings per row so MMR has something to diversify over.
        embeds = {
            "alpha chunk": [1.0, 0.0, 0.0, 0.0],
            "beta chunk": [0.99, 0.01, 0.0, 0.0],
            "gamma chunk": [0.0, 1.0, 0.0, 0.0],
        }

        def fake_encode(data):
            texts = list(data[DOC_TEXT_KEY])
            return pd.DataFrame({"embedding": [np.array(embeds.get(t, [0, 0, 0, 0])) for t in texts]})

        retriever.embedding_module.encode = fake_encode
        retriever.encode_query = lambda q: np.array([1.0, 0.0, 0.0, 0.0])

        records = retriever.retrieve_fused(["q1"], original_query="user q", return_metadata=True)
        order = [r["text"] for r in records]
        # Most relevant first; the diverse gamma chunk is pulled up ahead of the
        # near-duplicate beta chunk.
        self.assertEqual(order[0], "alpha chunk")
        self.assertEqual(order[1], "gamma chunk")
        self.assertEqual(order[2], "beta chunk")

    def test_identical_text_different_identity_not_collapsed(self):
        vector_db = MagicMock(VectorDatabaseModule)
        vector_db.metadata = pd.DataFrame(
            [
                {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "same text"},
                {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "same text"},
            ]
        )
        retriever = RetrieverModule(
            vector_database_module=vector_db,
            embedding_module=MagicMock(),
            top_k=5,
            use_reranker=False,
            use_rrf=True,
        )
        retriever.encode_query = lambda q: np.zeros(4)
        vector_db.search_vector_database.return_value = ([0, 1], [0.9, 0.8])
        # A reranker that would collapse by text if identity were not preserved.
        retriever.reranker = RecordingReranker(score_by_text={"same text": 5.0})

        records = retriever.retrieve_fused(["q1"], original_query="user q", return_metadata=True)
        self.assertEqual(len(records), 2)
        self.assertEqual({(r[DOC_ID_KEY], r[CHUNK_ID_KEY]) for r in records}, {(0, 0), (1, 0)})

    def test_expansion_happens_after_selection(self):
        # top_k=1 keeps a single chunk; only that surviving chunk is expanded.
        retriever, _ = self._fused_retriever(
            {"q1": ([0, 1, 2], [0.9, 0.8, 0.7])},
            use_rrf=True,
            chunk_read=1,
        )
        records = retriever.retrieve_fused(["q1"], original_query="user q", return_metadata=True, top_k=1)
        self.assertEqual(len(records), 1)
        # The kept chunk carries child provenance and an expanded neighbor window.
        self.assertIn("child_text", records[0])
        self.assertIn("neighbor_chunk_ids", records[0])

    def test_returns_none_when_no_candidates(self):
        retriever, vector_db = self._fused_retriever({"q1": ([], [])}, use_rrf=True)
        self.assertIsNone(retriever.retrieve_fused(["q1"], original_query="user q", return_metadata=True))


class TestParentExpansionDedup(unittest.TestCase):
    def test_sibling_chunks_do_not_duplicate_parent_context(self):
        # Two selected children share parent p0; a third has parent p1.
        parent_store = pd.DataFrame(
            [
                {PARENT_ID_KEY: "p0", DOC_TEXT_KEY: "PARENT ZERO"},
                {PARENT_ID_KEY: "p1", DOC_TEXT_KEY: "PARENT ONE"},
            ]
        )
        vector_db = MagicMock(VectorDatabaseModule)
        vector_db.metadata = pd.DataFrame(
            [
                {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "child a", PARENT_ID_KEY: "p0"},
                {DOC_ID_KEY: 0, CHUNK_ID_KEY: 1, DOC_TEXT_KEY: "child b", PARENT_ID_KEY: "p0"},
                {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "child c", PARENT_ID_KEY: "p1"},
            ]
        )
        retriever = RetrieverModule(
            vector_database_module=vector_db,
            embedding_module=MagicMock(),
            top_k=5,
            use_reranker=False,
            parent_store=parent_store,
        )
        retriever.encode_query = lambda q: np.zeros(4)

        records = retriever._expand_context(
            [
                {
                    DOC_ID_KEY: 0,
                    CHUNK_ID_KEY: 0,
                    DOC_TEXT_KEY: "child a",
                    PARENT_ID_KEY: "p0",
                    "retrieval_queries": ["q1"],
                    "source_signals": ["dense"],
                },
                {
                    DOC_ID_KEY: 0,
                    CHUNK_ID_KEY: 1,
                    DOC_TEXT_KEY: "child b",
                    PARENT_ID_KEY: "p0",
                    "retrieval_queries": ["q2"],
                    "source_signals": ["sparse"],
                },
                {
                    DOC_ID_KEY: 1,
                    CHUNK_ID_KEY: 0,
                    DOC_TEXT_KEY: "child c",
                    PARENT_ID_KEY: "p1",
                    "retrieval_queries": ["q1"],
                    "source_signals": ["dense"],
                },
            ]
        )

        # p0 emitted once (siblings collapsed), p1 once -> two expanded records.
        self.assertEqual(len(records), 2)
        parent_zero = records[0]
        self.assertIn("PARENT ZERO", parent_zero["text"])
        # The parent text appears exactly once, not duplicated per sibling.
        self.assertEqual(parent_zero["text"].count("PARENT ZERO"), 1)
        # Both siblings' child ids and provenance are preserved on the survivor.
        self.assertEqual(parent_zero["expanded_child_chunk_ids"], [0, 1])
        self.assertEqual(parent_zero["retrieval_queries"], ["q1", "q2"])
        self.assertEqual(parent_zero["source_signals"], ["dense", "sparse"])
        self.assertEqual(parent_zero["child_text"], "child a")

    def test_records_without_parent_are_not_collapsed(self):
        retriever, vector_db = _build_retriever(chunk_read=0)
        # No parent_store, no chunk_read effect -> distinct chunks stay distinct.
        records = retriever._expand_context(
            [
                {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "x"},
                {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "x"},
            ]
        )
        self.assertEqual(len(records), 2)


class TestInvalidIndexHandling(unittest.TestCase):
    def test_dense_path_drops_negative_and_out_of_range(self):
        # Dense hits with a FAISS -1 sentinel and an out-of-range index must be
        # dropped; -1 must never resolve to the last metadata row (iloc[-1]).
        retriever, vector_db = _build_retriever(use_rrf=True)
        vector_db.search_vector_database.return_value = ([-1, 0, 99, 2], [0.9, 0.8, 0.7, 0.6])

        records = retriever.retrieve("q", return_metadata=True)
        texts = [r["text"] for r in records]
        self.assertEqual(set(texts), {"alpha chunk", "gamma chunk"})
        self.assertNotIn("beta chunk", texts)  # would be iloc[-1] target region
        # Scores stay aligned with the surviving valid rows.
        by_text = {r["text"]: r["retrieval_score"] for r in records}
        self.assertEqual(by_text["alpha chunk"], 0.8)
        self.assertEqual(by_text["gamma chunk"], 0.6)

    def test_sparse_path_drops_negative_and_out_of_range(self):
        # Sparse (BM25) hits with invalid row indices are skipped as well.
        sparse = FakeSparseRetriever([(-1, 5.0), (2, 3.0), (99, 2.0), (0, 1.0)])
        retriever, vector_db = _build_retriever(sparse_retriever=sparse, use_hybrid=True, use_rrf=True)
        vector_db.search_vector_database.return_value = ([0], [0.9])

        records = retriever.retrieve("q", return_metadata=True)
        texts = [r["text"] for r in records]
        self.assertEqual(set(texts), {"alpha chunk", "gamma chunk"})

    def test_dense_path_returns_none_when_all_indices_invalid(self):
        retriever, vector_db = _build_retriever(use_rrf=True)
        vector_db.search_vector_database.return_value = ([-1, -1], [0.0, 0.0])

        self.assertIsNone(retriever.retrieve("q", return_metadata=True))


class TestChunkReadExpansion(unittest.TestCase):
    def test_neighbor_window_expands_context(self):
        retriever, vector_db = _build_retriever(chunk_read=1)
        vector_db.search_vector_database.return_value = ([1], [0.9])

        records = retriever.retrieve("q", return_metadata=True)
        record = records[0]
        # Precise child identity/text preserved.
        self.assertEqual(record["child_text"], "beta chunk")
        # Expanded text pulls the +/-1 neighbor window within doc 0.
        self.assertIn("alpha chunk", record["text"])
        self.assertIn("beta chunk", record["text"])
        self.assertEqual(record["neighbor_chunk_ids"], [0, 1])

    def test_parent_store_expands_to_parent(self):
        parent_store = pd.DataFrame([{PARENT_ID_KEY: "p0", DOC_TEXT_KEY: "full parent context"}])
        retriever, vector_db = _build_retriever(chunk_read=0, parent_store=parent_store)
        vector_db.metadata = pd.DataFrame(
            [{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "child bit", PARENT_ID_KEY: "p0"}]
        )
        vector_db.search_vector_database.return_value = ([0], [0.9])

        records = retriever.retrieve("q", return_metadata=True)
        self.assertEqual(records[0]["child_text"], "child bit")
        self.assertIn("full parent context", records[0]["text"])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from agrag.modules.retriever.fusion import dedup_records, default_dedup_key, mmr, reciprocal_rank_fusion


class TestReciprocalRankFusion(unittest.TestCase):
    def test_single_list_preserves_order(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
        self.assertEqual([key for key, _ in fused], ["a", "b", "c"])
        # Scores strictly decrease with rank.
        scores = [score for _, score in fused]
        self.assertTrue(scores[0] > scores[1] > scores[2])

    def test_agreement_boosts_shared_item(self):
        # "b" appears in both lists (ranks 2 and 1); it should overtake items
        # that only appear once at rank 1.
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        order = [key for key, _ in fused]
        self.assertEqual(order[0], "b")
        self.assertEqual(set(order), {"a", "b", "c"})

    def test_weights_shift_ranking(self):
        # Without weights the two singletons tie and first-seen wins ("a").
        # Heavily weighting the second list flips the top item to "b".
        fused = reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[0.1, 10.0])
        self.assertEqual(fused[0][0], "b")

    def test_k_controls_top_rank_dominance(self):
        # A small k makes rank differences sharper; a shared item's advantage
        # over a rank-1 singleton grows as k shrinks.
        small_k = dict(reciprocal_rank_fusion([["a", "b"], ["b"]], k=1))
        large_k = dict(reciprocal_rank_fusion([["a", "b"], ["b"]], k=1000))
        self.assertGreater(small_k["b"] - small_k["a"], large_k["b"] - large_k["a"])

    def test_tie_broken_by_first_appearance(self):
        fused = reciprocal_rank_fusion([["x"], ["y"]], k=60)
        self.assertEqual([key for key, _ in fused], ["x", "y"])

    def test_mismatched_weights_length_raises(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])


class TestMMR(unittest.TestCase):
    def test_lambda_one_is_pure_relevance(self):
        query = np.array([1.0, 0.0])
        keys = ["a", "b", "c"]
        embeddings = [[0.9, 0.1], [1.0, 0.0], [0.0, 1.0]]
        selected = mmr(query, keys, embeddings, lambda_mult=1.0)
        # Most query-aligned first, least aligned last.
        self.assertEqual(selected[0], "b")
        self.assertEqual(selected[-1], "c")

    def test_diversity_picks_orthogonal_second(self):
        # Two near-identical relevant docs and one diverse doc. With diversity
        # weighting, the second pick should be the diverse one, not the near-dup.
        query = np.array([1.0, 0.0])
        keys = ["dup1", "dup2", "diverse"]
        embeddings = [[1.0, 0.0], [0.99, 0.01], [0.2, 1.0]]
        selected = mmr(query, keys, embeddings, lambda_mult=0.3)
        self.assertEqual(selected[0], "dup1")
        self.assertEqual(selected[1], "diverse")

    def test_top_k_limits_output(self):
        query = np.array([1.0, 0.0])
        keys = ["a", "b", "c"]
        embeddings = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        selected = mmr(query, keys, embeddings, lambda_mult=0.5, top_k=2)
        self.assertEqual(len(selected), 2)

    def test_empty_candidates(self):
        self.assertEqual(mmr(np.array([1.0]), [], []), [])


class TestDedupRecords(unittest.TestCase):
    def test_default_key_prefers_doc_chunk_identity(self):
        self.assertEqual(default_dedup_key({"doc_id": 3, "chunk_id": 7}), (3, 7))
        self.assertEqual(default_dedup_key({"text": "hello"}), ("text", "hello"))

    def test_merges_provenance_across_subqueries(self):
        records = [
            {"doc_id": 0, "chunk_id": 0, "text": "a", "retrieval_query": "q1", "signal": "dense", "rank": 2},
            {"doc_id": 0, "chunk_id": 0, "text": "a", "retrieval_query": "q2", "signal": "sparse", "rank": 0},
        ]
        deduped = dedup_records(records)
        self.assertEqual(len(deduped), 1)
        merged = deduped[0]
        # Every subquery and signal that surfaced the chunk is preserved.
        self.assertEqual(merged["retrieval_queries"], ["q1", "q2"])
        self.assertEqual(merged["source_signals"], ["dense", "sparse"])
        # The strongest (minimum) rank is kept.
        self.assertEqual(merged["rank"], 0)

    def test_preserves_first_appearance_order(self):
        records = [
            {"doc_id": 1, "chunk_id": 0, "text": "b", "retrieval_query": "q1"},
            {"doc_id": 0, "chunk_id": 0, "text": "a", "retrieval_query": "q1"},
            {"doc_id": 1, "chunk_id": 0, "text": "b", "retrieval_query": "q2"},
        ]
        deduped = dedup_records(records)
        self.assertEqual([(r["doc_id"], r["chunk_id"]) for r in deduped], [(1, 0), (0, 0)])

    def test_no_duplicates_passthrough(self):
        records = [
            {"doc_id": 0, "chunk_id": 0, "text": "a", "retrieval_query": "q"},
            {"doc_id": 0, "chunk_id": 1, "text": "b", "retrieval_query": "q"},
        ]
        deduped = dedup_records(records)
        self.assertEqual(len(deduped), 2)


if __name__ == "__main__":
    unittest.main()

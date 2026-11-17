import unittest

from agrag.modules.retriever.retrievers.sparse_retriever import BM25Retriever


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning models require large training datasets",
            "the dog barked loudly at the mail carrier",
            "gradient descent optimizes neural network weights",
        ]

    def test_lexical_match_ranks_first(self):
        bm25 = BM25Retriever().build(self.corpus)
        hits = bm25.search("neural network gradient", top_k=4)
        self.assertTrue(hits)
        # Row 3 is the only doc with these terms; it must rank first.
        self.assertEqual(hits[0][0], 3)

    def test_returns_row_indices_aligned_to_corpus(self):
        bm25 = BM25Retriever().build(self.corpus)
        hits = bm25.search("dog", top_k=10)
        rows = {idx for idx, _ in hits}
        # "dog" appears in rows 0 and 2 only.
        self.assertEqual(rows, {0, 2})

    def test_scores_sorted_descending(self):
        bm25 = BM25Retriever().build(self.corpus)
        hits = bm25.search("the dog", top_k=10)
        scores = [score for _, score in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limits_results(self):
        bm25 = BM25Retriever().build(self.corpus)
        hits = bm25.search("the", top_k=1)
        self.assertLessEqual(len(hits), 1)

    def test_zero_score_docs_excluded(self):
        bm25 = BM25Retriever().build(self.corpus)
        hits = bm25.search("machine learning", top_k=10)
        # Only row 1 contains these terms; no zero-score padding.
        self.assertEqual([idx for idx, _ in hits], [1])

    def test_query_with_no_known_terms(self):
        bm25 = BM25Retriever().build(self.corpus)
        self.assertEqual(bm25.search("supercalifragilistic", top_k=5), [])

    def test_empty_query_returns_empty(self):
        bm25 = BM25Retriever().build(self.corpus)
        self.assertEqual(bm25.search("", top_k=5), [])

    def test_search_before_build_returns_empty(self):
        self.assertEqual(BM25Retriever().search("dog", top_k=5), [])

    def test_empty_corpus(self):
        bm25 = BM25Retriever().build([])
        self.assertEqual(bm25.search("dog", top_k=5), [])

    def test_k1_b_parameters_stored(self):
        bm25 = BM25Retriever(k1=1.2, b=0.5)
        self.assertEqual(bm25.k1, 1.2)
        self.assertEqual(bm25.b, 0.5)


if __name__ == "__main__":
    unittest.main()

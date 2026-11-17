import unittest

from agrag.evaluation.retrieval_metrics import (
    _gold_units,
    aggregate_retrieval_metrics,
    fact_is_retrieved,
    retrieval_metrics_for_query,
)


class TestFactIsRetrieved(unittest.TestCase):
    def test_containment_above_threshold_matches(self):
        fact = "the electric car company reported record deliveries"
        chunk = "In its update, the electric car company reported record deliveries this quarter."
        self.assertTrue(fact_is_retrieved(fact, chunk, threshold=0.6))

    def test_low_overlap_does_not_match(self):
        fact = "the electric car company reported record deliveries"
        chunk = "the weather forecast predicts rain over the weekend"
        self.assertFalse(fact_is_retrieved(fact, chunk, threshold=0.6))

    def test_empty_fact_never_matches(self):
        self.assertFalse(fact_is_retrieved("the and of", "anything at all", threshold=0.6))


class TestSentenceLevelGoldMatching(unittest.TestCase):
    def test_long_paragraph_matches_via_single_sentence(self):
        # A multi-sentence gold "fact" (paragraph). Only the second sentence is in
        # the retrieved chunk. Whole-paragraph containment would fall well under
        # 0.6 (the chunk holds a minority of the paragraph's tokens), but
        # sentence-level matching counts the fact as retrieved.
        fact = (
            "The company was founded in a small garage and grew steadily over many "
            "decades into a global enterprise employing thousands of people worldwide. "
            "Its headquarters relocated to the city of Portland in 1994. "
            "The founder later stepped down and pursued philanthropic ventures across "
            "several continents for the remainder of his career."
        )
        chunk = "Its headquarters relocated to the city of Portland in 1994."
        # Sanity: the chunk is far shorter than the paragraph, so paragraph-level
        # containment cannot clear the bar, but the sentence does.
        self.assertTrue(fact_is_retrieved(fact, chunk, threshold=0.6))

    def test_single_sentence_fact_unchanged(self):
        # A short single-sentence fact yields exactly one unit -> same as before.
        fact = "the electric car company reported record deliveries"
        chunk = "In its update, the electric car company reported record deliveries this quarter."
        self.assertTrue(fact_is_retrieved(fact, chunk, threshold=0.6))
        self.assertEqual(len(_gold_units(fact)), 1)

    def test_unrelated_chunk_still_does_not_match(self):
        # Splitting into sentences must not create spurious matches.
        fact = (
            "The treaty was signed in the spring of that year. "
            "Delegates from a dozen nations attended the ceremony."
        )
        chunk = "the weather forecast predicts rain over the weekend"
        self.assertFalse(fact_is_retrieved(fact, chunk, threshold=0.6))

    def test_gold_units_splits_sentences(self):
        fact = "First sentence here. Second sentence follows! Third one ends?"
        self.assertEqual(len(_gold_units(fact)), 3)

    def test_gold_units_bare_phrase_is_single_unit(self):
        self.assertEqual(_gold_units("Portland Oregon"), ["Portland Oregon"])


class TestRetrievalMetricsForQuery(unittest.TestCase):
    def test_hit_and_mrr_first_position(self):
        gold = ["quantum computing breakthrough announced"]
        retrieved = ["quantum computing breakthrough announced today", "unrelated chunk"]
        metrics = retrieval_metrics_for_query(retrieved, gold, k_values=(1, 3))
        self.assertEqual(metrics["hit@1"], 1.0)
        self.assertEqual(metrics["hit@3"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_hit_respects_cutoff(self):
        gold = ["quantum computing breakthrough announced"]
        retrieved = ["noise one", "noise two", "quantum computing breakthrough announced now"]
        metrics = retrieval_metrics_for_query(retrieved, gold, k_values=(1, 3))
        self.assertEqual(metrics["hit@1"], 0.0)  # first hit at rank 3
        self.assertEqual(metrics["hit@3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0 / 3.0)

    def test_no_hit(self):
        gold = ["quantum computing breakthrough announced"]
        retrieved = ["totally different topic", "another off topic chunk"]
        metrics = retrieval_metrics_for_query(retrieved, gold, k_values=(1, 3))
        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["hit@3"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)
        self.assertEqual(metrics["recall@3"], 0.0)

    def test_recall_at_k_partial_coverage(self):
        # Two gold facts; each matched by a chunk at a different position.
        gold = [
            "alpha corporation acquired beta industries",
            "gamma bank raised interest rates sharply",
        ]
        retrieved = [
            "alpha corporation acquired beta industries yesterday",  # covers fact 1
            "some filler chunk with no gold content whatsoever",
            "gamma bank raised interest rates sharply this morning",  # covers fact 2
        ]
        metrics = retrieval_metrics_for_query(retrieved, gold, k_values=(1, 3))
        # At cutoff 1 only the first fact is covered -> 0.5.
        self.assertAlmostEqual(metrics["recall@1"], 0.5)
        # At cutoff 3 both facts are covered -> 1.0.
        self.assertAlmostEqual(metrics["recall@3"], 1.0)
        # evidence_coverage over the whole list is full coverage.
        self.assertAlmostEqual(metrics["evidence_coverage"], 1.0)

    def test_recall_never_exceeds_evidence_coverage(self):
        gold = ["fact one here", "second distinct fact appears"]
        retrieved = ["fact one here exactly", "second distinct fact appears exactly"]
        metrics = retrieval_metrics_for_query(retrieved, gold, k_values=(1, 5))
        self.assertLessEqual(metrics["recall@1"], metrics["evidence_coverage"])
        self.assertLessEqual(metrics["recall@5"], metrics["evidence_coverage"])


class TestAggregateRetrievalMetrics(unittest.TestCase):
    def test_means_across_queries(self):
        per_query = [
            {"hit@1": 1.0, "recall@1": 1.0, "mrr": 1.0},
            {"hit@1": 0.0, "recall@1": 0.0, "mrr": 0.5},
        ]
        agg = aggregate_retrieval_metrics(per_query)
        self.assertAlmostEqual(agg["hit@1"], 0.5)
        self.assertAlmostEqual(agg["recall@1"], 0.5)
        self.assertAlmostEqual(agg["mrr"], 0.75)

    def test_empty_returns_empty(self):
        self.assertEqual(aggregate_retrieval_metrics([]), {})

    def test_recall_keys_are_aggregated(self):
        per_query = [retrieval_metrics_for_query(["fact one here exactly"], ["fact one here"], k_values=(1, 3))]
        agg = aggregate_retrieval_metrics(per_query)
        self.assertIn("recall@1", agg)
        self.assertIn("recall@3", agg)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

import torch

from agrag.modules.retriever.rerankers.reranker import Reranker


class TestReranker(unittest.TestCase):
    @patch("agrag.modules.retriever.rerankers.reranker.AutoModelForSequenceClassification.from_pretrained")
    @patch("agrag.modules.retriever.rerankers.reranker.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer, mock_model):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model

        self.reranker = Reranker(
            model_name="some-cross-encoder",
            batch_size=2,
            platform_args={
                "hf_tokenizer_params": {"padding": True, "max_length": 512, "return_tensors": "pt"},
            },
        )

    def _tokenizer_returns_two(self):
        """Tokenizer output for a batch of two (query, doc) pairs."""
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

    def _model_logits(self, logits):
        """Make the reranker's model return a seq-classification output with .logits.

        ``__init__`` calls ``.to(device)`` on the loaded model, so ``self.reranker.model``
        is not the raw ``self.mock_model``; assign the callable directly.
        """
        output = MagicMock()
        output.logits = torch.tensor(logits)
        self.reranker.model = MagicMock(return_value=output)

    def test_rerank_sorts_by_scalar_score(self):
        self._tokenizer_returns_two()
        # Cross-encoder head: one scalar logit per pair, shape (batch, 1).
        self._model_logits([[0.1], [0.9]])

        sorted_chunks = self.reranker.rerank("query", ["chunk 1", "chunk 2"])

        # chunk 2 has the higher relevance logit, so it ranks first.
        self.assertEqual(sorted_chunks, ["chunk 2", "chunk 1"])

    def test_rerank_top_k(self):
        self._tokenizer_returns_two()
        self._model_logits([[0.1], [0.9]])
        self.reranker.top_k = 1

        sorted_chunks = self.reranker.rerank("query", ["chunk 1", "chunk 2"])

        self.assertEqual(sorted_chunks, ["chunk 2"])

    def test_rerank_return_scores_are_scalars(self):
        self._tokenizer_returns_two()
        self._model_logits([[0.1], [0.9]])

        scored = self.reranker.rerank("query", ["chunk 1", "chunk 2"], return_scores=True)

        # (text, score) tuples sorted by descending score.
        self.assertEqual([t for t, _ in scored], ["chunk 2", "chunk 1"])
        # Regression guard against the old bug where scores were hidden-state
        # vectors: each score must be a single float, not a list/sequence.
        for _, score in scored:
            self.assertIsInstance(score, float)
        self.assertAlmostEqual(scored[0][1], 0.9, places=5)
        self.assertAlmostEqual(scored[1][1], 0.1, places=5)

    def test_rerank_handles_multi_label_head(self):
        # A model exposing >1 label: the first column is used as the relevance score.
        self._tokenizer_returns_two()
        self._model_logits([[0.2, 0.8], [0.7, 0.3]])

        scored = self.reranker.rerank("query", ["chunk 1", "chunk 2"], return_scores=True)

        self.assertEqual([t for t, _ in scored], ["chunk 2", "chunk 1"])
        self.assertAlmostEqual(scored[0][1], 0.7, places=5)

    def test_default_tokenizer_params_applied(self):
        # When no hf_tokenizer_params are supplied, safe defaults must be present
        # so the reranker works without explicit config.
        with patch(
            "agrag.modules.retriever.rerankers.reranker.AutoModelForSequenceClassification.from_pretrained"
        ), patch("agrag.modules.retriever.rerankers.reranker.AutoTokenizer.from_pretrained"):
            reranker = Reranker(model_name="some-cross-encoder")
        self.assertTrue(reranker.hf_tokenizer_params["padding"])
        self.assertTrue(reranker.hf_tokenizer_params["truncation"])
        self.assertEqual(reranker.hf_tokenizer_params["return_tensors"], "pt")
        self.assertEqual(reranker.hf_tokenizer_params["max_length"], 512)

    def test_user_tokenizer_params_override_defaults(self):
        with patch(
            "agrag.modules.retriever.rerankers.reranker.AutoModelForSequenceClassification.from_pretrained"
        ), patch("agrag.modules.retriever.rerankers.reranker.AutoTokenizer.from_pretrained"):
            reranker = Reranker(
                model_name="some-cross-encoder",
                platform_args={"hf_tokenizer_params": {"max_length": 128}},
            )
        # User value wins; other defaults remain.
        self.assertEqual(reranker.hf_tokenizer_params["max_length"], 128)
        self.assertTrue(reranker.hf_tokenizer_params["truncation"])


if __name__ == "__main__":
    unittest.main()

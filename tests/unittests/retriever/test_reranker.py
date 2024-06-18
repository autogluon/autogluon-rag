import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import AutoModel, AutoTokenizer

from agrag.modules.retriever.rerankers.reranker import Reranker


class TestReranker(unittest.TestCase):
    @patch("agrag.modules.retriever.rerankers.reranker.AutoModel.from_pretrained")
    @patch("agrag.modules.retriever.rerankers.reranker.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer, mock_model):

        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model

    def test_rerank(self):
        query = "Some query"
        text_chunks = ["test chunk 1", "test chunk 2"]

        self.mock_tokenizer.return_tensors.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        self.mock_model.return_value = [torch.rand((2, 3, 10))]

        reranker = Reranker()
        sorted_text_chunks = reranker.rerank(query, text_chunks)

        self.assertEqual(sorted_text_chunks, ["test chunk 1", "test chunk 2"])


if __name__ == "__main__":
    unittest.main()

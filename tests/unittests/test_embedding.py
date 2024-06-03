import unittest
from unittest.mock import MagicMock, patch

import torch

from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.embedding.utils import pool


class TestEmbeddingModule(unittest.TestCase):
    @patch("agrag.modules.embedding.embedding.AutoTokenizer.from_pretrained")
    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model

        self.embedding_module = EmbeddingModule(model_name="some-model", pooling_strategy=None)

    def test_create_embeddings_no_pooling(self):
        self.mock_tokenizer.return_tensors.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        self.mock_model.return_value = MagicMock(last_hidden_state=torch.rand((10, 20, 100)))

        data = ["test sentence 1", "test sentence 2"]
        embeddings = self.embedding_module.create_embeddings(data)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(isinstance(embedding, torch.Tensor) for embedding in embeddings))

    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def test_pool_average(self, mock_model):
        self.embedding_module.pooling_strategy = "average"
        mock_model.return_value = MagicMock(last_hidden_state=torch.rand((10, 20, 100)))
        embeddings = torch.rand((10, 20, 100))
        pooled_embeddings = pool(embeddings, self.embedding_module.pooling_strategy)

        self.assertEqual(pooled_embeddings.shape, (10, 100))

    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def test_pool_max(self, mock_model):
        self.embedding_module.pooling_strategy = "max"
        mock_model.return_value = MagicMock(last_hidden_state=torch.rand((10, 20, 100)))
        embeddings = torch.rand((10, 20, 100))
        pooled_embeddings = pool(embeddings, self.embedding_module.pooling_strategy)

        self.assertEqual(pooled_embeddings.shape, (10, 100))

    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def test_pool_cls(self, mock_model):
        self.embedding_module.pooling_strategy = "cls"
        mock_model.return_value = MagicMock(last_hidden_state=torch.rand((10, 20, 100)))
        embeddings = torch.rand((10, 20, 100))
        pooled_embeddings = pool(embeddings, self.embedding_module.pooling_strategy)

        self.assertEqual(pooled_embeddings.shape, (10, 100))


if __name__ == "__main__":
    unittest.main()

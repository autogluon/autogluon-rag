import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from agrag.constants import DOC_TEXT_KEY, EMBEDDING_KEY
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.embedding.utils import normalize_embedding, pool


class TestEmbeddingModule(unittest.TestCase):
    @patch("agrag.modules.embedding.embedding.AutoTokenizer.from_pretrained")
    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model

        model_name_params = {"param": "param"}
        hf_tokenizer_params = {"param": True}
        tokenizer_params = {"padding": 10, "max_length": 512}
        forward_params = {"param": True}

        self.embedding_module = EmbeddingModule(
            model_name="some-model",
            pooling_strategy=None,
            model_name_params=model_name_params,
            hf_tokenizer_init_params=hf_tokenizer_params,
            hf_tokenizer_params=tokenizer_params,
            hf_forward_params=forward_params,
        )

    def test_encode_hf(self):
        self.mock_tokenizer.return_tensors.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        self.mock_model.return_value = [torch.rand((2, 3, 10))]

        data = pd.DataFrame([{DOC_TEXT_KEY: "test sentence 1"}, {DOC_TEXT_KEY: "test sentence 2"}])
        embeddings_df = self.embedding_module.encode(data)

        self.assertEqual(len(embeddings_df), 2)
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings_df[EMBEDDING_KEY]))

    def test_encode_hf_pool(self):
        self.mock_tokenizer.return_tensors.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        self.mock_model.return_value = [torch.rand((2, 3, 10))]

        data = pd.DataFrame([{DOC_TEXT_KEY: "test sentence 1"}, {DOC_TEXT_KEY: "test sentence 2"}])
        self.embedding_module.pooling_strategy = "mean"
        embeddings_df = self.embedding_module.encode(data)

        self.assertEqual(len(embeddings_df), 2)
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings_df[EMBEDDING_KEY]))

    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def test_pool_mean(self, mock_model):
        self.embedding_module.pooling_strategy = "mean"
        mock_model.return_value = MagicMock(last_hidden_state=torch.rand((10, 20, 100)))
        embeddings = torch.rand((10, 20, 100))

        expected_pooled = embeddings.mean(dim=1)

        pooled_embeddings = pool(embeddings, self.embedding_module.pooling_strategy)

        self.assertTrue(torch.allclose(pooled_embeddings, expected_pooled, atol=1e-6))

    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def test_pool_max(self, mock_model):
        self.embedding_module.pooling_strategy = "max"
        mock_model.return_value = MagicMock(last_hidden_state=torch.rand((10, 20, 100)))
        embeddings = torch.rand((10, 20, 100))

        expected_pooled = embeddings.max(dim=1).values

        pooled_embeddings = pool(embeddings, self.embedding_module.pooling_strategy)

        self.assertTrue(torch.allclose(pooled_embeddings, expected_pooled, atol=1e-6))

    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def test_pool_cls(self, mock_model):
        self.embedding_module.pooling_strategy = "cls"
        embeddings = torch.rand((10, 20, 100))

        expected_pooled = embeddings[:, 0]

        pooled_embeddings = pool(embeddings, self.embedding_module.pooling_strategy)

        self.assertTrue(torch.allclose(pooled_embeddings, expected_pooled, atol=1e-6))

    def test_normalize_embedding(self):
        embedding = torch.rand((10, 100))
        normalized_embedding = normalize_embedding(embedding, {})

        expected_norm = torch.ones((10,))
        actual_norm = torch.norm(normalized_embedding, p=2, dim=1)

        self.assertTrue(torch.allclose(expected_norm, actual_norm, atol=1e-6))

    @patch.object(EmbeddingModule, "encode")
    def test_encode_queries_with_instruction(self, mock_encode):
        self.embedding_module.query_instruction_for_retrieval = "Instruction: "
        queries = ["query1", "query2"]
        expected_input_texts = ["Instruction: query1", "Instruction: query2"]
        expected_input_texts = pd.DataFrame({DOC_TEXT_KEY: expected_input_texts})
        mock_encode.return_value = np.random.rand(2, 10)

        embeddings = self.embedding_module.encode_queries(queries)

        pd.testing.assert_frame_equal(mock_encode.call_args[0][0], expected_input_texts)
        self.assertEqual(embeddings.shape, (2, 10))

    @patch.object(EmbeddingModule, "encode")
    def test_encode_queries_without_instruction(self, mock_encode):
        self.embedding_module.query_instruction_for_retrieval = None
        queries = ["query1", "query2"]
        expected_input_texts = queries
        expected_input_texts = pd.DataFrame({DOC_TEXT_KEY: expected_input_texts})
        mock_encode.return_value = np.random.rand(2, 10)

        embeddings = self.embedding_module.encode_queries(queries)

        pd.testing.assert_frame_equal(mock_encode.call_args[0][0], expected_input_texts)
        self.assertEqual(embeddings.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()

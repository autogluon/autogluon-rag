import unittest
from unittest.mock import MagicMock, patch

import torch
from sentence_transformers import SentenceTransformer

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

        hf_model_params = {"param": "param"}
        hf_tokenizer_params = {"param": True}
        tokenizer_params = {"padding": 10, "max_length": 512}
        forward_params = {"param": True}

        self.embedding_module = EmbeddingModule(
            hf_model="some-model",
            pooling_strategy=None,
            hf_model_params=hf_model_params,
            hf_tokenizer_init_params=hf_tokenizer_params,
            hf_tokenizer_params=tokenizer_params,
            hf_forward_params=forward_params,
        )

    def test_create_embeddings_hf_no_pooling(self):
        self.mock_tokenizer.return_tensors.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        self.mock_model.return_value = MagicMock(last_hidden_state=torch.rand((10, 20, 100)))

        data = ["test sentence 1", "test sentence 2"]
        self.embedding_module.use_sentence_transf = False
        embeddings = self.embedding_module.create_embeddings(data)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(isinstance(embedding, torch.Tensor) for embedding in embeddings))

    @patch("sentence_transformers.SentenceTransformer")
    def test_create_embeddings_sentence_transformer(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.rand((2, 100))
        mock_sentence_transformer.return_value = mock_model

        st_params = {"param": "param"}
        encode_params = {"batch_size": 32}

        with patch("agrag.modules.embedding.embedding.SentenceTransformer", return_value=mock_model):
            self.embedding_module = EmbeddingModule(
                st_model="sentence_transformer",
                pooling_strategy=None,
                st_params=st_params,
                st_encode_params=encode_params,
                use_sentence_transf=True,
            )

            data = ["test sentence 1", "test sentence 2"]
            embeddings = self.embedding_module.create_embeddings(data)

            self.assertEqual(embeddings.shape, (2, 100))

    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def test_pool_mean(self, mock_model):
        self.embedding_module.pooling_strategy = "mean"
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

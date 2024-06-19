import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from agrag.constants import DOC_TEXT_KEY
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.retriever.retrievers.retriever_base import RetrieverModule
from agrag.modules.vector_db.vector_database import VectorDatabaseModule


class TestRetrieverModule(unittest.TestCase):
    @patch("agrag.modules.embedding.embedding.AutoTokenizer.from_pretrained")
    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model

        self.embedding_module = EmbeddingModule(
            hf_model="some-model",
            pooling_strategy=None,
            hf_model_params={},
            hf_tokenizer_init_params={},
            hf_tokenizer_params={"padding": 10, "max_length": 512},
            hf_forward_params={},
        )

        self.vector_database_module = MagicMock(VectorDatabaseModule)
        self.retriever_module = RetrieverModule(
            vector_database_module=self.vector_database_module,
            embedding_module=self.embedding_module,
            top_k=5,
            reranker=Reranker(model_name="some model"),
        )

    def test_encode_query(self):
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        self.mock_model.return_value = [torch.rand((1, 10))]

        query = "test query"
        query_embedding = self.retriever_module.encode_query(query)

        self.assertIsInstance(query_embedding, np.ndarray)
        self.assertEqual(query_embedding.shape, (10,))

    @patch("agrag.modules.retriever.rerankers.reranker.Reranker.rerank")
    def test_retrieve(self, mock_rerank):
        query = "test query"
        text_chunks = ["test chunk 1", "test chunk 2", "test chunk 3"]
        self.vector_database_module.search_vector_database.return_value = [0, 1, 2]
        self.vector_database_module.metadata = pd.DataFrame(
            [{DOC_TEXT_KEY: "test chunk 1"}, {DOC_TEXT_KEY: "test chunk 2"}, {DOC_TEXT_KEY: "test chunk 3"}]
        )
        mock_rerank.return_value = text_chunks
        self.mock_model.return_value = [torch.rand((1, 3, 10))]

        retrieved_chunks = self.retriever_module.retrieve(query)

        self.assertEqual(retrieved_chunks, text_chunks)


if __name__ == "__main__":
    unittest.main()

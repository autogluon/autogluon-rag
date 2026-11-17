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
    @patch("agrag.modules.retriever.rerankers.reranker.AutoTokenizer.from_pretrained")
    @patch("agrag.modules.retriever.rerankers.reranker.AutoModelForSequenceClassification.from_pretrained")
    @patch("agrag.modules.embedding.embedding.AutoTokenizer.from_pretrained")
    @patch("agrag.modules.embedding.embedding.AutoModel.from_pretrained")
    def setUp(self, mock_model, mock_tokenizer, mock_reranker_model, mock_reranker_tokenizer):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model
        # The reranker loads its own cross-encoder; mock it so no model is downloaded.
        mock_reranker_model.return_value = MagicMock()
        mock_reranker_tokenizer.return_value = MagicMock()

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

    @patch("agrag.modules.retriever.rerankers.reranker.Reranker.rerank")
    def test_retrieve_with_metadata_realigns_after_rerank(self, mock_rerank):
        query = "test query"
        # The metadata path requests scores, so the DB returns (indices, scores).
        self.vector_database_module.search_vector_database.return_value = ([0, 1, 2], [0.1, 0.2, 0.3])
        self.vector_database_module.metadata = pd.DataFrame(
            [
                {"doc_id": 0, "chunk_id": 0, DOC_TEXT_KEY: "test chunk 1"},
                {"doc_id": 0, "chunk_id": 1, DOC_TEXT_KEY: "test chunk 2"},
                {"doc_id": 1, "chunk_id": 0, DOC_TEXT_KEY: "test chunk 3"},
            ]
        )
        # Reranker reverses the order and returns (text, score) pairs.
        mock_rerank.return_value = [("test chunk 3", 9.0), ("test chunk 2", 5.0), ("test chunk 1", 1.0)]
        self.mock_model.return_value = [torch.rand((1, 3, 10))]

        records = self.retriever_module.retrieve(query, return_metadata=True)

        # Records keep provenance, and rank reflects the reranked order.
        self.assertEqual([r["text"] for r in records], ["test chunk 3", "test chunk 2", "test chunk 1"])
        self.assertEqual([r["rank"] for r in records], [0, 1, 2])
        self.assertEqual(records[0]["doc_id"], 1)
        self.assertEqual(records[0]["chunk_id"], 0)
        # Scores are now threaded through: retrieval_score from the DB (realigned
        # by text) and rerank_score from the reranker.
        self.assertEqual(records[0]["rerank_score"], 9.0)
        self.assertEqual(records[0]["retrieval_score"], 0.3)  # "test chunk 3" was index 2

    def test_retrieve_with_metadata_no_reranker(self):
        query = "test query"
        self.vector_database_module.search_vector_database.return_value = ([0, 1], [0.5, 0.6])
        self.vector_database_module.metadata = pd.DataFrame(
            [
                {"doc_id": 0, "chunk_id": 0, DOC_TEXT_KEY: "chunk a"},
                {"doc_id": 0, "chunk_id": 1, DOC_TEXT_KEY: "chunk b"},
            ]
        )
        self.mock_model.return_value = [torch.rand((1, 2, 10))]
        # Disable the reranker for this case.
        self.retriever_module.reranker = None

        records = self.retriever_module.retrieve(query, return_metadata=True)

        self.assertEqual([r["text"] for r in records], ["chunk a", "chunk b"])
        self.assertEqual([r["rank"] for r in records], [0, 1])
        self.assertEqual(records[1]["chunk_id"], 1)
        # retrieval_score present; no reranker means no rerank_score.
        self.assertEqual([r["retrieval_score"] for r in records], [0.5, 0.6])
        self.assertNotIn("rerank_score", records[0])

    def test_retrieve_top_k_override(self):
        query = "test query"
        self.vector_database_module.search_vector_database.return_value = ([0], [0.1])
        self.vector_database_module.metadata = pd.DataFrame([{"doc_id": 0, "chunk_id": 0, DOC_TEXT_KEY: "chunk a"}])
        self.mock_model.return_value = [torch.rand((1, 1, 10))]
        self.retriever_module.reranker = None

        self.retriever_module.retrieve(query, return_metadata=True, top_k=3)

        # The override, not the module's top_k (5), is passed to the DB search.
        _, kwargs = self.vector_database_module.search_vector_database.call_args
        self.assertEqual(kwargs["top_k"], 3)

    def test_retrieve_default_top_k_used_when_override_none(self):
        query = "test query"
        self.vector_database_module.search_vector_database.return_value = ([0], [0.1])
        self.vector_database_module.metadata = pd.DataFrame([{"doc_id": 0, "chunk_id": 0, DOC_TEXT_KEY: "chunk a"}])
        self.mock_model.return_value = [torch.rand((1, 1, 10))]
        self.retriever_module.reranker = None

        self.retriever_module.retrieve(query, return_metadata=True)

        _, kwargs = self.vector_database_module.search_vector_database.call_args
        self.assertEqual(kwargs["top_k"], 5)  # module-level top_k

    def test_retrieve_returns_none_when_no_valid_indices(self):
        query = "test query"
        self.vector_database_module.search_vector_database.return_value = ([99], [0.1])
        self.vector_database_module.metadata = pd.DataFrame([{DOC_TEXT_KEY: "only chunk"}])
        self.mock_model.return_value = [torch.rand((1, 1, 10))]

        self.assertIsNone(self.retriever_module.retrieve(query, return_metadata=True))

    @patch("agrag.modules.retriever.rerankers.reranker.Reranker.rerank")
    def test_retrieve_text_drops_faiss_negative_sentinel(self, mock_rerank):
        # FAISS returns -1 for empty slots when there are fewer than top_k hits.
        # -1 must be dropped, never mapped to metadata.iloc[-1] (the last row).
        query = "test query"
        self.vector_database_module.search_vector_database.return_value = [0, -1]
        self.vector_database_module.metadata = pd.DataFrame(
            [{DOC_TEXT_KEY: "first chunk"}, {DOC_TEXT_KEY: "last chunk"}]
        )
        # Reranker is identity so we can assert on the content passed in.
        mock_rerank.side_effect = lambda q, content: content
        self.mock_model.return_value = [torch.rand((1, 1, 10))]

        retrieved = self.retriever_module.retrieve(query)

        # Only the valid row survives; the -1 sentinel does not become "last chunk".
        self.assertEqual(retrieved, ["first chunk"])

    def test_retrieve_metadata_drops_negative_and_out_of_range(self):
        # Both a negative sentinel (-1) and an out-of-range index (99) must be
        # dropped while index/score alignment is preserved for the valid rows.
        query = "test query"
        self.vector_database_module.search_vector_database.return_value = (
            [-1, 0, 99, 1],
            [0.9, 0.8, 0.7, 0.6],
        )
        self.vector_database_module.metadata = pd.DataFrame(
            [
                {"doc_id": 0, "chunk_id": 0, DOC_TEXT_KEY: "chunk a"},
                {"doc_id": 0, "chunk_id": 1, DOC_TEXT_KEY: "chunk b"},
            ]
        )
        self.mock_model.return_value = [torch.rand((1, 2, 10))]
        self.retriever_module.reranker = None

        records = self.retriever_module.retrieve(query, return_metadata=True)

        # Only the in-range rows survive, keeping their score alignment and order.
        self.assertEqual([r["text"] for r in records], ["chunk a", "chunk b"])
        self.assertEqual([r["retrieval_score"] for r in records], [0.8, 0.6])
        self.assertEqual([r["rank"] for r in records], [0, 1])

    def test_retrieve_returns_none_when_all_indices_negative(self):
        query = "test query"
        self.vector_database_module.search_vector_database.return_value = ([-1, -1], [0.0, 0.0])
        self.vector_database_module.metadata = pd.DataFrame([{DOC_TEXT_KEY: "only chunk"}])
        self.mock_model.return_value = [torch.rand((1, 1, 10))]
        self.retriever_module.reranker = None

        self.assertIsNone(self.retriever_module.retrieve(query, return_metadata=True))


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from agrag.modules.vector_db.utils import (
    cosine_similarity_fn,
    euclidean_similarity_fn,
    load_index,
    manhattan_similarity_fn,
    pad_embeddings,
    remove_duplicates,
    save_index,
)
from agrag.modules.vector_db.vector_database import VectorDatabaseModule


class TestVectorDatabaseModule(unittest.TestCase):
    def setUp(self):
        self.embedding1 = torch.rand(1, 10)
        self.embedding2 = torch.rand(1, 10)
        self.embedding_duplicate1 = self.embedding1.clone()
        self.embedding_duplicate2 = self.embedding2.clone()

        self.embeddings = [self.embedding1, self.embedding2, self.embedding_duplicate1, self.embedding_duplicate2]
        self.embeddings.extend([torch.rand(1, 10) for _ in range(6)])
        self.vector_db_module = VectorDatabaseModule(
            db_type="faiss", params={"gpu": False}, similarity_threshold=0.95, similarity_fn="cosine"
        )
        self.index_path = "test_index_path"

    def tearDown(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    @patch("agrag.modules.vector_db.faiss.faiss.construct_faiss_index")
    def test_construct_vector_database(self, mock_construct_faiss_index):
        mock_construct_faiss_index.return_value = MagicMock()
        index = self.vector_db_module.construct_vector_database(self.embeddings)
        self.assertIsNotNone(index)

    def test_cosine_similarity_fn(self):
        self.embeddings = pad_embeddings(self.embeddings)
        similarity_matrix = cosine_similarity_fn(self.embeddings)
        self.assertEqual(similarity_matrix.shape, (10, 10))

    def test_euclidean_similarity_fn(self):
        self.embeddings = pad_embeddings(self.embeddings)
        similarity_matrix = euclidean_similarity_fn(self.embeddings)
        self.assertEqual(similarity_matrix.shape, (10, 10))

    def test_manhattan_similarity_fn(self):
        self.embeddings = pad_embeddings(self.embeddings)
        similarity_matrix = manhattan_similarity_fn(self.embeddings)
        self.assertEqual(similarity_matrix.shape, (10, 10))

    def test_remove_duplicates(self):
        print(type(self.embeddings))
        deduplicated_embeddings = remove_duplicates(self.embeddings, 0.99, "cosine")
        self.assertLessEqual(len(deduplicated_embeddings), 8)  # 2 similar embeddings

        deduplicated_set = {tuple(embedding.flatten().tolist()) for embedding in deduplicated_embeddings}

        self.assertNotIn(tuple(self.embedding_duplicate1), deduplicated_set)
        self.assertNotIn(tuple(self.embedding_duplicate2), deduplicated_set)

    def test_pad_embeddings(self):
        padded_embeddings = pad_embeddings(self.embeddings)
        self.assertEqual(padded_embeddings.shape, (10, 10))

    @patch("agrag.modules.vector_db.utils.save_faiss_index")
    def test_save_index(self, mock_save_faiss_index):
        mock_index = MagicMock()
        index_path = self.index_path
        save_index("faiss", mock_index, index_path)
        mock_save_faiss_index.assert_called_once_with(mock_index, index_path)

    @patch("agrag.modules.vector_db.utils.load_faiss_index")
    def test_load_index(self, mock_load_faiss_index):
        mock_index = MagicMock()
        mock_load_faiss_index.return_value = mock_index
        index_path = self.index_path
        index = load_index("faiss", index_path)
        self.assertEqual(index, mock_index)
        mock_load_faiss_index.assert_called_once_with(index_path)


if __name__ == "__main__":
    unittest.main()

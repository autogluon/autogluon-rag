import os
import unittest
from unittest.mock import MagicMock, patch

import boto3
import torch

from agrag.modules.vector_db.faiss.faiss_db import construct_faiss_index, load_faiss_index, save_faiss_index


class TestFaissDB(unittest.TestCase):
    def setUp(self):
        self.embeddings = [torch.rand(1, 10) for _ in range(10)]
        self.index_path = "test_faiss_index_path"
        self.s3_bucket = "test_bucket"
        self.s3_client = MagicMock(spec=boto3.client("s3"))

    def tearDown(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    @patch("faiss.IndexFlatL2")
    def test_construct_faiss_index_flat(self, mock_index_flat_l2):
        mock_index = MagicMock()
        mock_index_flat_l2.return_value = mock_index
        mock_index.ntotal = len(self.embeddings)
        index = construct_faiss_index(
            self.embeddings, embedding_dim=self.embeddings[0].shape[-1], num_gpus=0, index_type="IndexFlatL2"
        )
        self.assertEqual(index, mock_index)

    @patch("faiss.IndexIVFPQ")
    def test_construct_faiss_index_ivfpq(self, mock_index_ivfpq):
        mock_index = MagicMock()
        mock_index_ivfpq.return_value = mock_index
        mock_index.ntotal = len(self.embeddings)
        mock_index.is_trained = True
        quantized_params = {"nlist": 10, "m": 8, "nbits": 8}
        index = construct_faiss_index(
            self.embeddings,
            embedding_dim=self.embeddings[0].shape[-1],
            num_gpus=0,
            index_type="IndexIVFPQ",
            faiss_quantized_index_params=quantized_params,
        )
        self.assertEqual(index, mock_index)

    @patch("faiss.IndexIVFFlat")
    def test_construct_faiss_index_ivfflat(self, mock_index_ivfflat):
        mock_index = MagicMock()
        mock_index_ivfflat.return_value = mock_index
        mock_index.ntotal = len(self.embeddings)
        mock_index.is_trained = True
        clustered_params = {"nlist": 10}
        index = construct_faiss_index(
            self.embeddings,
            embedding_dim=self.embeddings[0].shape[-1],
            num_gpus=0,
            index_type="IndexIVFFlat",
            faiss_clustered_index_params=clustered_params,
        )
        self.assertEqual(index, mock_index)

    @patch("faiss.write_index")
    def test_save_faiss_index(self, mock_write_index):
        mock_index = MagicMock()
        index_path = self.index_path
        result = save_faiss_index(mock_index, index_path)
        mock_write_index.assert_called_once_with(mock_index, index_path)
        self.assertTrue(result)

    @patch("faiss.write_index")
    def test_save_faiss_index_failure(self, mock_write_index):
        mock_write_index.side_effect = IOError("Failed to write index")
        mock_index = MagicMock()
        index_path = self.index_path
        result = save_faiss_index(mock_index, index_path)
        self.assertFalse(result)

    @patch("faiss.read_index")
    def test_load_faiss_index(self, mock_read_index):
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index
        index_path = self.index_path
        index = load_faiss_index(index_path)
        self.assertEqual(index, mock_index)
        mock_read_index.assert_called_once_with(index_path)

    @patch("faiss.read_index")
    def test_load_faiss_index_failure(self, mock_read_index):
        mock_read_index.side_effect = IOError("Failed to read index")
        index_path = self.index_path
        index = load_faiss_index(index_path)
        self.assertIsNone(index)


if __name__ == "__main__":
    unittest.main()

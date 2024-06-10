import os
import unittest
from unittest.mock import MagicMock, patch

import boto3
import torch
from botocore.exceptions import NoCredentialsError

from agrag.modules.vector_db.faiss.faiss_db import (
    construct_faiss_index,
    load_faiss_index,
    load_faiss_index_s3,
    save_faiss_index,
    save_faiss_index_s3,
)


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
    def test_construct_faiss_index(self, mock_index_flat_l2):
        mock_index = MagicMock()
        mock_index_flat_l2.return_value = mock_index
        index = construct_faiss_index(self.embeddings, gpu=False)
        self.assertEqual(index, mock_index)

    @patch("faiss.write_index")
    def test_save_faiss_index(self, mock_write_index):
        mock_index = MagicMock()
        index_path = self.index_path
        save_faiss_index(mock_index, index_path)
        mock_write_index.assert_called_once_with(mock_index, index_path)

    @patch("faiss.read_index")
    def test_load_faiss_index(self, mock_read_index):
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index
        index_path = self.index_path
        index = load_faiss_index(index_path)
        self.assertEqual(index, mock_index)
        mock_read_index.assert_called_once_with(index_path)

    def test_save_faiss_index_s3(self):
        save_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.s3_client.upload_file.assert_called_once_with(
            Filename=self.index_path, Bucket=self.s3_bucket, Key=self.index_path
        )

    def test_load_faiss_index_s3(self):
        load_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.s3_client.download_file.assert_called_once_with(
            Filename=self.index_path, Bucket=self.s3_bucket, Key=self.index_path
        )

    def test_save_faiss_index_s3_no_credentials(self):
        self.s3_client.upload_file.side_effect = NoCredentialsError()
        with self.assertRaises(NoCredentialsError):
            save_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)

    def test_load_faiss_index_s3_no_credentials(self):
        self.s3_client.download_file.side_effect = NoCredentialsError()
        with self.assertRaises(NoCredentialsError):
            load_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)


if __name__ == "__main__":
    unittest.main()

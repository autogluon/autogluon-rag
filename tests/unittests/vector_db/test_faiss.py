import os
import unittest
from unittest.mock import MagicMock, patch

import boto3
import torch
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

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
        index = construct_faiss_index(self.embeddings, num_gpus=0)
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

    def test_save_faiss_index_s3(self):
        result = save_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.s3_client.upload_file.assert_called_once_with(
            Filename=self.index_path, Bucket=self.s3_bucket, Key=self.index_path
        )
        self.assertTrue(result)

    def test_save_faiss_index_s3_no_credentials(self):
        self.s3_client.upload_file.side_effect = NoCredentialsError()
        result = save_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.assertFalse(result)

    def test_save_faiss_index_s3_partial_credentials(self):
        self.s3_client.upload_file.side_effect = PartialCredentialsError(provider="test", cred_var="test")
        result = save_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.assertFalse(result)

    def test_save_faiss_index_s3_client_error(self):
        self.s3_client.upload_file.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Test Error"}}, "UploadFile"
        )
        result = save_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.assertFalse(result)

    def test_load_faiss_index_s3(self):
        result = load_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.s3_client.download_file.assert_called_once_with(
            Filename=self.index_path, Bucket=self.s3_bucket, Key=self.index_path
        )
        self.assertTrue(result)

    def test_load_faiss_index_s3_no_credentials(self):
        self.s3_client.download_file.side_effect = NoCredentialsError()
        result = load_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.assertFalse(result)

    def test_load_faiss_index_s3_partial_credentials(self):
        self.s3_client.download_file.side_effect = PartialCredentialsError(provider="test", cred_var="test")
        result = load_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.assertFalse(result)

    def test_load_faiss_index_s3_client_error(self):
        self.s3_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Test Error"}}, "DownloadFile"
        )
        result = load_faiss_index_s3(self.index_path, self.s3_bucket, self.s3_client)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

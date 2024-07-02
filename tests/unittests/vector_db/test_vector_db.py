import os
import unittest
from unittest.mock import MagicMock, patch

import boto3
import faiss
import pandas as pd
import torch

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, EMBEDDING_KEY
from agrag.modules.vector_db.utils import (
    cosine_similarity_fn,
    euclidean_similarity_fn,
    load_index,
    load_metadata,
    manhattan_similarity_fn,
    pad_embeddings,
    remove_duplicates,
    save_index,
    save_metadata,
)
from agrag.modules.vector_db.vector_database import VectorDatabaseModule

torch.manual_seed(0)


class TestVectorDatabaseModule(unittest.TestCase):
    def setUp(self):
        self.embedding1 = torch.rand(1, 10)
        self.embedding2 = torch.rand(1, 10)
        self.embedding3 = torch.rand(1, 8)
        self.embedding4 = torch.rand(1, 6)
        self.embedding_duplicate1 = self.embedding1.clone()
        self.embedding_duplicate2 = self.embedding2.clone()

        self.test_pad_embeddings = [self.embedding3, self.embedding4]

        self.embeddings = [self.embedding1, self.embedding2, self.embedding_duplicate1, self.embedding_duplicate2]
        self.embeddings.extend([torch.rand(1, 10) for _ in range(6)])
        self.vector_db_module = VectorDatabaseModule(
            db_type="faiss", params={"gpu": False}, similarity_threshold=0.95, similarity_fn="cosine"
        )
        self.index_path = "test_index_path"
        self.s3_index_path = "s3://s3_bucket/test_index_path"
        self.metadata_path = "test_metadata_path"
        self.s3_bucket = "bucket"
        self.s3_client = boto3.client("s3")

    def tearDown(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

    @patch("agrag.modules.vector_db.vector_database.construct_faiss_index")
    def test_construct_vector_database(self, mock_construct_faiss_index):
        mock_faiss_index = MagicMock()
        mock_construct_faiss_index.return_value = mock_faiss_index

        embeddings = pd.DataFrame(
            [
                {EMBEDDING_KEY: torch.rand(1, 10).numpy(), DOC_ID_KEY: i, CHUNK_ID_KEY: i, DOC_TEXT_KEY: "some text"}
                for i in range(6)
            ]
        )
        self.vector_db_module.construct_vector_database(embeddings)
        self.assertIsNotNone(self.vector_db_module.index)
        self.assertEqual(len(self.vector_db_module.metadata), len(embeddings))
        metadata = embeddings.drop(columns=[EMBEDDING_KEY])
        pd.testing.assert_frame_equal(self.vector_db_module.metadata, metadata)

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
        deduplicated_embeddings = remove_duplicates(self.embeddings, 0.95, "cosine")
        self.assertLessEqual(len(deduplicated_embeddings), 8)  # 2 similar embeddings

        deduplicated_set = {tuple(embedding) for embedding in deduplicated_embeddings}

        self.assertNotIn(tuple(self.embedding_duplicate1), deduplicated_set)
        self.assertNotIn(tuple(self.embedding_duplicate2), deduplicated_set)

    def test_pad_embeddings(self):
        padded_embeddings = pad_embeddings(self.embeddings)
        max_len = max(embedding.size(1) for embedding in self.embeddings)

        self.assertEqual(padded_embeddings.size(1), max_len * self.embeddings[0].size(0))

        start_idx = 0
        for original_embedding in self.embeddings:
            end_idx = start_idx + original_embedding.size(0)
            self.assertTrue(
                torch.equal(padded_embeddings[start_idx:end_idx, : original_embedding.size(1)], original_embedding)
            )
            if original_embedding.size(1) < max_len:
                self.assertTrue(
                    torch.equal(
                        padded_embeddings[start_idx:end_idx, original_embedding.size(1) :],
                        torch.zeros(original_embedding.size(0), max_len - original_embedding.size(1)),
                    )
                )
            start_idx = end_idx

    @patch("agrag.modules.vector_db.utils.save_faiss_index")
    def test_save_index(self, mock_save_faiss_index):
        faiss_index = faiss.IndexFlatL2()
        index_path = self.index_path
        save_index("faiss", faiss_index, index_path)
        mock_save_faiss_index.assert_called_once_with(faiss_index, index_path)

    @patch("agrag.modules.vector_db.utils.save_faiss_index_s3")
    @patch("boto3.client")
    def test_save_index_with_s3(self, mock_s3_client, mock_save_faiss_index_s3):
        faiss_index = faiss.IndexFlatL2()
        s3_client = mock_s3_client.return_value
        index_path = self.index_path
        save_index("faiss", faiss_index, self.s3_index_path)
        mock_save_faiss_index_s3.assert_called_once_with(index_path, "s3_bucket", s3_client)

    @patch("agrag.modules.vector_db.faiss.faiss_db.save_faiss_index")
    def test_save_index_failure(self, mock_save_faiss_index):
        mock_save_faiss_index.side_effect = IOError("Failed to write index")
        faiss_index = faiss.IndexFlatL2()
        index_path = self.index_path
        result = save_index("faiss", faiss_index, index_path)
        self.assertFalse(result)

    @patch("agrag.modules.vector_db.utils.load_faiss_index")
    def test_load_index(self, mock_load_faiss_index):
        mock_index = MagicMock()
        mock_load_faiss_index.return_value = mock_index
        index_path = self.index_path
        index = load_index("faiss", index_path)
        self.assertEqual(index, mock_index)
        mock_load_faiss_index.assert_called_once_with(index_path)

    @patch("agrag.modules.vector_db.utils.load_faiss_index_s3")
    @patch("agrag.modules.vector_db.utils.load_faiss_index")
    @patch("boto3.client")
    def test_load_index_with_s3(self, mock_s3_client, mock_load_faiss_index, mock_load_faiss_index_s3):
        mock_index = MagicMock()
        mock_load_faiss_index.return_value = mock_index
        s3_client = mock_s3_client.return_value
        index_path = self.index_path
        index = load_index("faiss", self.s3_index_path)
        self.assertEqual(index, mock_index)
        mock_load_faiss_index_s3.assert_called_once_with(index_path, "s3_bucket", s3_client)
        mock_load_faiss_index.assert_called_once_with(index_path)

    @patch("agrag.modules.vector_db.faiss.faiss_db.load_faiss_index")
    def test_load_index_failure(self, mock_load_faiss_index):
        mock_load_faiss_index.side_effect = IOError("Failed to read index")
        index_path = self.index_path
        index = load_index("faiss", index_path)
        self.assertIsNone(index)

    @patch("pandas.DataFrame.to_json")
    @patch("os.makedirs")
    def test_save_metadata(self, mock_makedirs, mock_to_json):
        metadata = pd.DataFrame([{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0}, {DOC_ID_KEY: 1, CHUNK_ID_KEY: 1}])
        metadata_path = "test_metadata_path"
        save_metadata(metadata, metadata_path)

        mock_makedirs.assert_called_once_with(os.path.dirname(metadata_path))
        mock_to_json.assert_called_once_with(metadata_path, orient="records", lines=True)

    @patch("pandas.DataFrame.to_json")
    @patch("os.makedirs")
    @patch("boto3.client")
    def test_save_metadata_s3(self, mock_boto_client, mock_makedirs, mock_to_json):
        metadata = pd.DataFrame([{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0}, {DOC_ID_KEY: 1, CHUNK_ID_KEY: 1}])
        metadata_path = "metadata"
        s3_metadata_path = "s3://s3_bucket/metadata"
        mock_s3_client = mock_boto_client.return_value
        save_metadata(metadata, s3_metadata_path)

        mock_makedirs.assert_called_once_with(os.path.dirname(s3_metadata_path))
        mock_to_json.assert_called_once_with(s3_metadata_path, orient="records", lines=True)
        mock_s3_client.upload_file.assert_called_once_with(
            Filename=metadata_path, Bucket="s3_bucket", Key=metadata_path
        )

    @patch("pandas.read_json")
    def test_load_metadata(self, mock_read_json):
        mock_read_json.return_value = pd.DataFrame(
            [{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0}, {DOC_ID_KEY: 1, CHUNK_ID_KEY: 1}]
        )
        metadata_path = "test_metadata_path"
        metadata = load_metadata(metadata_path)

        mock_read_json.assert_called_once_with(metadata_path, orient="records", lines=True)
        pd.testing.assert_frame_equal(
            metadata, pd.DataFrame([{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0}, {DOC_ID_KEY: 1, CHUNK_ID_KEY: 1}])
        )

    @patch("pandas.read_json")
    @patch("boto3.client")
    def test_load_metadata_s3(self, mock_boto_client, mock_read_json):
        mock_read_json.return_value = pd.DataFrame(
            [{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0}, {DOC_ID_KEY: 1, CHUNK_ID_KEY: 1}]
        )
        mock_s3_client = mock_boto_client.return_value
        metadata_path = "metadata"
        s3_metadata_path = "s3://s3_bucket/metadata"
        metadata = load_metadata(s3_metadata_path)

        mock_s3_client.download_file.assert_called_once_with(
            Filename=metadata_path, Bucket="s3_bucket", Key=metadata_path
        )
        mock_read_json.assert_called_once_with(metadata_path, orient="records", lines=True)
        pd.testing.assert_frame_equal(
            metadata, pd.DataFrame([{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0}, {DOC_ID_KEY: 1, CHUNK_ID_KEY: 1}])
        )


if __name__ == "__main__":
    unittest.main()

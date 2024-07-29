import unittest
from unittest.mock import patch

import torch

from ragify.modules.vector_db.milvus.milvus_db import construct_milvus_index, load_milvus_index


class TestMilvusInterface(unittest.TestCase):
    @patch("ragify.modules.vector_db.milvus.milvus_db.MilvusClient")
    def test_construct_milvus_index(self, MockMilvusClient):
        mock_client = MockMilvusClient.return_value

        embeddings = [torch.rand(128) for _ in range(10)]
        collection_name = "test_collection"
        db_name = "test_db"
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"param": 1}}
        create_params = {"param": 0}

        mock_client.has_collection.return_value = True
        client = construct_milvus_index(
            embeddings=embeddings,
            collection_name=collection_name,
            db_name=db_name,
            index_params=index_params,
            create_params=create_params,
            embedding_dim=embeddings[0].shape[-1],
        )

        client.create_collection.assert_called_once_with(
            collection_name=collection_name,
            dimension=128,
            vector_field_name="embedding",
            index_params=index_params,
            **create_params,
        )
        self.assertEqual(mock_client.insert.call_count, 1)
        self.assertEqual(len(mock_client.insert.call_args[1]["data"]), 10)

    @patch("ragify.modules.vector_db.milvus.milvus_db.MilvusClient")
    def test_load_milvus_index(self, MockMilvusClient):
        mock_client_instance = MockMilvusClient.return_value
        index_path = "valid_path.db"

        client = load_milvus_index(index_path)

        MockMilvusClient.assert_called_once_with(index_path)
        self.assertEqual(client, mock_client_instance)

    @patch("ragify.modules.vector_db.milvus.milvus_db.MilvusClient", side_effect=Exception("Error loading index"))
    def test_load_milvus_index_exception(self, MockMilvusClient):
        index_path = "invalid_path.db"

        client = load_milvus_index(index_path)

        MockMilvusClient.assert_called_once_with(index_path)
        self.assertIsNone(client)


if __name__ == "__main__":
    unittest.main()

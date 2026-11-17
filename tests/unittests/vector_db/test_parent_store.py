import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agrag.constants import DOC_ID_KEY, DOC_TEXT_KEY, PARENT_ID_KEY
from agrag.modules.vector_db.utils import _parent_store_path, load_parent_store, save_parent_store


def _parent_store():
    return pd.DataFrame(
        [
            {PARENT_ID_KEY: 0, DOC_ID_KEY: 0, DOC_TEXT_KEY: "parent zero text"},
            {PARENT_ID_KEY: 1, DOC_ID_KEY: 1, DOC_TEXT_KEY: "parent one text"},
        ]
    )


class TestParentStoreLocalPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip_beside_metadata(self):
        # The parent store persists at a sibling path next to the metadata file
        # and round-trips unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = os.path.join(tmp, "metadata.jsonl")
            store = _parent_store()

            self.assertTrue(save_parent_store(store, metadata_path))
            parent_path = _parent_store_path(metadata_path)
            # It sits beside metadata, not on top of it.
            self.assertTrue(os.path.isfile(parent_path))
            self.assertNotEqual(parent_path, metadata_path)
            self.assertEqual(os.path.dirname(parent_path), os.path.dirname(metadata_path))

            loaded = load_parent_store(metadata_path)
            pd.testing.assert_frame_equal(loaded, store)

    def test_save_none_or_empty_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = os.path.join(tmp, "metadata.jsonl")
            self.assertFalse(save_parent_store(None, metadata_path))
            self.assertFalse(save_parent_store(pd.DataFrame(), metadata_path))
            self.assertFalse(os.path.isfile(_parent_store_path(metadata_path)))

    def test_load_missing_parent_store_returns_none(self):
        # Legacy compatibility: an index whose metadata has no sibling parent
        # store loads as None rather than raising.
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = os.path.join(tmp, "metadata.jsonl")
            pd.DataFrame([{DOC_ID_KEY: 0}]).to_json(metadata_path, orient="records", lines=True)
            self.assertIsNone(load_parent_store(metadata_path))


class TestParentStoreS3Persistence(unittest.TestCase):
    def setUp(self):
        self.s3_metadata_path = "s3://s3_bucket/vectordb/metadata.jsonl"
        # parse_path strips the bucket; the local/key path is bucket-relative.
        self.expected_parent_key = "vectordb/metadata.parents.jsonl"

    def test_save_parent_store_s3_uses_parse_path_and_boto3(self):
        # An s3:// path must upload via boto3 to the bucket-relative key and must
        # never create a literal local "s3:" directory.
        with patch("agrag.modules.vector_db.utils.boto3.client") as mock_client, patch(
            "pandas.DataFrame.to_json"
        ) as mock_to_json:
            mock_s3 = mock_client.return_value
            result = save_parent_store(_parent_store(), self.s3_metadata_path)

        self.assertTrue(result)
        mock_to_json.assert_called_once_with(self.expected_parent_key, orient="records", lines=True)
        mock_s3.upload_file.assert_called_once_with(
            Filename=self.expected_parent_key, Bucket="s3_bucket", Key=self.expected_parent_key
        )
        # No literal "s3:" directory leaked into the local filesystem.
        self.assertFalse(os.path.exists("s3:"))

    def test_load_parent_store_s3_uses_parse_path_and_boto3(self):
        with patch("agrag.modules.vector_db.utils.boto3.client") as mock_client, patch(
            "pandas.read_json"
        ) as mock_read_json:
            mock_s3 = mock_client.return_value
            mock_read_json.return_value = _parent_store()
            loaded = load_parent_store(self.s3_metadata_path)

        mock_s3.download_file.assert_called_once_with(
            Filename=self.expected_parent_key, Bucket="s3_bucket", Key=self.expected_parent_key
        )
        mock_read_json.assert_called_once_with(self.expected_parent_key, orient="records", lines=True)
        pd.testing.assert_frame_equal(loaded, _parent_store())
        self.assertFalse(os.path.exists("s3:"))


if __name__ == "__main__":
    unittest.main()

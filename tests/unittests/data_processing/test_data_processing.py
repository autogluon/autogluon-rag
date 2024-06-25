import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY
from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.data_processing.utils import download_directory_from_s3, get_all_file_paths

CURRENT_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(CURRENT_DIR, "../../test_docs/")


class TestDataProcessingModule(unittest.TestCase):
    @patch("langchain_community.document_loaders.PyPDFLoader.load_and_split")
    def test_process_file(self, mock_pdf_loader):
        mock_page = MagicMock()
        mock_page.page_content = "This is a test page."
        mock_pdf_loader.return_value = [mock_page]

        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None
        )

        result = data_processing_module.process_file(os.path.join(TEST_DIR, "Chatbot.pdf"), doc_id=1)

        expected_result = pd.DataFrame([{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page."}])
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("os.listdir")
    @patch("langchain_community.document_loaders.PyPDFLoader.load_and_split")
    @patch("concurrent.futures.ThreadPoolExecutor.map")
    def test_process_data(self, mock_thread_map, mock_pdf_loader, mock_listdir):
        mock_listdir.return_value = ["sample.pdf"]

        mock_page = MagicMock()
        mock_page.page_content = "This is a test page."
        mock_pdf_loader.return_value = [mock_page]

        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None
        )

        mock_thread_map.return_value = [
            pd.DataFrame([{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page."}])
        ]

        data = data_processing_module.process_data()

        expected_data = pd.DataFrame([{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page."}])
        pd.testing.assert_frame_equal(data, expected_data)

    def test_chunk_data_naive(self):
        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None
        )
        text = "This is a test document to check the chunking method."

        data = data_processing_module.chunk_data_naive(text)

        expected_data = ["This is a ", "test docum", "ent to che", "ck the chu", "nking meth", "od."]
        self.assertEqual(data, expected_data)

    @patch("boto3.client")
    @patch("langchain_community.document_loaders.PyPDFLoader.load_and_split")
    def test_process_file_from_s3(self, mock_pdf_loader, mock_boto_client):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"This is a test page.")
            tmp_file_path = tmp_file.name

        mock_page = MagicMock()
        mock_page.page_content = "This is a test page."
        mock_pdf_loader.return_value = [mock_page]

        mock_s3_client = mock_boto_client.return_value
        mock_s3_client.download_file.side_effect = lambda Bucket, Key, Filename: os.rename(tmp_file_path, Filename)

        data_processing_module = DataProcessingModule(
            data_dir="test_docs/", s3_bucket="autogluon-rag-github-dev", chunk_size=10, chunk_overlap=5
        )

        mock_s3_key = "test_docs/Chatbot.pdf"
        result = data_processing_module.process_file(f"s3://autogluon-rag-github-dev/{mock_s3_key}", doc_id=1)

        expected_result = pd.DataFrame([{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page."}])
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("boto3.client")
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("os.path.dirname")
    @patch("os.path.relpath")
    def test_download_directory_from_s3(
        self, mock_relpath, mock_dirname, mock_exists, mock_makedirs, mock_boto_client
    ):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "path/to/files/file1.txt"},
                {"Key": "path/to/files/file2.txt"},
            ]
        }

        mock_exists.return_value = False

        mock_relpath.side_effect = lambda s3_path, data_dir: s3_path.replace(data_dir, "")
        mock_dirname.side_effect = lambda local_file_path: os.path.split(local_file_path)[0]

        local_dir = download_directory_from_s3("my-s3-bucket", "path/to/files", mock_s3_client)

        self.assertEqual(local_dir, "s3_docs")
        mock_s3_client.list_objects_v2.assert_called_once_with(Bucket="my-s3-bucket", Prefix="path/to/files")
        mock_s3_client.download_file.assert_any_call("my-s3-bucket", "path/to/files/file1.txt", "/file1.txt")
        mock_s3_client.download_file.assert_any_call("my-s3-bucket", "path/to/files/file2.txt", "/file2.txt")
        self.assertEqual(mock_s3_client.download_file.call_count, 2)

        mock_makedirs.assert_any_call("s3_docs")
        mock_makedirs.assert_any_call("s3_docs")

    def test_get_all_file_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create some nested directories and files
            os.makedirs(os.path.join(tmp_dir, "subdir1"))
            os.makedirs(os.path.join(tmp_dir, "subdir2"))
            file1 = os.path.join(tmp_dir, "file1.pdf")
            file2 = os.path.join(tmp_dir, "subdir1", "file2.pdf")
            file3 = os.path.join(tmp_dir, "subdir2", "file3.pdf")
            with open(file1, "w") as f:
                f.write("Test file 1")
            with open(file2, "w") as f:
                f.write("Test file 2")
            with open(file3, "w") as f:
                f.write("Test file 3")

            file_paths = get_all_file_paths(tmp_dir, [".pdf"])

            expected_paths = [file1, file2, file3]
            self.assertCountEqual(file_paths, expected_paths)


if __name__ == "__main__":
    unittest.main()

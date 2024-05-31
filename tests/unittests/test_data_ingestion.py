import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

from agrag.modules.data_processing.data_processing import DataProcessingModule

CURRENT_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(CURRENT_DIR, "../test_docs/")


class TestDataProcessingModule(unittest.TestCase):
    @patch("langchain_community.document_loaders.PyPDFLoader.load_and_split")
    def test_process_file(self, mock_pdf_loader):
        mock_page = MagicMock()
        mock_page.page_content = "This is a test page."
        mock_pdf_loader.return_value = [mock_page]

        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None
        )

        data = data_processing_module.process_file(os.path.join(TEST_DIR, "Chatbot.pdf"))

        expected_data = ["This is a test page."]
        self.assertEqual(data, expected_data)

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

        mock_thread_map.return_value = [["This is a test page."]]

        data = data_processing_module.process_data()

        expected_data = ["This is a test page."]
        self.assertEqual(data, expected_data)

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
            data_dir="test_docs/", s3_bucket="yash-autogluon-rag-dev", chunk_size=10, chunk_overlap=5
        )

        mock_s3_key = "test_docs/Chatbot.pdf"
        data = data_processing_module.process_file(f"s3://yash-autogluon-rag-dev/{mock_s3_key}")

        expected_data = ["This is a test page."]
        self.assertEqual(data, expected_data)


if __name__ == "__main__":
    unittest.main()

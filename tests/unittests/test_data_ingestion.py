import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from agrag.modules.data_ingestion.data_ingestion import DataIngestionModule

CURRENT_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(CURRENT_DIR, "../test_docs/")


class TestDataIngestionModule(unittest.TestCase):
    @patch("langchain_community.document_loaders.PyPDFLoader.load_and_split")
    def test_process_file(self, mock_pdf_loader):
        mock_page = MagicMock()
        mock_page.page_content = "This is a test page."
        mock_pdf_loader.return_value = [mock_page]

        data_ingestion_module = DataIngestionModule(data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5)

        data = data_ingestion_module.process_file(os.path.join(TEST_DIR, "Chatbot.pdf"))

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

        data_ingestion_module = DataIngestionModule(data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5)

        mock_thread_map.return_value = [["This is a test page."]]

        data = data_ingestion_module.process_data()

        expected_data = ["This is a test page."]
        self.assertEqual(data, expected_data)

    def test_chunk_data_naive(self):
        data_ingestion_module = DataIngestionModule(data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5)
        text = "This is a test document to check the chunking method."

        data = data_ingestion_module.chunk_data_naive(text)

        expected_data = ["This is a ", "test docum", "ent to che", "ck the chu", "nking meth", "od."]
        self.assertEqual(data, expected_data)


if __name__ == "__main__":
    unittest.main()

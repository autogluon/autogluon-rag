import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, PARENT_ID_KEY, SOURCE_KEY
from agrag.modules.data_processing.data_processing import DataProcessingModule
from agrag.modules.data_processing.utils import bs4_extractor, download_directory_from_s3, get_all_file_paths

CURRENT_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(CURRENT_DIR, "../../test_docs/")


class TestDataProcessingModule(unittest.TestCase):
    @patch("langchain_community.document_loaders.PyPDFLoader.load_and_split")
    def test_process_file(self, mock_pdf_loader):
        mock_page = MagicMock()
        mock_page.page_content = "This is a test page."
        mock_pdf_loader.return_value = [mock_page]

        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None, web_urls=[]
        )

        file_path = os.path.join(TEST_DIR, "test_file.pdf")
        result = data_processing_module.process_file(file_path, doc_id=1)

        expected_result = pd.DataFrame(
            [{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page.", SOURCE_KEY: file_path}]
        )
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
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None, web_urls=[]
        )

        mock_thread_map.return_value = [
            pd.DataFrame([{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page."}])
        ]

        data = data_processing_module.process_data()

        expected_data = pd.DataFrame([{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page."}])
        pd.testing.assert_frame_equal(data, expected_data)

    def test_process_data_url_only(self):
        # URL-only ingestion: process_urls returns (dataframe, last_doc_id) and
        # process_data must unpack it, concatenating only the DataFrame.
        data_processing_module = DataProcessingModule(
            data_dir=None, chunk_size=10, chunk_overlap=5, s3_bucket=None, web_urls=["http://example.com"]
        )

        url_df = pd.DataFrame(
            [{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "url chunk", SOURCE_KEY: "http://example.com"}]
        )
        with patch.object(data_processing_module, "process_urls", return_value=(url_df, 1)) as mock_urls:
            data = data_processing_module.process_data()

        # URLs start at doc_id 0 when there are no files.
        _, kwargs = mock_urls.call_args
        self.assertEqual(kwargs["start_doc_id"], 0)
        pd.testing.assert_frame_equal(data, url_df)

    def test_process_data_mixed_files_and_urls_continuous_doc_ids(self):
        # Mixed ingestion: file doc IDs are assigned first, and URL processing
        # must start where files left off so document IDs stay continuous.
        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None, web_urls=["http://example.com"]
        )

        files_df = pd.DataFrame(
            [
                {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "file doc 0", SOURCE_KEY: "a.pdf"},
                {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "file doc 1", SOURCE_KEY: "b.pdf"},
            ]
        )
        urls_df = pd.DataFrame(
            [{DOC_ID_KEY: 2, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "url doc 2", SOURCE_KEY: "http://example.com"}]
        )

        with patch.object(
            data_processing_module, "process_files", return_value=(files_df, 2)
        ), patch.object(data_processing_module, "process_urls", return_value=(urls_df, 3)) as mock_urls:
            data = data_processing_module.process_data()

        # URLs must continue from the last file doc_id (2), not restart at 0.
        _, kwargs = mock_urls.call_args
        self.assertEqual(kwargs["start_doc_id"], 2)
        # The concatenated frame carries continuous, unique document IDs.
        self.assertEqual(sorted(data[DOC_ID_KEY].tolist()), [0, 1, 2])
        self.assertEqual(len(data), 3)

    def test_chunk_data_naive(self):
        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None, web_urls=[]
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
            data_dir="test_docs/", s3_bucket="autogluon-rag-github-dev", chunk_size=10, chunk_overlap=5, web_urls=[]
        )

        mock_s3_key = "test_docs/test_file.pdf"
        s3_path = f"s3://autogluon-rag-github-dev/{mock_s3_key}"
        result = data_processing_module.process_file(s3_path, doc_id=1)

        expected_result = pd.DataFrame(
            [{DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page.", SOURCE_KEY: s3_path}]
        )
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

    def test_unsupported_file_extension(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "unsupported_file.unsupported")
            with open(file_path, "w") as f:
                f.write("This is a test file with an unsupported file extension.")

            file_paths = get_all_file_paths(tmp_dir, file_exts=[".pdf"])

            self.assertEqual(len(file_paths), 0)

    @patch("agrag.modules.data_processing.utils.RecursiveCharacterTextSplitter.split_text")
    def test_process_url(self, mock_url_loader):
        mock_url_loader.return_value = ["This is a test page from a URL."]

        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, web_urls=["http://example.com"]
        )

        result = data_processing_module.process_url("http://example.com", doc_id=1)

        expected_result = pd.DataFrame(
            [
                {
                    DOC_ID_KEY: 1,
                    CHUNK_ID_KEY: 0,
                    DOC_TEXT_KEY: "This is a test page from a URL.",
                    SOURCE_KEY: "http://example.com",
                }
            ]
        )
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("agrag.modules.data_processing.utils.RecursiveUrlLoader.load_and_split")
    @patch("concurrent.futures.ThreadPoolExecutor.map")
    def test_process_urls(self, mock_thread_map, mock_url_loader):
        mock_page = MagicMock()
        mock_page.page_content = "This is a test page from a URL."
        mock_url_loader.return_value = [mock_page]

        data_processing_module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, web_urls=["http://example.com", "http://example.org"]
        )

        mock_thread_map.return_value = [
            pd.DataFrame([{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page from a URL."}]),
        ]

        data = data_processing_module.process_urls(data_processing_module.web_urls, start_doc_id=0)[0]

        expected_data = pd.DataFrame(
            [{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "This is a test page from a URL."}]
        )
        pd.testing.assert_frame_equal(data, expected_data)

    def _batch_frame(self, doc_ids):
        # One chunk per (doc_id, chunk_id) pair for parent-child grouping tests.
        rows = []
        for doc_id in doc_ids:
            for chunk_id in range(2):
                rows.append(
                    {DOC_ID_KEY: doc_id, CHUNK_ID_KEY: chunk_id, DOC_TEXT_KEY: f"d{doc_id}c{chunk_id}"}
                )
        return pd.DataFrame(rows)

    def test_build_parent_child_groups_and_tags(self):
        module = DataProcessingModule(
            data_dir=TEST_DIR,
            chunk_size=10,
            chunk_overlap=5,
            s3_bucket=None,
            web_urls=[],
            chunking_strategy="parent_child",
            children_per_parent=2,
        )
        df = self._batch_frame([0, 1])  # 2 docs x 2 chunks, 2 children/parent
        out = module.build_parent_child(df)

        # Each child is tagged with a parent_id and one parent exists per doc.
        self.assertIn(PARENT_ID_KEY, out.columns)
        self.assertEqual(out[PARENT_ID_KEY].tolist(), [0, 0, 1, 1])
        self.assertEqual(len(module.parent_store), 2)
        # Parent text concatenates its children in order.
        p0 = module.parent_store.loc[module.parent_store[PARENT_ID_KEY] == 0, DOC_TEXT_KEY].iloc[0]
        self.assertEqual(p0, "d0c0\nd0c1")

    def test_build_parent_child_unique_ids_and_accumulation_across_batches(self):
        # Simulates batched processing: build_parent_child is called once per
        # batch. parent_id must stay globally unique and the store must accumulate.
        module = DataProcessingModule(
            data_dir=TEST_DIR,
            chunk_size=10,
            chunk_overlap=5,
            s3_bucket=None,
            web_urls=[],
            chunking_strategy="parent_child",
            children_per_parent=2,
        )

        batch1 = module.build_parent_child(self._batch_frame([0, 1]))
        batch2 = module.build_parent_child(self._batch_frame([2, 3]))

        # Batch 1 owns parent_ids {0,1}; batch 2 continues at {2,3} (no reuse).
        self.assertEqual(sorted(batch1[PARENT_ID_KEY].unique().tolist()), [0, 1])
        self.assertEqual(sorted(batch2[PARENT_ID_KEY].unique().tolist()), [2, 3])
        # The store accumulated all four parents with globally unique IDs.
        self.assertEqual(sorted(module.parent_store[PARENT_ID_KEY].tolist()), [0, 1, 2, 3])
        self.assertEqual(len(module.parent_store), 4)

    def test_legacy_strategy_builds_no_parent_store(self):
        module = DataProcessingModule(
            data_dir=TEST_DIR, chunk_size=10, chunk_overlap=5, s3_bucket=None, web_urls=[]
        )
        # Default strategy is legacy/flat: no parent store is ever created.
        self.assertEqual(module.chunking_strategy, "legacy")
        self.assertIsNone(module.parent_store)

    def test_bs4_extractor(self):
        html_content = """
        <html>
            <body>
                <p>This is a paragraph.</p>
                <div>This is a div.</div>
                <p>Another paragraph.</p>
                <table>
                    <tr><td>1</td><td>2</td></tr>
                </table>
            </body>
        </html>
        """

        extracted_text = bs4_extractor(html_content, tags_to_extract=["p"])
        expected_text = "This is a paragraph.\nAnother paragraph."
        self.assertEqual(extracted_text, expected_text)

        extracted_text = bs4_extractor(html_content, tags_to_extract=["table"])
        expected_text = "1 2"
        self.assertEqual(extracted_text, expected_text)

        extracted_text = bs4_extractor(html_content, tags_to_extract=["p", "table"])
        expected_text = "This is a paragraph.\nAnother paragraph.\n1 2"
        self.assertEqual(extracted_text, expected_text)


if __name__ == "__main__":
    unittest.main()

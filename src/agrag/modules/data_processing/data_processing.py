import concurrent.futures
import logging
import os
from typing import List

import boto3
import pandas as pd

from agrag.constants import SUPPORTED_FILE_EXTENSIONS
from agrag.modules.data_processing.utils import (
    download_directory_from_s3,
    get_all_file_paths,
    process_csv,
    process_docx_doc,
    process_pdf,
    process_rtf,
    process_txt_md_py,
)

logger = logging.getLogger("rag-logger")


class DataProcessingModule:
    """
    A class used to ingest and preprocess documents for use in a Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
    ----------
    data_dir : str
        The directory containing the data files to be ingested.
    chunk_size : int
        The size of each chunk of text.
    chunk_overlap : int
        The overlap between consecutive chunks of text.
    s3_bucket : str
        The name of the S3 bucket containing the data files.
    file_exts: List[str]
        List of file extensions to support.
    **kwargs : dict
        Additional parameters for `DataProcessingModule`.

    Methods:
    -------
    chunk_data_naive(text: str) -> List[str]:
        Naively chunks text into segments of a specified size without any overlap.

    chunk_data(text: str) -> List[str]:
        Chunks text into segments using a specified overlap.

    process_file(file_path: str, doc_id: int) -> pd.DataFrame:
        Processes a single file, extracting and chunking the text.

    process_data() -> pd.DataFrame:
        Processes all files in the data directory, extracting and chunking text from each file, and compiles the results into a single DataFrame.
    """

    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.chunk_size = kwargs.get("chunk_size")
        self.chunk_overlap = kwargs.get("chunk_overlap")
        self.s3_bucket = kwargs.get("s3_bucket")
        self.s3_client = boto3.client("s3") if self.s3_bucket else None
        self.file_exts = kwargs.get("file_exts", SUPPORTED_FILE_EXTENSIONS)

    def chunk_data_naive(self, text: str) -> List[str]:
        """
        Naively chunks text into segments of a specified size without any overlap.

        Parameters:
        ----------
        text : str
            The text to be chunked.

        Returns:
        -------
        List[str]
            A list of text chunks.
        """
        chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return chunks

    def chunk_data(self, text: str) -> List[str]:
        """
        Chunks text into segments using a specified overlap.

        Parameters:
        ----------
        text : str
            The text to be chunked.

        Returns:
        -------
        List[str]
            A list of text chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def process_file(self, file_path: str, doc_id: int) -> pd.DataFrame:
        """
        Processes a single file, extracting and chunking the text.

        Parameters:
        ----------
        file_path : str
            The path to the file to be processed.
        doc_id : int
            The document ID.

        Returns:
        -------
        pd.DataFrame
            A table containing processed text chunks and metadata from the given file.
        """
        logger.info(f"Processing File: {file_path}")

        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == ".pdf":
            return process_pdf(file_path, self.chunk_size, self.chunk_overlap, doc_id)
        elif file_extension.lower() in [".txt", ".md", ".py"]:
            return process_txt_md_py(file_path, self.chunk_data, doc_id)
        elif file_extension.lower() in [".docx", ".doc"]:
            return process_docx_doc(file_path, self.chunk_data, doc_id)
        elif file_extension.lower() == ".rtf":
            return process_rtf(file_path, self.chunk_data, doc_id)
        elif file_extension.lower() == ".csv":
            return process_csv(file_path, self.chunk_data, doc_id)

        return pd.DataFrame()

    def process_data(self) -> pd.DataFrame:
        """
        Processes all files in the data directory.

        Extracts and chunks text from each file and compiles the results into a single DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame of processed text chunks from all files in the directory.
        """
        processed_data = []
        if self.s3_bucket:
            self.data_dir = download_directory_from_s3(
                s3_bucket=self.s3_bucket, data_dir=self.data_dir, s3_client=self.s3_client
            )

        file_paths = get_all_file_paths(self.data_dir, self.file_exts)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.process_file, file_paths, range(len(file_paths)))
            for result in results:
                processed_data.append(result)

        return pd.concat(processed_data).reset_index(drop=True)

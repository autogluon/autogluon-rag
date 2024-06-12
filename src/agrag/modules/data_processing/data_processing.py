import concurrent.futures
import logging
import os
from typing import List

import boto3
import textract
from docx import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agrag.modules.data_processing.utils import download_directory_from_s3, get_all_file_paths

logger = logging.getLogger("rag-logger")


class DataProcessingModule:
    """
    A class used to ingest and preprocess documents for use in a Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
    ----------
    data_dir : str
        The directory containing the data files to be ingested.
    chunk_size : int, optional
        The size of each chunk of text (default is 512).
    chunk_overlap : int, optional
        The overlap between consecutive chunks of text (default is 128).
    s3_bucket : str, optional
        The name of the S3 bucket containing the data files.

    Example:
    --------
    data_processing_module = DataProcessingModule(
        data_dir="path/to/files", chunk_size=512, chunk_overlap=128, s3_bucket=my-s3-bucket
    )
    """

    def __init__(self, data_dir: str, chunk_size: int, chunk_overlap: int, s3_bucket: str = None) -> None:
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client("s3") if s3_bucket else None

    def chunk_data_naive(self, text: str):
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

    def chunk_data(self, text: str):
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

    def process_file(self, file_path: str) -> List[str]:
        """
        Processes a single file, extracting and chunking the text.

        Parameters:
        ----------
        file_path : str
            The path to the file to be processed.

        Returns:
        -------
        List[str]
            A list of processed text chunks from the given file.
        """
        logger.info(f"Processing File: {file_path}")
        processed_data = []

        # Determine file extension
        _, file_extension = os.path.splitext(file_path)

        # Process based on file extension
        if file_extension.lower() == ".pdf":
            pdf_loader = PyPDFLoader(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[".", "\uff0e", "\n"],  # \uff0e -> Fullwidth full stop
                length_function=len,
                is_separator_regex=False,
            )
            pages = pdf_loader.load_and_split(text_splitter=text_splitter)
            for page in pages:
                page_content = "".join(page.page_content)
                processed_data.append(page_content)
        elif file_extension.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            processed_data.append(self.chunk_data(text))
        elif file_extension.lower() == ".docx":
            doc = Document(file_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            processed_data.append(self.chunk_data(text))
        elif file_extension.lower() == ".py":
            text = textract.process(file_path, method="text", encoding="utf-8")
            processed_data.append(self.chunk_data(text.decode("utf-8")))

        return processed_data

    def process_data(self) -> List[str]:
        """
        Processes all files in the data directory.

        Extracts and chunks text from each file and compiles the results into a single list.

        Returns:
        -------
        List[str]
            A list of processed text chunks from all files in the directory.
        """
        processed_data = []
        if self.s3_bucket:
            self.data_dir = download_directory_from_s3(
                s3_bucket=self.s3_bucket, data_dir=self.data_dir, s3_client=self.s3_client
            )

        file_paths = get_all_file_paths(self.data_dir)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process each file in parallel
            results = executor.map(self.process_file, file_paths)
            for result in results:
                processed_data.extend(result)
        return processed_data

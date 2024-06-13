import concurrent.futures
import logging
from typing import List

import boto3
import pandas as pd
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
        Chunks text into segments using a more sophisticated approach.

        This method will provide advanced text chunking
        that might include considerations like word boundaries or semantic coherence.

        Parameters:
        ----------
        text : str
            The text to be chunked.

        Returns:
        -------
        List[str]
            A list of text chunks.
        """
        raise NotImplementedError

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
        processed_data = []
        pdf_loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[".", "\uff0e", "\n"],  # \uff0e -> Fullwidth full stop
            length_function=len,
            is_separator_regex=False,
        )
        pages = pdf_loader.load_and_split(text_splitter=text_splitter)
        for chunk_id, page in enumerate(pages):
            page_content = "".join(page.page_content)
            processed_data.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": page_content})
        df = pd.DataFrame(processed_data)
        return df

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
            results = executor.map(self.process_file, file_paths, range(len(file_paths)))
            for result in results:
                processed_data.append(result)

        return pd.concat(processed_data).reset_index(drop=True)

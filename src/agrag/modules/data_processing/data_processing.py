import concurrent.futures
import logging
import os
import re
from typing import List

import boto3
import pandas as pd

from agrag.constants import SUPPORTED_FILE_EXTENSIONS, SUPPORTED_HTML_TAGS
from agrag.modules.data_processing.utils import (
    download_directory_from_s3,
    get_all_file_paths,
    get_text_from_url,
    process_csv,
    process_docx_doc,
    process_pdf,
    process_rtf,
    process_txt_md_py_log,
)
from agrag.utils import parse_path

logger = logging.getLogger("rag-logger")


class DataProcessingModule:
    """
    A class used to ingest and preprocess documents for use in a Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
    ----------
    data_dir : str
        The directory containing the data files to be ingested.
    web_urls: List[str]
        List of website URLs to be ingested and processed.
    chunk_size : int
        The size of each chunk of text.
    chunk_overlap : int
        The overlap between consecutive chunks of text.
    file_exts: List[str]
        List of file extensions to support. Default is [".pdf", ".txt", ".docx", ".doc", ".rtf", ".csv", ".md", ".py", ".log"]
    html_tags_to_extract: List[str]
        List of HTML tags to extract text from. Default is ["p", "table"]. We support ["p", "table", "li", "div", "span", "<h_tags>"]  currently.
    login_info: dict
        A dictionary containing login credentials for each URL. Required if the target URL requires authentication.
        Must be structured as {target_url: {"login_url": <login_url>, "credentials": {"username": "your_username", "password": "your_password"}}}

    Methods:
    -------
    chunk_data_naive(text: str) -> List[str]:
        Naively chunks text into segments of a specified size without any overlap.

    chunk_data(text: str) -> List[str]:
        Chunks text into segments using a specified overlap.

    process_file(file_path: str, doc_id: int) -> pd.DataFrame:
        Processes a single file, extracting and chunking the text.

    process_url(url: str, doc_id: int) -> pd.DataFrame:
        Processes a single URL, extracting and chunking the text.

    process_files(file_paths: List[str]) -> pd.DataFrame:
        Processes the given file paths, extracting and chunking text from each file.

    process_urls(urls: List[str]) -> pd.DataFrame:
        Processes the given URLs, extracting and chunking text from each URL.

    process_data() -> pd.DataFrame:
        Processes all files in the data directory and URLs, extracting and chunking text, and compiles the results into a single DataFrame.
    """

    def __init__(
        self,
        data_dir: str,
        web_urls: List[str],
        chunk_size: int,
        chunk_overlap: int,
        **kwargs,
    ):
        if not data_dir:
            data_dir = ""
        data_s3_bucket, data_dir = parse_path(data_dir)
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.s3_bucket = data_s3_bucket
        self.s3_client = boto3.client("s3") if self.s3_bucket else None
        self.file_exts = kwargs.get("file_exts", SUPPORTED_FILE_EXTENSIONS)
        self.web_urls = web_urls
        self.login_info = kwargs.get("login_info", {})
        if self.s3_bucket:
            self.data_dir = download_directory_from_s3(
                s3_bucket=self.s3_bucket, data_dir=self.data_dir, s3_client=self.s3_client
            )
        self.html_tags_to_extract = kwargs.get("html_tags_to_extract", SUPPORTED_HTML_TAGS)

        logger.info(f"Processing Data from Data Directory: {self.data_dir}")
        logger.info(f"Processing the Web URLs: {self.web_urls}")

        if self.web_urls:
            logger.info(f"\n Extracting text from the following HTML tags: {self.html_tags_to_extract}.")

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
        file_extension = file_extension.lower()
        if file_extension == ".pdf":
            return process_pdf(file_path, self.chunk_size, self.chunk_overlap, doc_id)
        elif file_extension in [".txt", ".md", ".py", ".log"]:
            return process_txt_md_py_log(file_path, self.chunk_data, doc_id)
        elif file_extension in [".docx", ".doc"]:
            return process_docx_doc(file_path, self.chunk_data, doc_id)
        elif file_extension == ".rtf":
            return process_rtf(file_path, self.chunk_data, doc_id)
        elif file_extension == ".csv":
            return process_csv(file_path, self.chunk_data, doc_id)

        return pd.DataFrame()

    def process_files(self, file_paths: List[str], start_doc_id: int = 0) -> pd.DataFrame:
        """
        Processes the given file paths, extracting and chunking text from each file.

        Parameters:
        ----------
        file_paths : List[str]
            A list of file paths to process.
        start_doc_id : int
            The starting document ID.

        Returns:
        -------
        pd.DataFrame
            A DataFrame of processed text chunks from the given files.
        """
        processed_data = [pd.DataFrame([])]
        doc_id_counter = start_doc_id

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(
                self.process_file, file_paths, range(doc_id_counter, doc_id_counter + len(file_paths))
            )
            for result in results:
                processed_data.append(result)
                doc_id_counter += 1

        return pd.concat(processed_data).reset_index(drop=True), doc_id_counter

    def process_url(self, url: str, doc_id: int, login_info: dict = {}) -> pd.DataFrame:
        """
        Processes a single URL, extracting and chunking the text.

        Parameters:
        ----------
        url : str
            The URL to be processed.
        doc_id : int
            The document ID.
        login_info: dict
            A dictionary containing login credentials for each URL. Required if the target URL requires authentication.

        Returns:
        -------
        pd.DataFrame
            A table containing processed text chunks and metadata from the given URL.
        """
        logger.info(f"Processing URL: {url}")

        login_url = login_info.get(url, {}).get("login_url", "")
        login_creds = login_info.get(url, {}).get("credentials", {})
        chunked_text_content = get_text_from_url(
            url,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tags_to_extract=self.html_tags_to_extract,
            login_url=login_url,
            credentials=login_creds,
        )

        data = {
            "doc_id": [doc_id] * len(chunked_text_content),
            "chunk_id": list(range(len(chunked_text_content))),
            "text": chunked_text_content,
        }

        return pd.DataFrame(data)

    def process_urls(self, urls: List[str], login_info: dict = {}, start_doc_id: int = 0) -> pd.DataFrame:
        """
        Processes the given URLs, extracting and chunking text from each URL.

        Parameters:
        ----------
        urls : List[str]
            A list of URLs to process.
        login_info: dict
            A dictionary containing login credentials for each URL. Required if the target URL requires authentication.
        start_doc_id : int
            The starting document ID.

        Returns:
        -------
        pd.DataFrame
            A DataFrame of processed text chunks from the given URLs.
        """
        processed_data = [pd.DataFrame([])]
        doc_id_counter = start_doc_id

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda url, doc_id: self.process_url(url, doc_id, login_info),
                urls,
                range(doc_id_counter, doc_id_counter + len(urls)),
            )
            for result in results:
                processed_data.append(result)
                doc_id_counter += 1
        return pd.concat(processed_data).reset_index(drop=True)

    def process_data(self) -> pd.DataFrame:
        """
        Processes all files in the data directory and URLs.

        Extracts and chunks text from each file and URL, and compiles the results into a single DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame of processed text chunks from all files in the directory and URLs.
        """

        last_doc_id = 0
        processed_files_data, processed_urls_data = pd.DataFrame(), pd.DataFrame()
        if self.data_dir:
            file_paths = get_all_file_paths(self.data_dir, self.file_exts)
            processed_files_data, last_doc_id = self.process_files(file_paths, start_doc_id=0)

        if self.web_urls:
            processed_urls_data = self.process_urls(
                self.web_urls, login_info=self.login_info, start_doc_id=last_doc_id
            )

        return pd.concat([processed_files_data, processed_urls_data]).reset_index(drop=True)

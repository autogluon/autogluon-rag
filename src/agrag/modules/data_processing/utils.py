import logging
import os
from typing import List

import boto3
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from langchain_community.document_loaders import PyPDFLoader, RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from requests.sessions import Session

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, SUPPORTED_FILE_EXTENSIONS

logger = logging.getLogger("rag-logger")


def download_directory_from_s3(s3_bucket: str, data_dir: str, s3_client: boto3.client):
    """
    Downloads an entire directory from an S3 bucket to a local directory.

    Parameters:
    ----------
    s3_bucket : str
        The name of the S3 bucket containing the data files.
    data_dir : str
        The directory within the S3 bucket to download.
    s3_client : boto3.client
        The boto3 S3 client used for interacting with S3.

    Returns:
    -------
    str
        The path to the local directory where the S3 files have been downloaded.
    """
    local_dir = "s3_docs"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=data_dir)
    for obj in response.get("Contents", []):
        s3_path = obj["Key"]
        if s3_path.endswith("/"):
            continue

        local_file_path = os.path.join(local_dir, os.path.relpath(s3_path, data_dir))
        local_file_dir = os.path.dirname(local_file_path)

        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)

        s3_client.download_file(s3_bucket, s3_path, local_file_path)
    return local_dir


def get_all_file_paths(dir_path: str, file_exts: List[str]) -> List[str]:
    """
    Recursively retrieves all file paths in the given directory.

    Parameters:
    ----------
    dir_path : str
        The directory to search for files.
    file_exts : List[str]
        List of file extensions to filter.

    Returns:
    -------
    List[str]
        A list of all file paths in the directory and its subdirectories.
    """
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(tuple(file_exts)):
                logger.warning(
                    f"\nWARNING: Skipping File {file_path}. Provided file extensions to use: {file_exts}.\nOnly file types {SUPPORTED_FILE_EXTENSIONS} are supported in this version.\n"
                )
                continue
            file_paths.append(file_path)
    return file_paths


def process_pdf(file_path: str, chunk_size: int, chunk_overlap: int, doc_id: int) -> pd.DataFrame:
    """
    Processes a PDF file, extracting and chunking the text.

    Parameters:
    ----------
    file_path : str
        The path to the PDF file to be processed.
    chunk_size : int
        The size of each chunk of text.
    chunk_overlap : int
        The overlap between consecutive chunks of text.
    doc_id : int
        The document ID.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing processed text chunks and metadata from the given PDF file.
    """
    processed_data = []
    pdf_loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[".", "\uff0e", "\n"],  # \uff0e -> Fullwidth full stop
        length_function=len,
        is_separator_regex=False,
    )
    pages = pdf_loader.load_and_split(text_splitter=text_splitter)
    for chunk_id, page in enumerate(pages):
        page_content = "".join(page.page_content)
        processed_data.append({DOC_ID_KEY: doc_id, CHUNK_ID_KEY: chunk_id, DOC_TEXT_KEY: page_content})
    return pd.DataFrame(processed_data)


def process_txt_md_py_log(file_path: str, chunk_data, doc_id: int) -> pd.DataFrame:
    """
    Processes a text-based file (TXT, MD, PY, LOG), extracting and chunking the text.

    Parameters:
    ----------
    file_path : str
        The path to the text-based file to be processed.
    chunk_data : function
        The function used to chunk the text.
    doc_id : int
        The document ID.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing processed text chunks and metadata from the given text-based file.
    """
    processed_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    for chunk_id, chunk in enumerate(chunk_data(text)):
        processed_data.append({DOC_ID_KEY: doc_id, CHUNK_ID_KEY: chunk_id, DOC_TEXT_KEY: chunk})
    return pd.DataFrame(processed_data)


def process_docx_doc(file_path: str, chunk_data, doc_id: int) -> pd.DataFrame:
    """
    Processes a DOCX or DOC file, extracting and chunking the text.

    Parameters:
    ----------
    file_path : str
        The path to the DOCX or DOC file to be processed.
    chunk_data : function
        The function used to chunk the text.
    doc_id : int
        The document ID.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing processed text chunks and metadata from the given DOCX or DOC file.
    """
    processed_data = []
    doc = Document(file_path)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    for chunk_id, chunk in enumerate(chunk_data(text)):
        processed_data.append({DOC_ID_KEY: doc_id, CHUNK_ID_KEY: chunk_id, DOC_TEXT_KEY: chunk})
    return pd.DataFrame(processed_data)


def process_rtf(file_path: str, chunk_data, doc_id: int) -> pd.DataFrame:
    """
    Processes an RTF file, extracting and chunking the text.

    Parameters:
    ----------
    file_path : str
        The path to the RTF file to be processed.
    chunk_data : function
        The function used to chunk the text.
    doc_id : int
        The document ID.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing processed text chunks and metadata from the given RTF file.
    """
    processed_data = []
    with open(file_path, "r") as f:
        text = f.read()
    for chunk_id, chunk in enumerate(chunk_data(text)):
        processed_data.append({DOC_ID_KEY: doc_id, CHUNK_ID_KEY: chunk_id, DOC_TEXT_KEY: chunk})
    return pd.DataFrame(processed_data)


def process_csv(file_path: str, chunk_data, doc_id: int) -> pd.DataFrame:
    """
    Processes a CSV file, extracting and chunking the text.

    Parameters:
    ----------
    file_path : str
        The path to the CSV file to be processed.
    chunk_data : function
        The function used to chunk the text.
    doc_id : int
        The document ID.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing processed text chunks and metadata from the given CSV file.
    """
    processed_data = []
    df = pd.read_csv(file_path)
    text = df.to_string(index=False)
    for chunk_id, chunk in enumerate(chunk_data(text)):
        processed_data.append({DOC_ID_KEY: doc_id, CHUNK_ID_KEY: chunk_id, DOC_TEXT_KEY: chunk})
    return pd.DataFrame(processed_data)


def get_authenticated_html(url: str, login_url: str, credentials: dict) -> str:
    """
    Authenticates and retrieves HTML content from a URL that requires login.

    Parameters:
    ----------
    url : str
        The URL to retrieve HTML content from after authentication.
    login_url : str
        The URL to submit the login form.
    credentials : dict
        A dictionary containing login credentials (e.g., {"username": "your_username", "password": "your_password"}).

    Returns:
    -------
    str
        The HTML content of the authenticated URL.
    """
    session = Session()
    session.post(login_url, data=credentials)
    response = session.get(url)
    response.raise_for_status()
    return response.text


def bs4_extractor(html: str, tags_to_extract: List[str] = ["p", "table"]) -> str:
    """
    Extracts text from HTML content using BeautifulSoup, focusing on specified HTML tags.

    Parameters:
    ----------
    html : str
        The raw HTML content.
    tags_to_extract : List
        A list of HTML tags to extract text from. Default is ["p", "table"].

    Returns:
    -------
    str
        The extracted text content from the specified HTML tags.
    """
    soup = BeautifulSoup(
        html,
        "lxml",
    )

    extracted_text = []

    for tag in tags_to_extract:
        elements = soup.find_all(tag)
        for element in elements:
            if tag == "table":
                table_text = "\n".join([" ".join(row.stripped_strings) for row in element.find_all("tr")])
                extracted_text.append(table_text)
            else:
                extracted_text.append(element.get_text())

    text = "\n".join(extracted_text).strip()
    return text


def get_text_from_url(
    url: str,
    chunk_size: int,
    chunk_overlap: int,
    tags_to_extract: List = ["p", "table"],
    login_url: str = None,
    credentials: dict = None,
) -> str:
    """
    Retrieves and extracts text content from a given URL using a specified extractor.
    Handles authentication if login URL and credentials are provided.

    Parameters:
    ----------
    url : str
        The URL to retrieve and extract text content from.
    chunk_size : int
        The size of each chunk of text.
    chunk_overlap : int
        The overlap between consecutive chunks of text.
    tags_to_extract : List
        A list of HTML tags to extract text from. Default is ["p", "table"].
    login_url : str, optional
        The URL to submit the login form. Required if the target URL requires authentication.
    credentials : dict, optional
        A dictionary containing login credentials. Required if the target URL requires authentication.

    Returns:
    -------
    str
        The extracted text content from the URL.
    """
    if login_url and credentials:
        html_content = get_authenticated_html(url, login_url, credentials)
    else:
        loader = RecursiveUrlLoader(
            url,
            continue_on_failure=True,
            max_depth=1,
        )
        html_content = "".join([page.page_content for page in loader.load()])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[".", "\uff0e", "\n"],  # \uff0e -> Fullwidth full stop
        length_function=len,
        is_separator_regex=False,
    )
    processed_data = []
    chunks = text_splitter.split_text(html_content)
    for chunk in chunks:
        chunk_content = bs4_extractor(chunk, tags_to_extract=tags_to_extract)
        processed_data.append(chunk_content)
    return processed_data

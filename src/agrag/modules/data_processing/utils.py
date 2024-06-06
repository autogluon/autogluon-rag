import logging
import os
from typing import List

import boto3

logger = logging.getLogger("rag-logger")


def download_directory_from_s3(s3_bucket: str, data_dir: str, s3_client: boto3.client):
    """
    Downloads an entire directory from an S3 bucket to a local directory.

    This function recursively downloads all files from the specified directory in the S3 bucket to a local directory, maintaining the same directory structure.

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

    Example:
    --------
    s3_client = boto3.client("s3")
    local_dir = download_directory_from_s3("my-s3-bucket", "path/to/files", s3_client)
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


def get_all_file_paths(dir_path: str) -> List[str]:
    """
    Recursively retrieves all file paths in the given directory.

    Parameters:
    ----------
    dir_path : str
        The directory to search for files.

    Returns:
    -------
    List[str]
        A list of all file paths in the directory and its subdirectories.
    """
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(".pdf"):  # Only PDFs for now
                logger.warning(
                    f"\nWARNING: Skipping File {file_path}. Only PDF files are supported in this version.\n"
                )
                continue
            file_paths.append(file_path)
    return file_paths

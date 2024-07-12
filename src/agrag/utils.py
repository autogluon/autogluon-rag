import logging
import os
from typing import Optional, Tuple

import psutil
import torch

logger = logging.getLogger("rag-logger")


def parse_path(path: str) -> Tuple[Optional[str], str]:
    """
    Parses a given path to determine if it is an S3 path or a local path.

    Parameters:
    ----------
    path : str
        The full path to be parsed.

    Returns:
    -------
    Tuple[Optional[str], str]
        A tuple containing:
        - The S3 bucket name if the path is an S3 path, otherwise None.
        - The actual path within the S3 bucket or the local path.
    """
    if path.startswith("s3://"):
        parts = path.split("/", 3)
        s3_bucket = parts[2]
        s3_path = parts[3] if len(parts) > 3 else ""
        return s3_bucket, s3_path
    else:
        return None, path


def read_openai_key(file_path: str) -> str:
    """
    Reads the OpenAI secret key from a text file.

    Parameters:
    ----------
    file_path : str
        The path to the text file containing the OpenAI secret key.

    Returns:
    -------
    str
        The OpenAI secret key.

    Raises:
    ------
    FileNotFoundError
        If the specified file does not exist.
    IOError
        If there is an error reading the file.
    """
    if not file_path:
        return None
    try:
        with open(file_path, "r") as file:
            key = file.read().strip()
            return key
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file at {file_path}: {e}")


def get_num_gpus(num_gpus):
    """
    Determines the number of GPUs to use, based on the available GPUs and the requested number.
    Parameters:
    ----------
    num_gpus : int or None
        The desired number of GPUs to use. If None, the maximum available GPUs will be used.
    Returns:
    -------
    int
        The number of GPUs to use, which is the minimum of the requested number and the available GPUs.
    """
    max_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = max_gpus
        return num_gpus
    elif num_gpus > 0:
        num_gpus = min(num_gpus, max_gpus)
    return num_gpus


def get_directory_info(directory_path: str):
    """
    Given a path to a directory, returns the number of files and the total size of all the files.

    Parameters:
    ----------
    directory_path : str
        The path to the directory.

    Returns:
    -------
    int
        The number of files in the directory.
    int
        The total size of all the files in bytes.
    """
    total_size = 0
    file_count = 0

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
            file_count += 1

    return file_count, total_size


def get_available_memory() -> int:
    """
    Returns the available system memory in bytes.

    Returns:
    -------
    int
        The available system memory in bytes.
    """
    mem = psutil.virtual_memory()
    return mem.available


def determine_batch_size(directory: str, safety_factor: float = 0.5, max_files_per_batch: int = 100) -> int:
    """
    Determines the batch size based on the file count, total size, and available system memory.

    This function calculates the average file size and then estimates the maximum batch size
    by considering the available system memory and a safety factor to avoid using all available memory.
    The safety factor accounts for memory overhead and ensures safe memory usage during processing.
    Increasing the safety factor will increase the calculated batch size because more memory is considered available for use.

    Parameters:
    ----------
    directory : str
        The path to the directory.
    safety_factor : float
        A factor to account for memory overhead and ensure safe memory usage (default is 0.5).
    max_files_per_batch : int
        The maximum number of files to include in a batch (default is 100).

    Returns:
    -------
    int
        The calculated batch size.
    """
    file_count, total_size = get_directory_info(directory)
    available_memory = get_available_memory()

    logger.info(f"Total files: {file_count}, Total size: {total_size} bytes")
    logger.info(f"Available memory: {available_memory} bytes")

    if file_count == 0:
        raise ValueError("No files found in the directory.")

    average_file_size = total_size / file_count
    batch_size_by_memory = int((available_memory * safety_factor) / average_file_size)

    logger.info(f"Average file size: {average_file_size} bytes")
    logger.info(f"Calculated batch size by available memory: {batch_size_by_memory}")

    batch_size = min(batch_size_by_memory, max_files_per_batch, file_count)
    logger.info(f"Final batch size: {batch_size}")

    return batch_size

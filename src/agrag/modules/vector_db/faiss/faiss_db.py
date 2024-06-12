import logging
from typing import List

import boto3
import faiss
import numpy as np
import torch
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

logger = logging.getLogger("rag-logger")


def construct_faiss_index(embeddings: List[torch.Tensor], num_gpus: int = 1) -> faiss.IndexFlatL2:
    """
    Constructs a FAISS index and stores the embeddings.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be stored in the FAISS index.
    num_gpus: int
        Number of GPUs to use when building the index

    Returns:
    -------
    Union[faiss.IndexFlatL2, faiss.GpuIndexFlatL2]
        The constructed FAISS index.
    """
    d = embeddings[0].shape[-1]  # dimension of the vectors
    logger.info(f"Constructing FAISS index with dimension: {d}")

    index = faiss.IndexFlatL2(d)  # Flat (CPU) index, L2 distance

    if num_gpus >= 1:
        devices = [faiss.StandardGpuResources() for _ in range(num_gpus)]
        config = [faiss.GpuIndexFlatConfig() for _ in range(num_gpus)]
        for i in range(num_gpus):
            config[i].device = i
        index = faiss.index_cpu_to_gpu_multiple(devices, index, config)
        logger.info(f"Using FAISS GPU index on {num_gpus} GPUs")

    embeddings_array = np.array(embeddings)
    index.add(embeddings_array)
    logger.info(f"Stored {embeddings_array.shape[0]} embeddings in the FAISS index")

    return index


def save_faiss_index(index: faiss.IndexFlatL2, index_path: str) -> None:
    """
    Saves the FAISS index to disk.

    Parameters:
    ----------
    index: faiss.IndexFlatL2
        The FAISS index to store
    index_path : str
        The path where the FAISS index will be saved.

    Returns:
    -------
    bool:
        True, if Vector DB Index loaded successfully from memory
        False, else
    """
    try:
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
        return True
    except (IOError, OSError) as e:
        logger.error(f"Failed to save FAISS index to {index_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving FAISS index to {index_path}: {e}")
        return False


def load_faiss_index(index_path: str) -> faiss.IndexFlatL2:
    """
    Loads the FAISS index from disk.

    Parameters:
    ----------
    index_path : str
        The path from where the FAISS index will be loaded.

    Returns:
    -------
    Union[faiss.IndexFlatL2, faiss.GpuIndexFlatL2]
        The loaded FAISS index.
    """
    try:
        index = faiss.read_index(index_path)
        logger.info(f"FAISS index loaded from {index_path}")
        return index
    except (IOError, OSError) as e:
        logger.error(f"Failed to load FAISS index from {index_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading FAISS index from {index_path}: {e}")
        return None


def save_faiss_index_s3(
    index_path: str,
    s3_bucket: str,
    s3_client: boto3.session.Session.client,
):
    """
    Saves the FAISS index to S3.

    Parameters:
    ----------
    index_path : str
        The path where the FAISS index will be saved.
    s3_bucket: str
        S3 bucket to store the index in
    s3_client: boto3.session.Session.client
        S3 client to interface with AWS resources

    Returns:
    -------
    bool:
        True, if Vector DB Index saved successfully to S3
        False, else
    """
    try:
        s3_client.upload_file(Filename=index_path, Bucket=s3_bucket, Key=index_path)
        logger.info(f"FAISS index saved to S3 Bucket {s3_bucket} at {index_path}")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("AWS credentials not found or incomplete.")
        return False
    except ClientError as e:
        logger.error(f"Failed to upload FAISS index to S3: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving FAISS index to S3: {e}")
        return False
    finally:
        s3_client.close()


def load_faiss_index_s3(
    index_path: str,
    s3_bucket: str,
    s3_client: boto3.session.Session.client,
):
    """
    Loads the FAISS index from S3.

    Parameters:
    ----------
    index_path : str
        The path from where the FAISS index will be loaded.
    s3_bucket: str
        S3 bucket to store the index in
    s3_client: boto3.session.Session.client
        S3 client to interface with AWS resources

    Returns:
    -------
    bool:
        True, if Vector DB Index loaded successfully from S3
        False, else
    """
    try:
        s3_client.download_file(Filename=index_path, Bucket=s3_bucket, Key=index_path)
        logger.info(f"FAISS index loaded from S3 Bucket {s3_bucket} and stored at {index_path}")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("AWS credentials not found or incomplete.")
        return False
    except ClientError as e:
        logger.error(f"Failed to download FAISS index from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading FAISS index from S3: {e}")
        return False
    finally:
        s3_client.close()

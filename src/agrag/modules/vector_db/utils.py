import logging
import os
from typing import List, Tuple, Union

import boto3
import faiss
import numpy as np
import pandas as pd
import torch
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from tqdm import tqdm

from agrag.modules.vector_db.faiss.faiss_db import (
    load_faiss_index,
    load_faiss_index_s3,
    save_faiss_index,
    save_faiss_index_s3,
)

logger = logging.getLogger("rag-logger")


def cosine_similarity_fn(embeddings: np.ndarray) -> np.ndarray:
    return cosine_similarity(embeddings)


def euclidean_similarity_fn(embeddings: np.ndarray) -> np.ndarray:
    return -euclidean_distances(embeddings)


def manhattan_similarity_fn(embeddings: np.ndarray) -> np.ndarray:
    return -manhattan_distances(embeddings)


SUPPORTED_SIMILARITY_FUNCTIONS = {
    "cosine": cosine_similarity_fn,
    "euclidean": euclidean_similarity_fn,
    "manhattan": manhattan_similarity_fn,
}


def remove_duplicates(
    embeddings: List[torch.Tensor], similarity_threshold: float, similarity_fn: str
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Removes duplicate embeddings based on cosine similarity.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be deduplicated.
    similarity_threshold : float
        The threshold for considering embeddings as duplicates based on cosine similarity.
    similarity_fn : str
        The name of the similarity function to use. Must be one of the supported functions.

    Returns:
    -------
    Tuple[List[torch.Tensor], List[int]]
        A list of deduplicated embeddings and their indices.
    """
    if len(embeddings) <= 1:
        return embeddings, list(range(len(embeddings)))

    if similarity_fn not in SUPPORTED_SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"Unsupported similarity function: {similarity_fn}. Please choose from: {list(SUPPORTED_SIMILARITY_FUNCTIONS.keys())}"
        )

    embeddings_array = np.array([emb.numpy().flatten() for emb in embeddings])
    sim_fn = SUPPORTED_SIMILARITY_FUNCTIONS[similarity_fn]
    similarity_matrix = sim_fn(embeddings_array)

    remove = set()
    indices_to_keep = []
    for i in range(len(similarity_matrix)):
        if i in remove:
            continue
        indices_to_keep.append(i)
        duplicates = np.where(similarity_matrix[i, i + 1 :] > similarity_threshold)[0] + (
            i + 1
        )  # Only consider the upper triangle of the similarity matrix.
        remove.update(duplicates)

    deduplicated_embeddings = [embedding for i, embedding in enumerate(embeddings) if i not in remove]
    logger.info(f"Removed {len(remove)} duplicate embeddings")
    return deduplicated_embeddings, indices_to_keep


def pad_embeddings(embeddings: List[torch.Tensor]) -> torch.Tensor:
    """
    Pads embeddings to ensure they have the same length.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be padded.

    Returns:
    -------
    torch.Tensor
        A tensor containing the padded embeddings.
    """
    max_len = max(embedding.shape[1] for embedding in embeddings)
    padded_embeddings = [
        torch.nn.functional.pad(embedding, (0, 0, 0, max_len - embedding.shape[1])) for embedding in embeddings
    ]
    return torch.cat(padded_embeddings, dim=0)


def save_index(
    db_type: str,
    index: Union[faiss.IndexFlatL2],
    index_path: str,
    s3_bucket: str = None,
    s3_client: boto3.session.Session.client = None,
) -> None:
    """
    Saves the Vector DB index to disk.

    Parameters:
    ----------
    db_type: str
        The type of Vector DB being used
    index: Union[faiss.IndexFlatL2]
        The Vector DB index to store
    index_path : str
        The path where the index will be saved.
    s3_bucket: str
        S3 bucket to store the index in
    s3_client: boto3.session.Session.client
        S3 client to interface with AWS resources
    """
    if not index:
        raise ValueError("No index to save. Please construct the index first.")
    if not index_path:
        logger.warning(f"Cannot save index. Invalid path {index_path}.")
        return
    if not os.path.isfile(index_path):
        with open(index_path, "w") as fp:
            pass
    if db_type == "faiss":
        if not type(index) is faiss.IndexFlatL2:
            raise TypeError("Index for FAISS incorrectly created. Not of type IndexFlatL2.")
        success = save_faiss_index(index, index_path)
        if s3_bucket and success:
            save_faiss_index_s3(index_path, s3_bucket, s3_client)
        else:
            logger.warning(f"Failed to save index")
    else:
        logger.warning(f"Cannot save index. Unsupported Vector DB {db_type}.")


def load_index(
    db_type: str,
    index_path: str,
    s3_bucket: str = None,
    s3_client: boto3.session.Session.client = None,
    pbar: tqdm = None,
) -> Union[faiss.IndexFlatL2]:
    """
    Loads the Vector DB index from disk.

    Parameters:
    ----------
    db_type: str
        The type of Vector DB being used
    index_path : str
        The path from where the index will be loaded.
    s3_bucket: str
        S3 bucket to store the index in
    s3_client: boto3.session.Session.client
        S3 client to interface with AWS resources

    Returns:
    -------
    Union[faiss.IndexFlatL2]
        Vector DB Index
    """
    index = None
    if db_type == "faiss":
        if s3_bucket:
            load_faiss_index_s3(index_path, s3_bucket, s3_client)
        index = load_faiss_index(index_path)
    else:
        raise ValueError("Cannot load index. Unsupported Vector DB {db_type}.")
    if pbar:
        pbar.n = pbar.total
        pbar.refresh()
    return index


def save_metadata(
    metadata: pd.DataFrame, metadata_path: str, s3_bucket: str = None, s3_client: boto3.session.Session.client = None
):
    """
    Saves metadata to file.

    Parameters:
    ----------
    metadata: pd.DataFrame
        Metadata to store
    metadata_path : str
        The path to the metadata file.
    s3_bucket : str
        The S3 bucket name.
    s3_client : boto3.session.Session.client
        The S3 client to interface with AWS resources.

    Returns:
    -------
    bool:
        True, if metadata saved successfully to file
        False, else
    """
    if metadata.empty:
        raise ValueError("No metadata to save. Please construct metadata first.")
    if not metadata_path:
        logger.warning(f"Cannot save metadata. Invalid path {metadata_path}.")
        return False
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Metadata not a pandas DataFrame.")

    metadata_dir = os.path.dirname(metadata_path)
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    try:
        metadata.to_json(metadata_path, orient="records", lines=True)
        logger.info(f"Metadata saved to {metadata_path}")
    except (IOError, Exception) as e:
        logger.error(f"Failed to save metadata to {metadata_path}: {e}")
        return False

    if s3_bucket:
        try:
            s3_client.upload_file(Filename=metadata_path, Bucket=s3_bucket, Key=metadata_path)
            logger.info(f"Metadata saved to S3 Bucket {s3_bucket} at {metadata_path}.")
            return True
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return False
        except ClientError as e:
            logger.error(f"Failed to upload metadata to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving metadata to S3: {e}")
            return False
    return True


def load_metadata(
    metadata_path: str, s3_bucket: str = None, s3_client: boto3.session.Session.client = None
) -> pd.DataFrame:
    """
    Loads metadata from file.

    Parameters:
    ----------
    metadata_path : str
        The path to the metadata file.
    s3_bucket : str
        The S3 bucket name.
    s3_client : boto3.session.Session.client
        The S3 client to interface with AWS resources.

    Returns:
    -------
    pd.DataFrame
        Metadata for Vector DB
    """
    if s3_bucket:
        try:
            s3_client.download_file(Filename=metadata_path, Bucket=s3_bucket, Key=metadata_path)
            logger.info(f"Metadata loaded from S3 Bucket {s3_bucket} at {metadata_path}.")
        except (NoCredentialsError, PartialCredentialsError):
            logger.error("AWS credentials not found or incomplete.")
            return pd.DataFrame()
        except ClientError as e:
            logger.error(f"Failed to download metadata from S3: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading metadata from S3: {e}")
            return pd.DataFrame()

    try:
        metadata = pd.read_json(metadata_path, orient="records", lines=True)
        logger.info(f"Metadata loaded from {metadata_path}")
    except (IOError, Exception) as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return pd.DataFrame()

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Metadata not a pandas DataFrame.")
    return metadata

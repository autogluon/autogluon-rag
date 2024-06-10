import logging
import os
from typing import List, Union

import boto3
import faiss
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

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
) -> List[torch.Tensor]:
    """
    Removes duplicate embeddings based on cosine similarity.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be deduplicated.
    similarity_threshold : float
        The threshold for considering embeddings as duplicates based on cosine similarity
    similarity_fn : str
        The name of the similarity function to use. Must be one of the supported functions.

    Returns:
    -------
    List[torch.Tensor]
        A list of deduplicated embeddings.
    """
    if len(embeddings) <= 1:
        return embeddings

    if similarity_fn not in SUPPORTED_SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"Unsupported similarity function: {similarity_fn}. Please choose from: {list(SUPPORTED_SIMILARITY_FUNCTIONS.keys())}"
        )

    embeddings_array = np.array(embeddings).reshape(len(embeddings), -1)
    sim_fn = SUPPORTED_SIMILARITY_FUNCTIONS[similarity_fn]
    similarity_matrix = sim_fn(embeddings_array)

    remove = set()
    for i in range(len(similarity_matrix)):
        if i in remove:
            continue
        duplicates = np.where(similarity_matrix[i, i + 1 :] > similarity_threshold)[0] + (
            i + 1
        )  # Only consider the upper triangle of the similarity matrix.
        remove.update(duplicates)

    deduplicated_embeddings = [embedding for i, embedding in enumerate(embeddings) if i not in remove]
    logger.info(f"Removed {len(remove)} duplicate embeddings")
    return deduplicated_embeddings


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
    return torch.cat(padded_embeddings, dim=0).view(len(padded_embeddings), -1)


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
        save_faiss_index(index, index_path)
        if s3_bucket:
            save_faiss_index_s3(index_path, s3_bucket, s3_client)
    else:
        logger.warning(f"Cannot save index. Unsupported Vector DB {db_type}.")


def load_index(
    db_type: str,
    index_path: str,
    s3_bucket: str = None,
    s3_client: boto3.session.Session.client = None,
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
    """
    index = None
    if db_type == "faiss":
        if s3_bucket:
            index = load_faiss_index_s3(index_path, s3_bucket, s3_client)
        else:
            index = load_faiss_index(index_path)
    else:
        raise ValueError("Cannot load index. Unsupported Vector DB {db_type}.")
    return index

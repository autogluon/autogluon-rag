import logging
import os
from typing import Any, List

import faiss
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from agrag.modules.vector_db.faiss.faiss import load_faiss_index, save_faiss_index

logger = logging.getLogger("rag-logger")


def remove_duplicates(embeddings: List[torch.Tensor], similarity_threshold: float) -> List[torch.Tensor]:
    """
    Removes duplicate embeddings based on cosine similarity.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be deduplicated.

    Returns:
    -------
    List[torch.Tensor]
        A list of deduplicated embeddings.
    """
    if len(embeddings) <= 1:
        return embeddings

    embeddings_tensor = torch.stack(embeddings)
    embeddings_array = embeddings_tensor.numpy().reshape(len(embeddings), -1)
    similarity_matrix = cosine_similarity(embeddings_array)

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


def save_index(db_type: str, index: Any, index_path: str) -> None:
    """
    Saves the Vector DB index to disk.

    Parameters:
    ----------
    db_type: Vector DB type
        The type of Vector DB being used
    index: Vector DB index data type
        The Vector DB index to store
    index_path : str
        The path where the FAISS index will be saved.
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
        save_faiss_index(index, index_path)
    else:
        logger.warning(f"Cannot save index. Unsupported Vector DB {db_type}.")


def load_index(db_type: str, index_path: str) -> faiss.IndexFlatL2:
    """
    Loads the Vector DB index from disk.

    Parameters:
    ----------
    db_type: Vector DB type
        The type of Vector DB being used
    index_path : str
        The path from where the index will be loaded.

    """
    index = None
    if db_type == "faiss":
        load_faiss_index(index_path)
    else:
        raise ValueError("Cannot load index. Unsupported Vector DB {db_type}.")
    return index

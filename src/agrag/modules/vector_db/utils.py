import logging
from typing import List

import torch
from sklearn.metrics.pairwise import cosine_similarity

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

    embeddings_array = embeddings.numpy().reshape(len(embeddings), -1)
    similarity_matrix = cosine_similarity(embeddings_array)

    to_remove = set()
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > similarity_threshold:
                to_remove.add(j)

    deduplicated_embeddings = [embedding for i, embedding in enumerate(embeddings) if i not in to_remove]
    logger.info(f"Removed {len(to_remove)} duplicate embeddings")
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

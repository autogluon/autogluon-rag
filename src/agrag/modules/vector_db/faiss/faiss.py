import logging
from typing import List, Union

import faiss
import numpy as np
import torch
import os

logger = logging.getLogger("rag-logger")


def construct_faiss_index(embeddings: List[torch.Tensor], gpu: bool) -> faiss.IndexFlatL2:
    """
    Constructs a FAISS index and stores the embeddings.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be stored in the FAISS index.

    Returns:
    -------
    Union[faiss.IndexFlatL2, faiss.GpuIndexFlatL2]
        The constructed FAISS index.
    """
    d = embeddings[0].shape[-1]  # dimension of the vectors
    logger.info(f"Constructing FAISS index with dimension: {d}")

    index = faiss.IndexFlatL2(d)  # Flat (CPU) index, L2 distance

    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        logger.info("Using FAISS GPU index")

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
    """
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index saved to {index_path}")


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
    index = faiss.read_index(index_path)
    logger.info(f"FAISS index loaded from {index_path}")
    return index

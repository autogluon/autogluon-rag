import logging
from typing import List

import faiss
import numpy as np
import torch

logger = logging.getLogger("rag-logger")


def construct_faiss_index(embeddings: List[torch.Tensor], embedding_dim: int, num_gpus: int = 1) -> faiss.IndexFlatL2:
    """
    Constructs a FAISS index and stores the embeddings.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be stored in the FAISS index.
    embedding_dim: int
        Dimension of embeddings to be used to create index of appropriate dimension
    num_gpus: int
        Number of GPUs to use when building the index

    Returns:
    -------
    Union[faiss.IndexFlatL2, faiss.GpuIndexFlatL2]
        The constructed FAISS index.
    """
    d = embeddings[0].shape[-1]
    assert d == embedding_dim, f"Dimension of embeddings is incorrect {embedding_dim}"
    logger.info(f"Constructing FAISS index with dimension: {d}")

    index = faiss.IndexFlatL2(d)  # Flat (CPU) index, L2 distance

    if num_gpus >= 1:
        devices = [faiss.StandardGpuResources() for _ in range(num_gpus)]
        config = [faiss.GpuIndexFlatConfig() for _ in range(num_gpus)]
        for i in range(num_gpus):
            config[i].device = i
        index = faiss.index_cpu_to_gpu_multiple(devices, index, config)
        logger.info(f"Using FAISS GPU index on {num_gpus} GPUs")

    index.add(np.array(embeddings))
    if len(embeddings) != index.ntotal:
        raise ValueError(
            f"Stored {index.ntotal} embeddings in the FAISS index, expected {len(embeddings)}. Number of embeddings is {len(embeddings)}."
        )
    logger.info(f"Stored {index.ntotal} embeddings in the FAISS index.")

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

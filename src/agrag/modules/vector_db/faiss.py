import logging
from typing import List, Union

import faiss
import numpy as np
import torch

from agrag.modules.vector_db.utils import pad_embeddings

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

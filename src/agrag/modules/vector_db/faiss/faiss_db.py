import logging
from typing import List

import faiss
import numpy as np
import torch

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def construct_faiss_index(
    embeddings: List[torch.Tensor],
    embedding_dim: int,
    num_gpus: int = 1,
    index_type: str = "IndexFlatL2",
    **kwargs,
) -> faiss.IndexFlatL2:
    """
    Constructs a FAISS index and stores the embeddings.

    Parameters:
    ----------
    index_type: str
        Type of FAISS Index to use (IndexFlatL2, IndexIVFFlat, IndexIVFPQ)
    embeddings : List[torch.Tensor]
        A list of embeddings to be stored in the FAISS index.
    embedding_dim: int
        Dimension of embeddings to be used to create index of appropriate dimension
    num_gpus: int
        Number of GPUs to use when building the index

    Returns:
    -------
    Union[faiss.IndexFlatL2,
        faiss.IndexFlatIP,
        faiss.IndexHNSWFlat,
        faiss.IndexLSH,
        faiss.IndexPQ,
        faiss.IndexIVFFlat,
        faiss.IndexScalarQuantizer,
        faiss.IndexIVFPQ,]
        The constructed FAISS index.
    """
    d = embeddings[0].shape[-1]
    assert d == embedding_dim, f"Dimension of embeddings is incorrect {embedding_dim}"
    logger.info(f"Constructing FAISS index with dimension: {d}")

    faiss_index_params = kwargs.get("faiss_index_params", {})
    faiss_search_params = kwargs.get("faiss_search_params", {})
    faiss_index_nprobe = faiss_search_params.get("nprobe")
    index = None

    # Define a mapping of quantizer names to FAISS index classes
    quantizer_mapping = {
        "IndexFlatL2": lambda d: faiss.IndexFlatL2(d),
        "IndexFlatIP": lambda d: faiss.IndexFlatIP(d),
        "IndexHNSWFlat": lambda d: faiss.IndexHNSWFlat(d),
        # Add more quantizers as needed
    }

    # Define a mapping for scalar quantizers (these are not callable)
    scalar_quantizer_mapping = {
        "QT_8bit": faiss.ScalarQuantizer.QT_8bit,
        # Add more quantizer types as needed
    }

    # Initialize quantizer based on the given string
    def get_quantizer(quantizer_name, d):
        if quantizer_name in quantizer_mapping:
            return quantizer_mapping[quantizer_name](d)  # Call the index classes
        elif quantizer_name in scalar_quantizer_mapping:
            return scalar_quantizer_mapping[quantizer_name]  # Return the scalar quantizer type
        else:
            raise ValueError(f"Unsupported quantizer: {quantizer_name}")

    logger.info(f"Using FAISS Index {index_type}")
    if index_type == "IndexIVFPQ":
        nlist = faiss_index_params.get("nlist")
        m = faiss_index_params.get("m")
        bits = faiss_index_params.get("nbits")
        quantizer = get_quantizer(faiss_index_params.get("quantizer"), d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
    elif index_type == "IndexIVFFlat":
        quantizer = get_quantizer(faiss_index_params.get("quantizer"), d)
        nlist = faiss_index_params.get("nlist")
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(d)
    elif index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(d)
    elif index_type == "IndexHNSWFlat":
        m = faiss_index_params.get("m")
        index = faiss.IndexHNSWFlat(d, m)
    elif index_type == "IndexLSH":
        bits = faiss_index_params.get("nbits")
        index = faiss.IndexLSH(d, bits)
    elif index_type == "IndexPQ":
        m = faiss_index_params.get("m")
        bits = faiss_index_params.get("nbits")
        index = faiss.IndexPQ(d, m, bits)
    elif index_type == "IndexScalarQuantizer":
        quantizer = get_quantizer(faiss_index_params.get("quantizer"), d)
        index = faiss.IndexScalarQuantizer(d, quantizer)
    else:
        raise ValueError(f"Unsupported FAISS index type {index_type}")

    if num_gpus >= 1:
        index = faiss.index_cpu_to_gpus_list(index=index, ngpu=num_gpus)
        logger.info(f"Using FAISS GPU index on {num_gpus} GPUs")

    if index_type in ("IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"):
        index.add(np.array(embeddings))
    elif index_type in ("IndexIVFPQ", "IndexIVFFlat", "IndexPQ", "IndexScalarQuantizer"):
        index.train(np.array(embeddings))
        assert (
            index.is_trained
        ), f"Index {index_type} not trained. Make sure index.train(embeddings) is being called correctly."
        index.add(np.array(embeddings))
        if faiss_index_nprobe:
            index.nprobe = faiss_index_nprobe

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

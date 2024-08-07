import logging
import os
from typing import List

import torch
from pymilvus import MilvusClient

from agrag.constants import LOGGER_NAME

logger = logging.getLogger("AutoGluon-RAG-logger")


def construct_milvus_index(
    embeddings: List[torch.Tensor],
    collection_name: str,
    db_name: str,
    embedding_dim: int,
    index_params: dict = {},
    create_params: dict = {},
) -> MilvusClient:
    """
    Constructs a Milvus index and stores the embeddings.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of embeddings to be stored in the Milvus collection.
    collection_name : str
        The name of the collection in Milvus.
    db_name: str
        The name of the client in Milvus. This will also be the path at which the Milvus index is stored.
    embedding_dim: int
        Dimension of embeddings to be used to create index of appropriate dimension
    index_params: dict
        Additional params to pass into the Milvus index
    create_params: dict
        Additional params to pass into the Milvus collection creation

    Returns:
    -------
    MilvusClient
        The constructed Milvus Client Database.
    """

    d = embeddings[0].shape[-1]
    assert d == embedding_dim, f"Dimension of embeddings is incorrect {embedding_dim}"

    basedir = os.path.dirname(db_name)
    if basedir and not os.path.exists(basedir):
        logger.info(f"Creating directory for Milvus Vector Index save at {basedir}")
        os.makedirs(basedir)

    client = MilvusClient(db_name)
    if client.has_collection(collection_name):
        logger.info(f"Removing existing index {collection_name}")
        client.drop_collection(collection_name)
    logger.info(f"Creating new index {collection_name} and adding to the database {db_name}")
    client.create_collection(
        collection_name=collection_name,
        dimension=d,
        vector_field_name="embedding",
        index_params=index_params,
        **create_params,
    )
    vectors = [embedding.numpy() for embedding in embeddings]
    data = [{"id": i, "embedding": vectors[i]} for i in range(len(vectors))]

    _ = client.insert(collection_name=collection_name, data=data)

    logger.info(f"Stored {len(embeddings)} embeddings in the Milvus collection.")

    return client


def load_milvus_index(index_path):
    client = None
    if not index_path.endswith(".db"):
        logger.warning("\n\nWARNING: Incorrect Index Path. Milvus index must be a file of type '.db'\n\n")
    try:
        client = MilvusClient(index_path)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading Milvus index from {index_path}: {e}")

    logger.info(f"Milvus index loaded from {index_path}")
    return client


def save_milvus_index():
    logger.warning(
        "Milvus does not support saving the index explicitly."
        + "Milvus handles the persistence of data and indexes internally and automatically saves the database to the directory."
    )

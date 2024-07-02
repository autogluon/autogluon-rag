import logging
from typing import List

import torch
from pymilvus import MilvusClient

from agrag.constants import MILVUS_DB_COLLECTION_NAME, MILVUS_DB_NAME

logger = logging.getLogger("rag-logger")


def construct_milvus_index(
    embeddings: List[torch.Tensor],
    collection_name: str = MILVUS_DB_COLLECTION_NAME,
    db_name: str = MILVUS_DB_NAME,
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
        The name of the client in Milvus
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

    client = MilvusClient(db_name)
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
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


def load_milvus_index():
    raise NotImplementedError(
        "Milvus does not support loading the index directly."
        + "Milvus handles the persistence of data and indexes internally."
    )


def save_milvus_index():
    logger.warning(
        "Milvus does not support saving the index directly."
        + "Milvus handles the persistence of data and indexes internally."
    )

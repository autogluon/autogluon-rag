import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from torch.nn import DataParallel

from agrag.constants import DOC_TEXT_KEY, EMBEDDING_KEY
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.vector_db.vector_database import VectorDatabaseModule

logger = logging.getLogger("rag-logger")


class RetrieverModule:
    """
    Initializes the RetrieverModule with the VectorDatabaseModule.

    Attributes:
    ----------
    vector_database_module : VectorDatabaseModule
        The module containing the vector database and metadata.
    embedding_model: EmbeddingModule,
        The module for generating embeddings.
    top_k: int
        The top-k documents to retrieve (default is 50).
    **kwargs : dict
        Additional parameters for `RetrieverModule`.

    Methods:
    -------
    encode_query(query: str) -> np.ndarray:
        Encodes the query into an embedding.

    retrieve(query: str) -> List[Dict[str, Any]]:
        Retrieves the top_k most similar embeddings to the query.
    """

    def __init__(
        self,
        vector_database_module: VectorDatabaseModule,
        embedding_module: EmbeddingModule,
        top_k: int = 50,
        reranker: Reranker = None,
        **kwargs,
    ):
        self.embedding_module = embedding_module

        self.vector_database_module = vector_database_module
        self.top_k = top_k

        num_gpus = kwargs.get("num_gpus", 0)
        self.num_gpus = num_gpus
        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model)

        vector_database_module.num_gpus = num_gpus
        self.reranker = reranker

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes the query into an embedding.

        Parameters:
        ----------
        query : str
            The query to be encoded.

        Returns:
        -------
        np.ndarray
            The embedding of the query.
        """
        query_embedding = self.embedding_module.encode(data=pd.DataFrame([{DOC_TEXT_KEY: query}]))
        query_embedding = query_embedding[EMBEDDING_KEY][0]
        return query_embedding

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar embeddings to the query.

        Parameters:
        ----------
        query : str
            The query to retrieve documents for.

        Returns:
        -------
        List[str]
            A list of chunks for the top_k most similar embeddings.
        """
        logger.info(f"\nRetrieving top {self.top_k} most similar embeddings")
        query_embedding = self.encode_query(query)
        indices = self.vector_database_module.search_vector_database(embedding=query_embedding, top_k=self.top_k)

        valid_indices = [idx for idx in indices if idx < self.vector_database_module.metadata.shape[0]]

        if not valid_indices:
            logger.warning("No valid indices returned from the vector database search.")
            return None

        retrieved_content = self.vector_database_module.metadata.iloc[valid_indices].to_dict(orient="records")
        retrieved_content = [chunk["text"] for chunk in retrieved_content]

        if self.reranker:
            retrieved_content = self.reranker.rerank(query, retrieved_content)

        return retrieved_content

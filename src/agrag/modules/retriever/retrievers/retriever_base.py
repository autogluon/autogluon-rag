import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from agrag.constants import DOC_TEXT_KEY, EMBEDDING_KEY, LOGGER_NAME
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.vector_db.vector_database import VectorDatabaseModule

logger = logging.getLogger(LOGGER_NAME)


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
        If this is set to 0, no documents will be retrieved and the generator will be used without providing additional context.
    reranker: Reranker
        Optional Reranker object to use for reranking
    use_reranker: bool
        Whether or not to use a reranker.
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
        use_reranker: bool = True,
        **kwargs,
    ):
        self.embedding_module = embedding_module
        self.vector_database_module = vector_database_module
        self.top_k = top_k
        self.reranker = None
        if use_reranker:
            assert isinstance(reranker, Reranker), "reranker must be of type <class> Reranker"
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
        if self.embedding_module.model_platform == "bedrock" and "cohere" in self.embedding_module.model_name:
            self.embedding_module.bedrock_embedding_params["input_type"] = "search_query"
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

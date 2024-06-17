import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn import DataParallel
from transformers import AutoModel, AutoTokenizer

from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.embedding.utils import normalize_embedding, pool
from agrag.modules.vector_db.vector_database import VectorDatabaseModule

logger = logging.getLogger("rag-logger")


class RetrieverModule:
    """
    Initializes the RetrieverModule with the VectorDatabaseModule.

    Parameters:
    ----------
    vector_database_module : VectorDatabaseModule
        The module containing the vector database and metadata.
    embedding_model: EmbeddingModule,
        The module for generating embeddings.
    top_k: int
        The top-k documents to retrieve (default is 5).
    reranker: Any
        An optional reranker instance to rerank the retrieved documents.
    """

    def __init__(
        self,
        vector_database_module: VectorDatabaseModule,
        embedding_module: EmbeddingModule,
        top_k: int = 5,
        reranker: Any = None,
    ):
        self.embedding_module = embedding_module

        self.normalize_embedding = normalize_embedding
        self.vector_database_module = vector_database_module
        self.top_k = top_k

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model)

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
        query_embedding = self.embedding_module.encode(data=[{"text": query}])
        query_embedding = query_embedding[0].squeeze(0).numpy()
        return query_embedding

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar document chunks to the query.

        Parameters:
        ----------
        query : str
            The query to retrieve documents for.

        Returns:
        -------
        List[str]
            A list of text chunks for the top_k most similar documents.
        """
        logger.info(f"\nRetrieving top {self.top_k} embeddings to query")
        query_embedding = self.encode_query(query)
        indices = self.vector_database_module.search_vector_database(embedding=query_embedding, top_k=self.top_k)
        print(indices)
        retrieved_docs = self.vector_database_module.metadata[indices].to_dict(orient="records")
        text_chunks = [chunk["text"] for chunk in retrieved_docs]

        if self.reranker:
            text_chunks = self.reranker.rerank(query, text_chunks)

        print(text_chunks)

        return text_chunks

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn import DataParallel
from transformers import AutoModel, AutoTokenizer

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
    hf_model : str
        The name of the Huggingface embedding model to use for encoding the query.
    hf_model_params : dict, optional
        Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
    hf_tokenizer_init_params : dict, optional
        Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
    pooling_strategy : str, optional
        The strategy to use for pooling embeddings. Options are 'mean', 'max', 'cls' (default is None).
    normalize_embeddings: bool, optional
        Whether to normalize the embeddings generated by the Embedding model. Default is `False`.
    """

    def __init__(
        self,
        vector_database_module: VectorDatabaseModule,
        hf_model: str,
        hf_model_params: Dict[str, Any] = None,
        hf_tokenizer_init_params: Dict[str, Any] = None,
        normalize_embedding: bool = False,
        pooling_strategy: str = None,
    ):
        self.hf_model = hf_model
        self.normalize_embedding = normalize_embedding
        self.pooling_strategy = pooling_strategy
        self.hf_model_params = hf_model_params or {}
        self.hf_tokenizer_init_params = hf_tokenizer_init_params or {}
        self.vector_database_module = vector_database_module
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model, **self.hf_tokenizer_init_params)
        self.model = AutoModel.from_pretrained(self.hf_model, **self.hf_model_params)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model)
        self.model.to(self.device)

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
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        query_embedding = pool(outputs.last_hidden_state, self.pooling_strategy).cpu()
        if self.normalize_embedding:
            query_embedding = normalize_embedding(query_embedding, **self.normalization_params)

        return query_embedding.numpy()

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar document chunks to the query.

        Parameters:
        ----------
        query : str
            The query to retrieve documents for.
        top_k : int, optional
            The number of top documents to retrieve (default is 10).

        Returns:
        -------
        List[str]
            A list of text chunks for the top_k most similar documents.
        """
        query_embedding = self.encode_query(query)
        indices = self.vector_database_module.search(n=1, x=query_embedding, k=top_k)
        retrieved_docs = self.vector_database_module.metadata.iloc[indices].to_dict(orient="records")
        text_chunks = [chunk["text"] for chunk in retrieved_docs]
        return text_chunks

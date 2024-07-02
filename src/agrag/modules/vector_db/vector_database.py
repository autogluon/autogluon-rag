import logging
from typing import Any, Dict, List, Union

import boto3
import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agrag.constants import EMBEDDING_KEY
from agrag.modules.vector_db.faiss.faiss_db import construct_faiss_index
from agrag.modules.vector_db.utils import SUPPORTED_SIMILARITY_FUNCTIONS, remove_duplicates

logger = logging.getLogger("rag-logger")


class VectorDatabaseModule:
    """
    A class used to construct and manage a vector database for storing embeddings.

    Attributes:
    ----------
    db_type : str
        The type of vector database to use (default is 'faiss').
    params : dict
        Additional parameters for configuring the Vector DB index.
    similarity_threshold : float
        The threshold for considering embeddings as duplicates based on a similarity function (default is 0.95)
    similarity_fn : str
        The similarity function used for determining similarity scores for embeddings. Options are 'cosine', 'euclidean', 'manhattan' (default is 'cosine').
    num_gpus: int
        Number of GPUs to use when building the index
    metadata: List[dict]
        Metadata for each embedding stored in the database
    **kwargs : dict
        Additional parameters for `VectorDatabaseModule`.

    Methods:
    -------
    construct_vector_database(embeddings: List[torch.Tensor]) -> Any:
        Constructs the vector database and stores the embeddings.

    search_vector_database(embedding: torch.Tensor, top_k: int) -> List[torch.Tensor]:
        Searches the vector database for the top k most similar embeddings to the given embedding
    """

    def __init__(
        self, db_type: str = "faiss", similarity_threshold: float = 0.95, similarity_fn: str = "cosine", **kwargs
    ) -> None:
        self.db_type = db_type
        self.params = kwargs.get("params", {})
        self.similarity_threshold = similarity_threshold
        self.similarity_fn = similarity_fn
        if self.similarity_fn not in SUPPORTED_SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"Unsupported similarity function: {self.similarity_fn}. Please choose from: {list(SUPPORTED_SIMILARITY_FUNCTIONS.keys())}"
            )
        self.num_gpus = kwargs.get("num_gpus", 0)
        self.metadata = []
        self.index = None

    def construct_vector_database(
        self,
        embeddings: pd.DataFrame,
        pbar: tqdm = None,
    ) -> Union[faiss.IndexFlatL2,]:
        """
        Constructs the vector database and stores the embeddings.

        Parameters:
        ----------
        embeddings : pd.DataFrame
            A DataFrame containing embeddings and metadata to be stored in the vector database.

        Returns:
        -------
        Union[faiss.IndexFlatL2,]
            The constructed vector database index.
        """
        logger.info("Initializing Vector DB Construction")

        self.metadata = embeddings.drop(columns=[EMBEDDING_KEY])
        if pbar:
            pbar.update(1)

        vectors = [torch.tensor(embedding) for embedding in embeddings[EMBEDDING_KEY].values]

        logger.info("\nRemoving Duplicates")
        vectors, indices_to_keep = remove_duplicates(vectors, self.similarity_threshold, self.similarity_fn)
        self.metadata = self.metadata.iloc[indices_to_keep]
        if pbar:
            pbar.update(1)

        logger.info("Constructing FAISS Index")
        if self.db_type == "faiss":
            self.index = construct_faiss_index(vectors, self.num_gpus)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        if pbar:
            pbar.update(len(embeddings))

    def search_vector_database(self, embedding: np.array, top_k: int) -> List[torch.Tensor]:
        """
        Searches the vector database for the top k most similar embeddings to the given embedding

        Parameters:
        ----------
        embedding : np.array
            Embedding of the user query. The database is searched to find the k most similar vectors to this embedding
        top_k: int
            Number of similar embeddings to search for in the database

        Returns:
        -------
        List[torch.Tensor]
            Top k most similar embeddings
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, embedding.shape[0])
        if self.db_type == "faiss":
            _, indices = self.index.search(x=embedding, k=top_k)
            return indices[0].tolist()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

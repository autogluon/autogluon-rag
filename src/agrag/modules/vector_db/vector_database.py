import logging
from typing import List, Union

import boto3
import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agrag.constants import EMBEDDING_KEY
from agrag.modules.vector_db.faiss.faiss_db import construct_faiss_index
from agrag.modules.vector_db.utils import SUPPORTED_SIMILARITY_FUNCTIONS, pad_embeddings, remove_duplicates

logger = logging.getLogger("rag-logger")


class VectorDatabaseModule:
    """
    A class used to construct and manage a vector database for storing embeddings.

    Attributes:
    ----------
    db_type : str
        The type of vector database to use (default is 'faiss').
    index : Any
        The vector database index.
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

    Methods:
    -------
    construct_vector_database(embeddings: List[torch.Tensor]) -> Any:
        Constructs the vector database and stores the embeddings.
    search_vector_database(embedding: torch.Tensor, top_k: int) -> List[torch.Tensor]:
        Searches the vector database for the top k most similar embeddings to the given embedding
    """

    def __init__(
        self,
        db_type: str = "faiss",
        params: dict = None,
        similarity_threshold: float = 0.95,
        similarity_fn: str = "cosine",
        s3_bucket: str = None,
        num_gpus: int = 0,
    ) -> None:
        self.db_type = db_type
        self.params = params if params is not None else {}
        self.similarity_threshold = similarity_threshold
        if similarity_fn not in SUPPORTED_SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"Unsupported similarity function: {similarity_fn}. Please choose from: {list(SUPPORTED_SIMILARITY_FUNCTIONS.keys())}"
            )
        self.similarity_fn = similarity_fn
        self.index = None
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client("s3") if s3_bucket else None
        self.num_gpus = num_gpus
        self.metadata = []

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
        if pbar:
            pbar.reset(total=4)

        self.metadata = embeddings.drop(columns=[EMBEDDING_KEY])
        if pbar:
            pbar.update(1)

        vectors = [torch.tensor(embedding) for embedding in embeddings[EMBEDDING_KEY].values]
        logger.info("\nRemoving Duplicates")
        if pbar:
            pbar.update(1)

        vectors, indices_to_keep = remove_duplicates(vectors, self.similarity_threshold, self.similarity_fn)
        self.metadata = self.metadata.iloc[indices_to_keep]
        logger.info("Constructing FAISS Index")
        if pbar:
            pbar.update(1)

        if self.db_type == "faiss":
            self.index = construct_faiss_index(vectors, self.num_gpus)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        if pbar:
            pbar.update(1)
            pbar.close()

        return self.index

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

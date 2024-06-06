import logging
from typing import Any, List, Union

import faiss
import torch

from agrag.modules.vector_db.faiss.faiss import construct_faiss_index
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
        The threshold for considering embeddings as duplicates based on cosine similarity (default is 0.95)
    similarity_fn : str
        The name of the similarity function for determining duplicate embeddings. Options are 'cosine', 'euclidean', 'manhattan' (default is 'cosine').

    Methods:
    -------
    construct_vector_database(embeddings: List[torch.Tensor]) -> Any:
        Constructs the vector database and stores the embeddings.
    """

    def __init__(
        self,
        db_type: str = "faiss",
        params: dict = None,
        similarity_threshold: float = 0.95,
        similarity_fn: str = "cosine",
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

    def construct_vector_database(self, embeddings: List[torch.Tensor]) -> Union[faiss.IndexFlatL2, Any]:
        """
        Constructs the vector database and stores the embeddings.

        Parameters:
        ----------
        embeddings : List[torch.Tensor]
            A list of embeddings to be stored in the vector database.

        Returns:
        -------
        Union[faiss.IndexFlatL2, Any]
            The constructed vector database index.
        """
        embeddings = pad_embeddings(embeddings)
        embeddings = remove_duplicates(embeddings, self.similarity_threshold, self.similarity_fn)
        if self.db_type == "faiss":
            self.index = construct_faiss_index(embeddings, self.params.get("gpu", False))
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        return self.index

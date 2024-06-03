from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from agrag.modules.embedding.utils import pool


class EmbeddingModule:
    """
    A class used to generate embeddings for text dat.

    Attributes:
    ----------
    model_name : str
        The name of the Huggingface model to use for generating embeddings.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the Huggingface model.
    model : transformers.PreTrainedModel
        The Huggingface model used for generating embeddings.
    pooling_strategy : str
        The strategy used for pooling embeddings. Options are 'average', 'max', 'cls'.
        If no option is provided, will default to using no pooling method.

    Methods:
    -------
    create_embeddings(data: List[str]) -> List[torch.Tensor]:
        Generates embeddings for a list of text data chunks.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en", pooling_strategy: str = None):
        """
        Initializes the EmbeddingModule with a Huggingface model.

        Parameters:
        ----------
        model_name : str, optional
            The name of the Huggingface model to use for generating embeddings (default is "BAAI/bge-large-en").
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy

    def create_embeddings(self, data: List[str]) -> List[torch.Tensor]:
        """
        Generates embeddings for a list of text data chunks.

        Parameters:
        ----------
        data : List[str]
            A list of text data chunks to generate embeddings for.

        Returns:
        -------
        List[torch.Tensor]
            A list of embeddings corresponding to the input data chunks.
        """
        embeddings = []
        for text in data:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = pool(outputs.last_hidden_state, self.pooling_strategy)
            embeddings.append(embedding)
        return embeddings

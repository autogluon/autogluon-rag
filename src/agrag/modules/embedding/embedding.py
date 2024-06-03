import logging
from typing import List, Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from agrag.modules.embedding.utils import pool

logger = logging.getLogger("rag-logger")


class EmbeddingModule:
    """
    A class used to generate embeddings for text dat.

    Attributes:
    ----------
    model_name : str
        The name of the Huggingface model or SentenceTransformer to use for generating embeddings.
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

    def __init__(
        self,
        hf_model: str = "BAAI/bge-large-en",
        st_model: str = "paraphrase-MiniLM-L6-v2",
        pooling_strategy: str = None,
    ):
        self.sentence_transf = False
        self.hf_model = hf_model
        self.st_model = st_model
        if st_model == "sentence_transformer":
            self.model = SentenceTransformer(self.st_model)
            self.sentence_transf = True
        else:
            logger.info(f"Default to using Huggingface since no model was provided.")
            logger.info(f"Using Huggingface Model: {self.hf_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            self.model = AutoModel.from_pretrained(self.hf_model)
        self.pooling_strategy = pooling_strategy

    def create_embeddings(self, data: List[str]) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Generates embeddings for a list of text data chunks.

        Parameters:
        ----------
        data : List[str]
            A list of text data chunks to generate embeddings for.

        Returns:
        -------
        Union[List[torch.Tensor], torch.Tensor]
            A list of embeddings corresponding to the input data chunks if pooling_strategy is 'none',
            otherwise a single tensor with the pooled embeddings.
        """
        if self.sentence_transf:
            embeddings = self.model.encode(data, convert_to_tensor=True)
            embeddings = pool(embeddings, self.pooling_strategy)
        else:
            embeddings = []
            for text in data:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = pool(outputs.last_hidden_state, self.pooling_strategy)
                embeddings.append(embedding)
        if not self.pooling_strategy:
            return embeddings
        else:
            # Combine pooled embeddings into a single tensor
            return torch.cat(embeddings, dim=0)

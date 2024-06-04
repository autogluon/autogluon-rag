import logging
from typing import Any, Dict, List, Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from agrag.modules.embedding.utils import pool

logger = logging.getLogger("rag-logger")


class EmbeddingModule:
    """
    A class used to generate embeddings for text data from documents.

    Attributes:
    ----------
    hf_model : str, optional
        The name of the Huggingface model to use for generating embeddings (default is "BAAI/bge-large-en").
    use_sentence_transf: bool, optional
        Whether or not to use the SentenceTransformer model
    st_model : str, optional
        The name of the SentenceTransformer model to use for generating embeddings (default is "paraphrase-MiniLM-L6-v2").
    pooling_strategy : str, optional
        The strategy to use for pooling embeddings. Options are 'mean', 'max', 'cls' (default is None).
    hf_model_params : dict, optional
        Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
    hf_tokenizer_init_params : dict, optional
        Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
    st_params : dict, optional
        Additional parameters to pass to the `SentenceTransformer` initializer method.
    hf_tokenizer_params : dict, optional
        Additional parameters to pass to the `tokenizer` method for the Huggingface model.
    st_encode_params : dict, optional
        Additional parameters to pass to the SentenceTransformer `encode` method.
    hf_forward_params : dict, optional
        Additional parameters to pass to the Huggingface model's `forward` method.

    Methods:
    -------
    create_embeddings(data: List[str]) -> List[torch.Tensor]:
        Generates embeddings for a list of text data chunks.
    """

    def __init__(
        self,
        hf_model: str = "BAAI/bge-large-en",
        use_sentence_transf: bool = False,
        st_model: str = "paraphrase-MiniLM-L6-v2",
        pooling_strategy: str = None,
        hf_model_params: Dict[str, Any] = None,
        hf_tokenizer_init_params: Dict[str, Any] = None,
        st_params: Dict[str, Any] = None,
        hf_tokenizer_params: Dict[str, Any] = None,
        st_encode_params: Dict[str, Any] = None,
        hf_forward_params: Dict[str, Any] = None,
    ):
        self.use_sentence_transf = use_sentence_transf
        self.hf_model = hf_model
        self.st_model = st_model
        self.hf_model_params = hf_model_params or {}
        self.hf_tokenizer_init_params = hf_tokenizer_init_params or {}
        self.st_params = st_params or {}
        self.hf_tokenizer_params = hf_tokenizer_params or {}
        self.st_encode_params = st_encode_params or {}
        self.hf_forward_params = hf_forward_params or {}
        if self.use_sentence_transf:
            self.model = SentenceTransformer(self.st_model, **self.st_params)
            self.use_sentence_transf = True
        else:
            logger.info(f"Default to using Huggingface since no model was provided.")
            logger.info(f"Using Huggingface Model: {self.hf_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model, **self.hf_tokenizer_init_params)
            self.model = AutoModel.from_pretrained(self.hf_model, **self.hf_model_params)
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
        if self.use_sentence_transf:
            embeddings = self.model.encode(data, convert_to_tensor=True, **self.st_encode_params)
            embeddings = pool(embeddings, self.pooling_strategy)
        else:
            embeddings = []
            for text in data:
                inputs = self.tokenizer(text, return_tensors="pt", **self.hf_tokenizer_params)
                with torch.no_grad():
                    outputs = self.model(**inputs, **self.hf_forward_params)
                embedding = pool(outputs.last_hidden_state, self.pooling_strategy)
                embeddings.append(embedding)
        if not self.pooling_strategy:
            return embeddings
        else:
            # Combine pooled embeddings into a single tensor
            return torch.cat(embeddings, dim=0)

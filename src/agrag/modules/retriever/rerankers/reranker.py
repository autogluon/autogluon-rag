import logging
from typing import List

import torch
from torch.nn import DataParallel
from transformers import AutoModel, AutoTokenizer

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Reranker:
    """
    A unified reranker class that initializes and uses any model from Huggingface for reranking.

    Attributes:
    ----------
    model_name : str
        The name of the Huggingface model to use for the reranker (default is "BAAI/bge-large-en").
    model_platform: str
        The name of the platform where the model is hosted. Currently only Huggingface ("huggingface") models are supported.
    platform_args: dict
        Additional platform-specific parameters to use when initializing the model, reranking, etc.
    batch_size : int
        The size of the batch. If you have limited CUDA memory, decrease the size of the batch (default is 64).
    num_gpus: int
        Number of GPUs to use for generating responses. If no value is provided, the maximum available GPUs will be used. 
        Otherwise, the minimum of the provided value and maximum available GPUs will be used.
    top_k: int,
        The top-k documents to use as context for generation (default is 10).
    **kwargs : dict
        Additional parameters for `Reranker`.

    Methods:
    -------
    rerank(query: str, text_chunks: List[str]) -> List[str]:
        Reranks the text chunks based on their relevance to the query.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en",
        top_k: int = 10,
        model_platform: str = "huggingface",
        platform_args: dict = {},
        **kwargs,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.model_platform = model_platform
        self.platform_args = platform_args

        self.batch_size = kwargs.get("batch_size", 64)
        self.num_gpus = kwargs.get("num_gpus", 0)
        self.device = "cpu" if not self.num_gpus else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_platform == "huggingface":
            self.hf_model_params = self.platform_args.get("hf_model_params", {})
            self.hf_tokenizer_init_params = self.platform_args.get("hf_tokenizer_init_params", {})
            self.hf_tokenizer_params = self.platform_args.get("hf_tokenizer_params", {})
            self.hf_forward_params = self.platform_args.get("hf_forward_params", {})
        else:
            raise NotImplementedError(f"Unsupported platform type: {model_platform}")

        self.model = AutoModel.from_pretrained(self.model_name, **self.hf_model_params).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.hf_tokenizer_init_params)

        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model, device_ids=list(range(self.num_gpus)))
            self.model = self.model.to("cuda" if self.num_gpus > 0 else "cpu")

        self.model.eval()

    def rerank(self, query: str, text_chunks: List[str]) -> List[str]:
        """
        Reranks the given text chunks based on their relevance to the query.

        Parameters:
        ----------
        query : str
            The query string for which the text chunks need to be reranked.
        text_chunks : List[str]
            The list of text chunks to be reranked.

        Returns:
        -------
        List[str]
            A list of text chunks sorted by their relevance to the query.
        """
        scores = []

        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i : i + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                **self.hf_tokenizer_params,
            )
            if self.num_gpus > 1:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, **self.hf_forward_params, return_dict=True)
                batch_scores = outputs[0][:, 0].cpu().numpy().tolist()
            scores.extend(batch_scores)

        scored_chunks = list(zip(text_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        sorted_text_chunks = [chunk for chunk, score in scored_chunks]

        return sorted_text_chunks[: self.top_k]

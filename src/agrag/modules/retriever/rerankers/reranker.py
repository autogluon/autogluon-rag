import logging
from typing import List

import torch
from torch.nn import DataParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


# Sensible tokenizer defaults so the reranker works out of the box even when the
# caller supplies no ``hf_tokenizer_params``. A cross-encoder needs padding (to
# batch variable-length pairs), truncation with a max length (to stay within the
# model's context), and PyTorch tensors. User-provided keys always win.
_DEFAULT_TOKENIZER_PARAMS = {
    "padding": True,
    "truncation": True,
    "max_length": 512,
    "return_tensors": "pt",
}


class Reranker:
    """
    A unified reranker class that initializes and uses a Huggingface cross-encoder for reranking.

    The model is loaded with ``AutoModelForSequenceClassification`` and must expose a
    relevance-scoring head (e.g. ``BAAI/bge-reranker-large``): for each (query, document)
    pair it produces a single scalar relevance logit. Documents are ranked by that scalar.

    Attributes:
    ----------
    model_name : str
        The name of the Huggingface cross-encoder to use for the reranker
        (default is "BAAI/bge-reranker-large").
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
        model_name: str = "BAAI/bge-reranker-large",
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
            # Merge caller-provided tokenizer params over safe defaults so the
            # reranker tokenizes correctly even when no config is supplied.
            self.hf_tokenizer_params = {**_DEFAULT_TOKENIZER_PARAMS, **self.platform_args.get("hf_tokenizer_params", {})}
            self.hf_forward_params = self.platform_args.get("hf_forward_params", {})
        else:
            raise NotImplementedError(f"Unsupported platform type: {model_platform}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **self.hf_model_params
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.hf_tokenizer_init_params)

        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model, device_ids=list(range(self.num_gpus)))
            self.model = self.model.to("cuda" if self.num_gpus > 0 else "cpu")

        self.model.eval()

    def rerank(self, query: str, text_chunks: List[str], return_scores: bool = False):
        """
        Reranks the given text chunks based on their relevance to the query.

        Parameters:
        ----------
        query : str
            The query string for which the text chunks need to be reranked.
        text_chunks : List[str]
            The list of text chunks to be reranked.
        return_scores : bool
            If False (default), returns the sorted text chunks only, preserving
            the original behavior. If True, returns a list of ``(chunk, score)``
            tuples sorted by relevance, so callers (the agentic retrieval path)
            can populate ``rerank_score`` on each ``Evidence``.

        Returns:
        -------
        List[str] or List[Tuple[str, float]]
            Text chunks sorted by relevance, or ``(chunk, score)`` tuples when
            ``return_scores`` is True. Both are truncated to ``top_k``.
        """
        scores = []

        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i : i + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                **self.hf_tokenizer_params,
            )
            # Move inputs to the model's device whenever it is on CUDA. This must
            # happen for a single GPU too, not only for multi-GPU DataParallel.
            if str(self.device) != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, **self.hf_forward_params, return_dict=True)
                # Cross-encoder head: logits are (batch, num_labels). A reranker
                # head has num_labels == 1, so squeeze the last dim to get one
                # scalar relevance score per (query, document) pair. If a model
                # exposes multiple labels, use the first as the relevance score.
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    batch_scores = logits[:, 0]
                batch_scores = batch_scores.float().cpu().numpy().tolist()
            scores.extend(batch_scores)

        scored_chunks = list(zip(text_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        if return_scores:
            return scored_chunks[: self.top_k]

        sorted_text_chunks = [chunk for chunk, score in scored_chunks]

        return sorted_text_chunks[: self.top_k]

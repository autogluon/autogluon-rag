import logging
from typing import Any, Dict, List

import torch
from torch.nn import DataParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("rag-logger")


class Reranker:
    """
    A unified reranker class that initializes and uses any model from Huggingface for reranking.

    Parameters:
    ----------
    model_name : str
        The name of the Huggingface model to use for the reranker (default is "BAAI/bge-large-en").
    batch_size : int
        The size of the batch. If you have limited CUDA memory, decrease the size of the batch (default is 64).

    Methods:
    -------
    rerank(query: str, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        Reranks the text chunks based on their relevance to the query.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i : i + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                batch_scores = outputs.logits.view(-1).float().cpu().numpy()
            scores.extend(batch_scores)

        scored_chunks = list(zip(text_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        sorted_text_chunks = [chunk for chunk, score in scored_chunks]

        return sorted_text_chunks

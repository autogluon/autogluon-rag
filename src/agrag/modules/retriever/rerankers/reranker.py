import logging
from typing import Any, Dict, List

import torch
from torch.nn import DataParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("rag-logger")


class Reranker:
    """
    A unified reranker class that initializes and uses any model from Hugging Face for reranking.

    Parameters:
    ----------
    model_name : str
        The model name to use for the reranker.
    batch_size : int
        The size of the batch. If you have limited CUDA memory, decrease the size of the batch (default is 64).

    Methods:
    -------
    rerank(query: str, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        Reranks the text chunks based on their relevance to the query.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rerank(self, query: str, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scores = []
        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i : i + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                [chunk["text"] for chunk in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.softmax(dim=-1)[:, 1].cpu().numpy()
            scores.extend(batch_scores)

        for i, chunk in enumerate(text_chunks):
            chunk["score"] = scores[i]

        text_chunks.sort(key=lambda x: x["score"], reverse=True)
        return text_chunks

from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FlagEmbeddingReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", batch_size: int = 64):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

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
            with torch.no_grad():
                inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.softmax(dim=-1)[:, 1].cpu().numpy()
            scores.extend(batch_scores)
        for i, chunk in enumerate(text_chunks):
            chunk["score"] = scores[i]
        text_chunks.sort(key=lambda x: x["score"], reverse=True)
        return text_chunks

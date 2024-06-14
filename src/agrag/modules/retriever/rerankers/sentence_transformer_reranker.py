from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


class SentenceTransformerReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2", batch_size: int = 64):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(self, query: str, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inputs = [(query, chunk["text"]) for chunk in text_chunks]
        scores = self.model.predict(inputs, batch_size=self.batch_size)
        for i, chunk in enumerate(text_chunks):
            chunk["score"] = scores[i]
        text_chunks.sort(key=lambda x: x["score"], reverse=True)
        return text_chunks

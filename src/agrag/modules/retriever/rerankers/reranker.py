from agrag.modules.retriever.rerankers.flag_embedding_reranker import FlagEmbeddingReranker
from agrag.modules.retriever.rerankers.sentence_transformer_reranker import SentenceTransformerReranker


class Reranker:
    """
    A factory class to initialize different types of rerankers.

    Parameters:
    ----------
    reranker_type : str
        The type of reranker to initialize.
    model_name : str
        The model name to use for the reranker (default is based on reranker type).
    sentence_transformer_max_length : int
        The maximum length of the input text for Sentence Transformer Reranker (default is 512).
    use_fp16 : bool
        Whether to use fp16 for Flag Embedding Reranker (default is False).

    Returns:
    -------
    Any
        An instance of the specified reranker type.
    """

    def __init__(
        self,
        reranker_type: str,
        model_name: str = None,
        batch_size: int = 64,
    ):
        self.reranker_type = reranker_type
        self.model_name = model_name
        self.batch_size = batch_size

    def get_reranker(self):
        if self.reranker_type == "sentence_transformer":
            return SentenceTransformerReranker(model_name=self.model_name, batch_size=self.batch_size)
        elif self.reranker_type == "flag_embedding":
            return FlagEmbeddingReranker(model_name=self.model_name, batch_size=self.batch_size)
        else:
            raise ValueError(f"Unsupported reranker type: {self.reranker_type}")

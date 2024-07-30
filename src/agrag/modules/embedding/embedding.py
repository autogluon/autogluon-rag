import json
import logging
from typing import List, Union

import boto3
import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from agrag.constants import DOC_TEXT_KEY, EMBEDDING_HIDDEN_DIM_KEY, EMBEDDING_KEY
from agrag.modules.embedding.utils import get_embeddings_bedrock, normalize_embedding, pool

logger = logging.getLogger("rag-logger")


class EmbeddingModule:
    """
    A class used to generate embeddings for text data from documents.

    Attributes:
    ----------
    model_name : str
        The name of the Huggingface or Bedrock model to use for generating embeddings (default is "BAAI/bge-large-en" from Huggingface).
    pooling_strategy : str
        The strategy to use for pooling embeddings. Options are 'mean', 'max', 'cls' (default is None).
    normalize_embeddings: bool
        Whether to normalize the embeddings generated by the Embedding model. Default is `False`.
    hf_model_params : dict
        Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
    hf_tokenizer_init_params : dict
        Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
    hf_tokenizer_params : dict
        Additional parameters to pass to the `tokenizer` method for the Huggingface model.
    hf_forward_params : dict
        Additional parameters to pass to the Huggingface model's `forward` method.
    normalization_params: dict
        Additional parameters to pass to the PyTorch `nn.functional.normalize` method.
    query_instruction_for_retrieval: str
        Instruction for query when using embedding model.
    use_bedrock: str
        Whether to use the provided model from AWS Bedrock API. https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
        Currently only Cohere (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html) and Amazon Titan (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan.html) embedding models are supported.
    bedrock_embedding_params: dict
        Additional parameters to pass into the model when generating the embeddings.
    bedrock_aws_region: str
        AWS region where the model is hosted on Bedrock.

    Methods:
    -------
    encode(data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        Generates embeddings for a list of text data chunks in batches.

    encode_queries(queries: Union[List[str], str]) -> np.ndarray:
        Generates embeddings for a list of queries.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en",
        pooling_strategy: str = None,
        normalize_embeddings: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        self.hf_model_params = kwargs.get("hf_model_params", {})
        self.hf_tokenizer_init_params = kwargs.get("hf_tokenizer_init_params", {})
        self.hf_tokenizer_params = kwargs.get("hf_tokenizer_params", {})
        self.hf_forward_params = kwargs.get("hf_forward_params", {})
        self.normalization_params = kwargs.get("normalization_params", {})
        self.query_instruction_for_retrieval = kwargs.get("query_instruction_for_retrieval", None)
        self.num_gpus = kwargs.get("num_gpus", 0)
        self.device = "cpu" if not self.num_gpus else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bedrock = kwargs.get("use_bedrock")
        if self.use_bedrock:
            if not "embed" in self.model_name:
                raise ValueError(
                    f"Invalid model_id {self.model_name}. Must use an embedding model from Bedrock. The model_id should contain 'embed'."
                )
            logger.info(f"Using Bedrock Model {self.model_name} for Embedding Module")
            self.bedrock_embedding_params = kwargs.get("bedrock_embedding_params", {})
            if "cohere" in self.model_name:
                self.bedrock_embedding_params["input_type"] = "search_document"
            self.client = boto3.client("bedrock-runtime", region_name=kwargs.get("bedrock_aws_region", None))
        else:
            logger.info(f"Using Huggingface Model {self.model_name} for Embedding Module")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.hf_tokenizer_init_params)
            self.model = AutoModel.from_pretrained(self.model_name, **self.hf_model_params)
            if self.num_gpus > 1:
                logger.info(f"Using {self.num_gpus} GPUs")
                self.model = DataParallel(self.model)
            self.model.to(self.device)

    def encode(self, data: pd.DataFrame, pbar: tqdm = None, batch_size: int = 32) -> pd.DataFrame:
        """
        Generates embeddings for a list of text data chunks in batches.

        Parameters:
        ----------
        data : pd.DataFrame
            A table of text data chunks to generate embeddings for.
        pbar : tqdm
            A tqdm progress bar to show progress.
        batch_size : int
            The batch size to use for encoding (default is 32).

        Returns:
        -------
        pd.DataFrame
            The input DataFrame with an additional column for the embeddings.

        Example:
        --------
        data = pd.DataFrame({DOC_TEXT_KEY: ["This is a test sentence.", "This is another test sentence."]})
        embeddings = encode(data)
        """

        texts = data[DOC_TEXT_KEY].tolist()
        all_embeddings = []
        all_embeddings_hidden_dim = []

        logger.info(f"Using batch size {batch_size}")

        batch_num = 1

        for i in range(0, len(texts), batch_size):
            logger.info(f"Embedding Batch {batch_num}")

            logger.info("\nTokenizing text chunks")
            batch_texts = texts[i : i + batch_size]

            logger.info("\nGenerating embeddings")
            if self.use_bedrock:
                batch_embeddings = get_embeddings_bedrock(
                    batch_texts=batch_texts,
                    client=self.client,
                    model_id=self.model_name,
                    embedding_params=self.bedrock_embedding_params,
                )
            else:
                inputs = self.tokenizer(batch_texts, return_tensors="pt", **self.hf_tokenizer_params)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs, **self.hf_forward_params)

                # The first element in the tuple returned by the model is the embeddings generated
                # The tuple elements are (embeddings, hidden_states, past_key_values, attentions, cross_attentions)
                batch_embeddings = outputs[0]

            logger.info("\nProcessing embeddings")

            batch_embeddings = pool(batch_embeddings, self.pooling_strategy)
            if self.normalize_embeddings:
                batch_embeddings = normalize_embedding(batch_embeddings, **self.normalization_params)

            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.cpu().numpy()
            else:
                batch_embeddings = np.array(batch_embeddings)
            all_embeddings.extend(batch_embeddings)
            all_embeddings_hidden_dim.extend([batch_embeddings.shape[-1]] * batch_embeddings.shape[0])

            if pbar is not None:
                pbar.update(len(batch_texts))

            batch_num += 1

        if pbar is not None:
            pbar.close()

        data[EMBEDDING_KEY] = all_embeddings
        data[EMBEDDING_HIDDEN_DIM_KEY] = all_embeddings_hidden_dim

        return data

    def encode_queries(self, queries: Union[List[str], str]) -> np.ndarray:
        """
        Function is used as written in: https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/flag_models.py
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text

        Parameters:
        ----------
        queries : Union[List[str], str]
            A list of queries to generate embeddings for.

        Returns:
        -------
        Union[List[torch.Tensor], torch.Tensor]
            A list of embeddings corresponding to the input queries if pooling_strategy is 'none',
            otherwise a single tensor with the pooled embeddings.
        """
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ["{}{}".format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(pd.DataFrame({DOC_TEXT_KEY: input_texts}))

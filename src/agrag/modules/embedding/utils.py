import logging
from typing import Dict, List

import boto3
import torch
from torch.nn import functional as F

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)
import json

from agrag.modules.embedding.constants import COHERE_MAX_TOKENS, TITAN_MAX_TOKENS


def pool(embeddings: List[torch.Tensor], pooling_strategy: str) -> List[torch.Tensor]:
    """
    Applies the specified pooling strategy to the embeddings.
    The pooling strategies supported are:
    - 'mean': Mean pooling of token embeddings.
    - 'max': Max pooling of token embeddings.
    - 'cls': Using the embedding of the CLS token.
    - None: No pooling, the input embeddings are returned as is.

    Parameters:
    ----------
    embeddings : List[torch.Tensor]
        A list of token embeddings generated by the Huggingface model. Each element in the list is a tensor of shape
        [batch_size, sequence_length, hidden_size].

    Returns:
    -------
    torch.Tensor
        A tensor of pooled embeddings according to the specified strategy. The output shape depends on the pooling
        strategy:
        - 'mean', 'cls', and 'max': [batch_size, hidden_size]
        - None: [batch_size, sequence_length, hidden_size]

    Example:
    --------
    output = self.model(input)
    embedding = output.last_hidden_state
    embedding = pool(embedding, 'mean')
    """
    if pooling_strategy == "mean":
        embeddings = embeddings.mean(dim=1)
    elif pooling_strategy == "max":
        embeddings = embeddings.max(dim=1).values
    elif pooling_strategy == "cls":
        embeddings = embeddings[:, 0]
    elif pooling_strategy:
        raise NotImplementedError("Provided pooling strategy not implemented")
    return embeddings


def normalize_embedding(embeddings, args=None):
    """
    Normalizes the input tensor (embedding).

    This function normalizes the input embeddings along a specified dimension using the specified parameters.
    It wraps the `torch.nn.functional.normalize` function, which applies Lp normalization over a specified dimension.

    Parameters:
    ----------
    embeddings : torch.Tensor
        The input tensor containing the embeddings to be normalized.
    args : dict
        Additional arguments to be passed to `torch.nn.functional.normalize`. This can include:
        - p (float): The exponent value in the norm formulation. Default: 2.
        - dim (int): The dimension to reduce. Default: 1.
        - eps (float): A small value to avoid division by zero. Default: 1e-12.

    Returns:
    -------
    torch.Tensor
        A tensor containing the normalized embeddings.

    Example:
    --------
    embeddings = torch.rand(10, 100)
    args = {'p': 2, 'dim': 1, 'eps': 1e-12}
    normalized_embeddings = normalize(embeddings, args)
    """
    return F.normalize(embeddings, **args)


def get_embeddings_bedrock(
    batch_texts: List[str], client: boto3.client, model_id: str, embedding_params: dict = {}
) -> List[float]:
    embeddings = []
    if "titan" in model_id:
        batch_texts = [text[:TITAN_MAX_TOKENS] for text in batch_texts]
        for text in batch_texts:
            body = json.dumps(
                {
                    "inputText": text,
                    **embedding_params,
                }
            )
            response = client.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
            )
            outputs = json.loads(response["body"].read())
            embeddings.append(outputs.get("embedding"))
    elif "cohere" in model_id:
        batch_texts = [text[:COHERE_MAX_TOKENS] for text in batch_texts]
        body = json.dumps(
            {
                "texts": batch_texts,
                **embedding_params,
            }
        )
        response = client.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        outputs = json.loads(response["body"].read())
        embeddings = outputs.get("embeddings")
    else:
        raise NotImplementedError(f"Unsupported Embedding Model for Bedrock {model_id}")
    return embeddings


def extract_response(output: Dict) -> str:
    """
    Extracts the response embeddings from the model output.

    Parameters:
    ----------
    output : Dict
        The output dictionary from the Bedrock model.

    Returns:
    -------
    str
        The extracted response text.
    """
    # Used for Mistral response
    if "outputs" in output and isinstance(output["outputs"], list) and "embedding" in output["outputs"][0]:
        return output["outputs"][0]["embedding"]
    # Used for Anthropic response
    elif "content" in output and output["type"] == "message":
        return output["content"][0]["embedding"]
    # Used for Llama response
    elif "generation" in output:
        return output["generation"]
    else:
        raise ValueError("Unknown output structure: %s", output)

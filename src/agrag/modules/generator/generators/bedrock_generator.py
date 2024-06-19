import json
import logging
from typing import Dict, List

import boto3

logger = logging.getLogger("rag-logger")


class BedrockGenerator:
    """
    A class used to generate responses based on a query and a given context using AWS Bedrock.

    Attributes:
    ----------
    model_name : str
        The name of the Bedrock model to use for response generation.
    bedrock_generate_params : dict, optional
        Additional parameters to pass to the Bedrock generate API method.

    Methods:
    -------
    generate_response(query: str, context: List[str]) -> str:
        Generates a response based on the query and context.
    """

    def __init__(
        self,
        model_name: str,
        bedrock_generate_params: Dict = None,
    ):
        self.model_name = model_name
        self.bedrock_generate_params = bedrock_generate_params or {}
        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")

        logger.info(f"Using AWS Bedrock Model {self.model_name} for Generator Module")

    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generates a response based on the query and context.

        Parameters:
        ----------
        query : str
            The user query.
        context : List[str]
            A list of context text chunks to be included in the query.

        Returns:
        -------
        str
            The generated response.
        """
        combined_context = "\n".join(context)
        final_query = f"{query}\n\nHere is some useful context:\n{combined_context}"

        body = json.dumps({"prompt": final_query, **self.bedrock_generate_params})

        accept = "application/json"
        contentType = "application/json"

        output = self.client.invoke_model(body=body, modelId=self.model_name, accept=accept, contentType=contentType)

        output = json.loads(output.get("body").read())
        response = output["outputs"][0]["text"]
        return response

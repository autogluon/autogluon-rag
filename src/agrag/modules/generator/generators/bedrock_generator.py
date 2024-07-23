import json
import logging
from typing import Dict

import boto3

logger = logging.getLogger("rag-logger")


class BedrockGenerator:
    """
    A class used to generate responses based on a query and a given context using AWS Bedrock.
    Refer to https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html for supported models in Bedrock.

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
        aws_region: str = None,
        bedrock_generate_params: Dict = None,
    ):
        self.model_name = model_name
        self.bedrock_generate_params = bedrock_generate_params or {}
        self.client = boto3.client("bedrock-runtime", region_name=aws_region)

        logger.info(f"Using AWS Bedrock Model {self.model_name} for Generator Module")

    def generate_response(self, query: str) -> str:
        """
        Generates a response based on the query.

        Parameters:
        ----------
        query : str
            The user query.

        Returns:
        -------
        str
            The generated response.
        """

        body = json.dumps({"prompt": query, **self.bedrock_generate_params})

        accept = "application/json"
        contentType = "application/json"

        output = self.client.invoke_model(body=body, modelId=self.model_name, accept=accept, contentType=contentType)

        output = json.loads(output.get("body").read())
        response = self.extract_response(output)
        return response

    def extract_response(self, output: Dict) -> str:
        """
        Extracts the response text from the model output.

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
        if "outputs" in output and isinstance(output["outputs"], list) and "text" in output["outputs"][0]:
            return output["outputs"][0]["text"].strip()
        # Used for Anthropic response
        elif "type" in output and output["type"] == "completion":
            return output["completion"].strip()
        # Used for Llama response
        elif "generation" in output:
            return output["generation"].strip()
        else:
            logger.error("Unknown output structure: %s", output)
            return ""

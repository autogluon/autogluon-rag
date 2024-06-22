import logging
from typing import Dict

from openai import OpenAI

logger = logging.getLogger("rag-logger")


class GPTGenerator:
    """
    A class to generate responses using OpenAI's GPT models.
    Refer to https://platform.openai.com/docs/models for supported models through the OpenAI API.

    Attributes:
    ----------
    model_name : str
        The name of the GPT model to use for response generation.
    openai_api_key : str
        The API key for accessing the OpenAI models.
    gpt_generate_params : dict, optional
        Additional parameters to pass to the OpenAI generate API method.
    client : OpenAI
        The OpenAI client for interacting with the OpenAI API.

    Methods:
    -------
    generate_response(query: str) -> str:
        Generates a response based on the provided query.
    """

    def __init__(
        self,
        model_name: str,
        openai_api_key: str,
        gpt_generate_params: Dict = None,
    ):
        """
        Initializes the GPTGenerator with the specified model and parameters.
        This uses the GPT model provided by OpenAI.
        Refer to https://platform.openai.com/docs/models.
        The model will be used in "Chat" mode (https://platform.openai.com/docs/guides/text-generation/chat-completions-api).

        Parameters:
        ----------
        model_name : str
            The name of the GPT model to use for generating responses.
        openai_api_key : str
            The API key for accessing OpenAI models.
        gpt_generate_params : Dict, optional
            Additional parameters for the OpenAI generate API method.
        """
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.gpt_generate_params = gpt_generate_params or {}
        self.client = OpenAI(api_key=self.openai_api_key)

        logger.info(f"Using OpenAI GPT Model {self.model_name} for GPT Generator")

    def generate_response(self, query: str) -> str:
        """
        Generates a response based on the provided query.

        Parameters:
        ----------
        query : str
            The user query for which a response is to be generated.

        Returns:
        -------
        str
            The generated response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            **self.gpt_generate_params,
        )

        return response.choices[0].message.content.strip()

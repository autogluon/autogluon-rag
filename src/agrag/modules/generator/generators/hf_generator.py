import logging
from typing import Dict

import torch
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agrag.constants import LOGGER_NAME

logger = logging.getLogger("AutoGluon-RAG-logger")


class HFGenerator:
    """
    A class to generate responses using Huggingface models.
    Refer to https://huggingface.co/models for supported models through the HuggingFace API.

    Attributes:
    ----------
    model_name : str
        The name of the Huggingface model to use for response generation.
    hf_model_params : dict, optional
        Additional parameters to pass to the Huggingface model's `from_pretrained` initializer method.
    hf_tokenizer_init_params : dict, optional
        Additional parameters to pass to the Huggingface tokenizer's `from_pretrained` initializer method.
    hf_tokenizer_params : dict, optional
        Additional parameters to pass to the `tokenizer` method for the Huggingface model.
    hf_generate_params : dict, optional
        Additional parameters to pass to the Huggingface model's `generate` method.
    num_gpus: int
        Number of GPUs to use for generation.

    Methods:
    -------
    generate_response(query: str) -> str:
        Generates a response based on the provided query.
    """

    def __init__(
        self,
        model_name: str,
        hf_model_params: Dict = None,
        hf_tokenizer_init_params: Dict = None,
        hf_tokenizer_params: Dict = None,
        hf_generate_params: Dict = None,
        num_gpus: int = 0,
        local_model_path: str = None,
    ):
        """
        Initializes the HFGenerator with the specified model and parameters.

        Parameters:
        ----------
        model_name : str
            The name of the Huggingface model to use for generating responses.
        hf_model_params : Dict, optional
            Additional parameters for the Huggingface model's `from_pretrained` initializer method.
        hf_tokenizer_init_params : Dict, optional
            Additional parameters for the Huggingface tokenizer's `from_pretrained` initializer method.
        hf_tokenizer_params : Dict, optional
            Additional parameters for the `tokenizer` method for the Huggingface model.
        hf_generate_params : Dict, optional
            Additional parameters for the Huggingface model's `generate` method.
        num_gpus : int
            Number of GPUs to use for generation.
        local_model_path : str, optional
            Path to a local model to use for generation.
        """
        self.model_name = local_model_path if local_model_path else model_name

        self.hf_model_params = hf_model_params or {}
        self.hf_tokenizer_init_params = hf_tokenizer_init_params or {}
        self.hf_tokenizer_params = hf_tokenizer_params or {}
        self.hf_generate_params = hf_generate_params or {}

        self.num_gpus = num_gpus
        self.device = "cpu" if not self.num_gpus else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using Huggingface Model {self.model_name} for HF Generator")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.hf_tokenizer_init_params)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.hf_model_params).to(self.device)

        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model, device_ids=list(range(self.num_gpus)))

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
        inputs = self.tokenizer(query, return_tensors="pt", **self.hf_tokenizer_params)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"], **self.hf_generate_params, pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

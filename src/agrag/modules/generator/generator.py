from typing import Dict

from agrag.modules.generator.generators.bedrock_generator import BedrockGenerator
from agrag.modules.generator.generators.gpt_generator import GPTGenerator
from agrag.modules.generator.generators.hf_generator import HFGenerator
from agrag.modules.generator.generators.vllm_generator import VLLMGenerator


class GeneratorModule:
    """
    A unified interface for generating responses using different types of models.

    Depending on the model name, this module can use one of several generator classes:
    - GPTGenerator for GPT-3 and GPT-4 models.
    - BedrockGenerator for AWS Bedrock models.
    - VLLMGenerator for vLLM models.
    - HFGenerator for HuggingFace models.

    Attributes:
    ----------
    model_name : str
        The name of the model to be used for generating responses.
    use_vllm : bool
        Flag indicating whether to use vLLM.
    generator : object
        The specific generator instance to be used based on the model_name.

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
        gpt_generate_params: Dict = None,
        vllm_sampling_params: Dict = None,
        use_bedrock: bool = False,
        bedrock_generate_params: Dict = None,
        num_gpus: int = 0,
        use_vllm: bool = False,
        openai_api_key: str = None,
    ):
        """
        Initializes the GeneratorModule with the specified model and parameters.

        Parameters:
        ----------
        model_name : str
            The name of the model to be used for generating responses.
        hf_model_params : Dict, optional
            Additional parameters for HuggingFace model initialization.
        hf_tokenizer_init_params : Dict, optional
            Additional parameters for HuggingFace tokenizer initialization.
        hf_tokenizer_params : Dict, optional
            Additional parameters for the HuggingFace tokenizer.
        hf_generate_params : Dict, optional
            Additional parameters for HuggingFace model generation.
        gpt_generate_params : Dict, optional
            Additional parameters for GPT model generation.
        vllm_sampling_params : Dict, optional
            Additional sampling parameters for vLLM models.
        use_bedrock : bool, optional
            Flag indicating whether to use AWS Bedrock.
        bedrock_generate_params : Dict, optional
            Additional parameters for Bedrock model generation.
        num_gpus : int, optional
            Number of GPUs to use for model inference.
        use_vllm : bool, optional
            Flag indicating whether to use vLLM.
        openai_api_key : str, optional
            API key for accessing OpenAI models.
        """
        self.model_name = model_name
        self.use_vllm = use_vllm

        if "gpt-3" in self.model_name or "gpt-4" in self.model_name:
            self.generator = GPTGenerator(
                model_name=self.model_name,
                openai_api_key=openai_api_key,
                gpt_generate_params=gpt_generate_params,
            )
        elif use_bedrock:
            self.generator = BedrockGenerator(
                model_name=self.model_name,
                bedrock_generate_params=bedrock_generate_params,
            )
        elif self.use_vllm:
            self.generator = VLLMGenerator(
                model_name=self.model_name,
                vllm_sampling_params=vllm_sampling_params,
            )
        else:
            self.generator = HFGenerator(
                model_name=self.model_name,
                hf_model_params=hf_model_params,
                hf_tokenizer_init_params=hf_tokenizer_init_params,
                hf_tokenizer_params=hf_tokenizer_params,
                hf_generate_params=hf_generate_params,
                num_gpus=num_gpus,
            )

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
        return self.generator.generate_response(query)

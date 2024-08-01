from agrag.modules.generator.generators.bedrock_generator import BedrockGenerator
from agrag.modules.generator.generators.gpt_generator import GPTGenerator
from agrag.modules.generator.generators.hf_generator import HFGenerator
from agrag.modules.generator.generators.vllm_generator import VLLMGenerator


class GeneratorModule:
    """
    A class for generating responses using different types of models.

    Attributes:
    ----------
    model_name : str
        The name of the model to be used for generating responses.
    model_platform: str
        The name of the platform where the model is hosted. Currently the following models are supported.
            - GPTGenerator for GPT-3 and GPT-4 models. Platform is "openai".
            - BedrockGenerator for AWS Bedrock models. Platform is "bedrock".
            - VLLMGenerator for vLLM models. Platform is "vllm".
            - HFGenerator for HuggingFace models. Platform is "huggingface".
    platform_args: dict
        Additional platform-specific parameters to use when initializing the model, generating text, etc.
    **kwargs : dict, optional
        Additional parameters for the generator classes.

    Methods:
    -------
    generate_response(query: str) -> str:
        Generates a response based on the provided query.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        model_platform: str = "huggingface",
        platform_args: dict = {},
        **kwargs,
    ):
        self.model_name = model_name
        self.model_platform = model_platform
        self.platform_args = platform_args

        if self.model_platform == "openai":
            self.generator = GPTGenerator(
                model_name=self.model_name,
                openai_api_key=kwargs.get("openai_api_key"),
                gpt_generate_params=self.platform_args.get("gpt_generate_params", {}),
            )
        elif self.model_platform == "bedrock":
            self.generator = BedrockGenerator(
                model_name=self.model_name,
                aws_region=kwargs.get("bedrock_aws_region", None),
                bedrock_generate_params=self.platform_args.get("bedrock_generate_params", {}),
            )
        elif self.model_platform == "vllm":
            self.generator = VLLMGenerator(
                model_name=self.model_name,
                vllm_sampling_params=self.platform_args.get("vllm_sampling_params", {}),
            )
        elif self.model_platform == "huggingface":
            self.generator = HFGenerator(
                model_name=self.model_name,
                local_model_path=kwargs.get("hf_local_model_path", None),
                hf_model_params=self.platform_args.get("hf_model_params", {}),
                hf_tokenizer_init_params=self.platform_args.get("hf_tokenizer_init_params", {}),
                hf_tokenizer_params=self.platform_args.get("hf_tokenizer_params", {}),
                hf_generate_params=self.platform_args.get("hf_generate_params", {}),
                num_gpus=kwargs.get("num_gpus", 0),
            )
        else:
            raise NotImplementedError(f"Unsupported platform type: {model_platform}")

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

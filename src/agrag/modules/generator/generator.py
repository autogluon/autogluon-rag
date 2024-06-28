from agrag.modules.generator.generators.bedrock_generator import BedrockGenerator
from agrag.modules.generator.generators.gpt_generator import GPTGenerator
from agrag.modules.generator.generators.hf_generator import HFGenerator
from agrag.modules.generator.generators.vllm_generator import VLLMGenerator


class GeneratorModule:
    """
    A class for generating responses using different types of models.

    Depending on the model name, this module will use one of the following generator classes:
    - GPTGenerator for GPT-3 and GPT-4 models.
    - BedrockGenerator for AWS Bedrock models.
    - VLLMGenerator for vLLM models.
    - HFGenerator for HuggingFace models.

    Attributes:
    ----------
    model_name : str
        The name of the model to be used for generating responses.
    generator : Union[BedrockGenerator, GPTGenerator, HFGenerator, VLLMGenerator]
        The specific generator instance to be used based on the model_name.
    **kwargs : dict, optional
        Additional parameters for the generator classes.

    Methods:
    -------
    generate_response(query: str) -> str:
        Generates a response based on the provided query.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

        if "gpt-3" in self.model_name or "gpt-4" in self.model_name:
            self.generator = GPTGenerator(
                model_name=self.model_name,
                openai_api_key=kwargs.get("openai_api_key"),
                gpt_generate_params=kwargs.get("gpt_generate_params", {}),
            )
        elif kwargs.get("use_bedrock", False):
            self.generator = BedrockGenerator(
                model_name=self.model_name,
                bedrock_generate_params=kwargs.get("bedrock_generate_params", {}),
            )
        elif kwargs.get("use_vllm", False):
            self.generator = VLLMGenerator(
                model_name=self.model_name,
                vllm_sampling_params=kwargs.get("vllm_sampling_params", {}),
            )
        else:
            self.generator = HFGenerator(
                model_name=self.model_name,
                local_model_path=kwargs.get("hf_local_model_path", None),
                hf_model_params=kwargs.get("hf_model_params", {}),
                hf_tokenizer_init_params=kwargs.get("hf_tokenizer_init_params", {}),
                hf_tokenizer_params=kwargs.get("hf_tokenizer_params", {}),
                hf_generate_params=kwargs.get("hf_generate_params", {}),
                num_gpus=kwargs.get("num_gpus", 0),
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

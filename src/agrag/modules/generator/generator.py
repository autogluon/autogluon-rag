from typing import Dict, List

from agrag.modules.generator.generators.bedrock_generator import BedrockGenerator
from agrag.modules.generator.generators.gpt_generator import GPTGenerator
from agrag.modules.generator.generators.hf_generator import HFGenerator
from agrag.modules.generator.generators.vllm_generator import VLLMGenerator


class GeneratorModule:
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
        return self.generator.generate_response(query)

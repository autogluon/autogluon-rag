import importlib.util
import logging
from typing import Dict, List

spec = importlib.util.find_spec("vllm")
found = spec is not None

if found:
    from vllm import LLM, SamplingParams

logger = logging.getLogger("rag-logger")


class VLLMGenerator:
    """
    A class used to generate responses based on a query and a given context using the vLLM library.

    Attributes:
    ----------
    model_name : str
        The name of the vLLM model to use for response generation.
    sampling_params: SamplingParams
        The sampling parameters for vLLM.

    Methods:
    -------
    generate_response(query: str, context: List[str]) -> str:
        Generates a response based on the query and context.
    """

    def __init__(
        self,
        model_name: str,
        vllm_sampling_params: Dict = None,
    ):
        self.model_name = model_name

        self.sampling_params = SamplingParams(**vllm_sampling_params) if vllm_sampling_params else SamplingParams()

        logger.info(f"Using vLLM Model {self.model_name} for VLLM Generator")
        self.llm = LLM(model=self.model_name)

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
        prompts = [final_query]

        outputs = self.llm.generate(prompts, self.sampling_params)

        generated_text = outputs[0].outputs[0].text
        return generated_text

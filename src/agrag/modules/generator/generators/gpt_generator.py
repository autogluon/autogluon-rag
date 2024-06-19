import logging
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger("rag-logger")


class GPTGenerator:
    def __init__(
        self,
        model_name: str,
        openai_api_key: str,
        gpt_generate_params: Dict = None,
    ):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.gpt_generate_params = gpt_generate_params or {}
        self.client = OpenAI(api_key=self.openai_api_key)

        logger.info(f"Using OpenAI GPT Model {self.model_name} for GPT Generator")

    def generate_response(self, query: str, context: List[str]) -> str:
        combined_context = "\n".join(context)
        final_query = f"{query}\n\nHere is some useful context:\n{combined_context}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_query},
            ],
            **self.gpt_generate_params,
        )

        return response.choices[0].message.content.strip()

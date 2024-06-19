import logging
from typing import Dict, List

import torch
from torch.nn import DataParallel
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger("rag-logger")


class HFGenerator:
    def __init__(
        self,
        model_name: str,
        hf_model_params: Dict = None,
        hf_tokenizer_init_params: Dict = None,
        hf_tokenizer_params: Dict = None,
        hf_generate_params: Dict = None,
        num_gpus: int = 0,
    ):
        self.model_name = model_name

        self.hf_model_params = hf_model_params or {}
        self.hf_tokenizer_init_params = hf_tokenizer_init_params or {}
        self.hf_tokenizer_params = hf_tokenizer_params or {}
        self.hf_generate_params = hf_generate_params or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using Huggingface Model {self.model_name} for HF Generator")

        self.model = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.hf_tokenizer_init_params)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.hf_model_params).to(self.device)

        self.num_gpus = num_gpus
        if self.num_gpus > 1:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = DataParallel(self.model, device_ids=list(range(self.num_gpus))) if self.model else None

    def generate_response(self, query: str, context: List[str]) -> str:
        combined_context = "\n".join(context)
        final_query = f"{query}\n\nHere is some useful context:\n{combined_context}"
        inputs = self.tokenizer(final_query, return_tensors="pt", **self.hf_tokenizer_params).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], **self.hf_generate_params, pad_token_id=tokenizer.eos_token_id)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # def generate_mistral_response(self, query: str, context: List[str]) -> str:
    #     combined_context = "\n".join(context)
    #     final_query = f"{query}\n\nHere is some useful context:\n{combined_context}"
    #     messages = [
    #         {"role": "system", "content": "You are a helpful chatbot assistant"},
    #         {"role": "user", "content": final_query},
    #     ]
    #     chatbot = pipeline("text-generation", model=self.model_name)
    #     return chatbot(messages)
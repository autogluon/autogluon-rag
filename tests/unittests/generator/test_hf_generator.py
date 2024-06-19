import unittest
from unittest.mock import MagicMock, patch

import torch

from agrag.modules.generator.generators.hf_generator import HFGenerator


class TestHFGenerator(unittest.TestCase):
    @patch("agrag.modules.generator.generators.hf_generator.AutoModelForCausalLM.from_pretrained")
    @patch("agrag.modules.generator.generators.hf_generator.AutoTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer, mock_model):
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        mock_model.return_value = self.mock_model

        self.model_name = "hf-model"
        self.hf_model_params = {"param1": "value1"}
        self.hf_tokenizer_init_params = {"param2": "value2"}
        self.hf_tokenizer_params = {"param3": "value3"}
        self.hf_generate_params = {"param4": "value4"}
        self.num_gpus = 1

        self.hf_generator = HFGenerator(
            model_name=self.model_name,
            hf_model_params=self.hf_model_params,
            hf_tokenizer_init_params=self.hf_tokenizer_init_params,
            hf_tokenizer_params=self.hf_tokenizer_params,
            hf_generate_params=self.hf_generate_params,
            num_gpus=self.num_gpus,
        )

    def test_generate_response(self):
        query = "What is the weather like today?"
        context = ["It is summer.", "The weather has been warm recently."]
        final_query = f"{query}\n\nHere is some useful context:\n{context[0]}\n{context[1]}"

        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        self.mock_model.generate = torch.tensor([[0.1], [0.2], [0.3]])

        self.mock_tokenizer.decode.return_value = "The weather is sunny and warm."

        response = self.hf_generator.generate_response(query, context)

        self.assertEqual(response, "The weather is sunny and warm.")


if __name__ == "__main__":
    unittest.main()

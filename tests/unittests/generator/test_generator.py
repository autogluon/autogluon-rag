import unittest
from unittest.mock import MagicMock, patch

from agrag.modules.generator.generator import GeneratorModule
from agrag.modules.generator.generators.bedrock_generator import BedrockGenerator
from agrag.modules.generator.generators.gpt_generator import GPTGenerator
from agrag.modules.generator.generators.hf_generator import HFGenerator
from agrag.modules.generator.generators.vllm_generator import VLLMGenerator


class TestGeneratorModule(unittest.TestCase):
    @patch("agrag.modules.generator.generators.gpt_generator.GPTGenerator.__init__", return_value=None)
    @patch("agrag.modules.generator.generators.bedrock_generator.BedrockGenerator.__init__", return_value=None)
    @patch("agrag.modules.generator.generators.vllm_generator.VLLMGenerator.__init__", return_value=None)
    @patch("agrag.modules.generator.generators.hf_generator.HFGenerator.__init__", return_value=None)
    def setUp(
        self,
        mock_hf_generator,
        mock_vllm_generator,
        mock_bedrock_generator,
        mock_gpt_generator,
    ):
        self.mock_hf_generator = mock_hf_generator
        self.mock_vllm_generator = mock_vllm_generator
        self.mock_bedrock_generator = mock_bedrock_generator
        self.mock_gpt_generator = mock_gpt_generator
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()

    def test_gpt_generator_initialization(self):
        model_name = "gpt-3"
        openai_api_key = "fake-api-key"
        gpt_generate_params = {"max_tokens": 100}

        generator_module = GeneratorModule(
            model_name=model_name,
            openai_api_key=openai_api_key,
            gpt_generate_params=gpt_generate_params,
        )

        self.assertIsInstance(generator_module.generator, GPTGenerator)

    def test_bedrock_generator_initialization(self):
        model_name = "bedrock-model"
        bedrock_generate_params = {"max_length": 100}

        generator_module = GeneratorModule(
            model_name=model_name,
            use_bedrock=True,
            bedrock_generate_params=bedrock_generate_params,
        )

        self.assertIsInstance(generator_module.generator, BedrockGenerator)

    def test_vllm_generator_initialization(self):
        model_name = "vllm-model"
        vllm_sampling_params = {"temperature": 0.7, "top_p": 0.9}

        generator_module = GeneratorModule(
            model_name=model_name,
            use_vllm=True,
            vllm_sampling_params=vllm_sampling_params,
        )

        self.assertIsInstance(generator_module.generator, VLLMGenerator)

    @patch("agrag.modules.generator.generators.hf_generator.AutoModelForCausalLM.from_pretrained")
    @patch("agrag.modules.generator.generators.hf_generator.AutoTokenizer.from_pretrained")
    def test_hf_generator_initialization(self, mock_hf_tokenizer, mock_hf_model):
        model_name = "hf-model"
        hf_model_params = {"param1": "value1"}
        hf_tokenizer_init_params = {"param2": "value2"}
        hf_tokenizer_params = {"param3": "value3"}
        hf_generate_params = {"param4": "value4"}

        mock_hf_tokenizer.return_value = self.mock_tokenizer
        mock_hf_model.return_value = self.mock_model

        generator_module = GeneratorModule(
            model_name=model_name,
            hf_model_params=hf_model_params,
            hf_tokenizer_init_params=hf_tokenizer_init_params,
            hf_tokenizer_params=hf_tokenizer_params,
            hf_generate_params=hf_generate_params,
            num_gpus=1,
        )

        self.assertIsInstance(generator_module.generator, HFGenerator)

    @patch("agrag.modules.generator.generators.hf_generator.AutoModelForCausalLM.from_pretrained")
    @patch("agrag.modules.generator.generators.hf_generator.AutoTokenizer.from_pretrained")
    def test_generate_response(self, mock_hf_tokenizer, mock_hf_model):
        query = "What is the weather like today?"
        context = ["It is summer.", "The weather has been warm recently."]

        mock_hf_tokenizer.return_value = self.mock_tokenizer
        mock_hf_model.return_value = self.mock_model
        generator_module = GeneratorModule(model_name="hf-model")
        mock_response = "The weather is sunny and warm."

        generator_module.generator.generate_response = MagicMock(return_value=mock_response)

        response = generator_module.generate_response(query)

        generator_module.generator.generate_response.assert_called_once_with(query)
        self.assertEqual(response, mock_response)


if __name__ == "__main__":
    unittest.main()

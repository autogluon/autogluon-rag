import unittest
from unittest.mock import MagicMock, patch

from agrag.modules.generator.generators.gpt_generator import GPTGenerator


class TestGPTGenerator(unittest.TestCase):
    @patch("agrag.modules.generator.generators.gpt_generator.OpenAI")
    def setUp(self, mock_openai):
        self.mock_openai = mock_openai
        self.model_name = "gpt-3"
        self.openai_api_key = "fake-api-key"
        self.gpt_generate_params = {"max_tokens": 100}
        self.gpt_generator = GPTGenerator(
            model_name=self.model_name,
            openai_api_key=self.openai_api_key,
            gpt_generate_params=self.gpt_generate_params,
        )
        self.mock_openai_instance = self.mock_openai.return_value
        self.mock_openai_instance.chat.completions.create = MagicMock()

    def test_generate_response(self):
        query = "What is the weather like today?"
        context = ["It is summer.", "The weather has been warm recently."]
        final_query = f"{query}\n\nHere is some useful context:\n{context[0]}\n{context[1]}"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "The weather is sunny and warm."
        self.mock_openai_instance.chat.completions.create.return_value = mock_response

        response = self.gpt_generator.generate_response(final_query)

        self.mock_openai_instance.chat.completions.create.assert_called_once_with(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_query},
            ],
            **self.gpt_generate_params,
        )
        self.assertEqual(response, "The weather is sunny and warm.")


if __name__ == "__main__":
    unittest.main()

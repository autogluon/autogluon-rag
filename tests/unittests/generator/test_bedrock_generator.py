import json
import unittest
from unittest.mock import MagicMock, patch

from ragify.modules.generator.generators.bedrock_generator import BedrockGenerator


class TestBedrockGenerator(unittest.TestCase):
    @patch("ragify.modules.generator.generators.bedrock_generator.boto3.client")
    def setUp(self, mock_boto_client):
        self.mock_boto_client = mock_boto_client
        self.model_name = "bedrock-model"
        self.bedrock_generate_params = {"max_length": 100}
        self.bedrock_generator = BedrockGenerator(
            model_name=self.model_name,
            bedrock_generate_params=self.bedrock_generate_params,
            aws_region="us-west-2",
        )
        self.mock_boto_client_instance = self.mock_boto_client.return_value
        self.mock_boto_client_instance.invoke_model = MagicMock()

    def test_generate_response(self):
        query = "What is the weather like today?"
        context = ["It is summer.", "The weather has been warm recently."]
        final_query = f"{query}\n\nHere is some useful context:\n{context[0]}\n{context[1]}"

        mock_response = MagicMock()
        mock_body = json.dumps({"outputs": [{"text": "The weather is sunny and warm."}], "stop_reason": "length"})
        mock_response.get.return_value.read.return_value = mock_body
        self.mock_boto_client_instance.invoke_model.return_value = mock_response

        response = self.bedrock_generator.generate_response(final_query)

        self.mock_boto_client_instance.invoke_model.assert_called_once_with(
            body=json.dumps({"prompt": final_query, **self.bedrock_generate_params}),
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )
        self.assertEqual(response, "The weather is sunny and warm.")


if __name__ == "__main__":
    unittest.main()

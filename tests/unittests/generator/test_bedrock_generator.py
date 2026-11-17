import json
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from agrag.modules.generator.generators.bedrock_generator import BedrockGenerator


class TestBedrockGenerator(unittest.TestCase):
    @patch("agrag.modules.generator.generators.bedrock_generator.boto3.client")
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

    @patch("agrag.modules.generator.generators.bedrock_generator.boto3.client")
    def test_drops_deprecated_param_and_retries(self, mock_boto_client):
        """A ValidationException naming a deprecated param drops it and retries once."""
        generator = BedrockGenerator(
            model_name="anthropic.claude-opus-4-8",
            bedrock_generate_params={"max_tokens": 1024, "temperature": 0},
            aws_region="us-west-2",
        )

        validation_error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "`temperature` is deprecated for this model."}},
            "InvokeModel",
        )
        success = MagicMock()
        success.get.return_value.read.return_value = json.dumps(
            {"type": "message", "content": [{"text": "hi"}]}
        )
        generator.client.invoke_model = MagicMock(side_effect=[validation_error, success])

        response = generator.generate_response("hello")

        self.assertEqual(response, "hi")
        self.assertEqual(generator.client.invoke_model.call_count, 2)
        self.assertIn("temperature", generator._unsupported_params)
        # The retried body must no longer carry temperature, but keeps max_tokens.
        retried_body = json.loads(generator.client.invoke_model.call_args_list[1].kwargs["body"])
        self.assertNotIn("temperature", retried_body)
        self.assertEqual(retried_body["max_tokens"], 1024)

    @patch("agrag.modules.generator.generators.bedrock_generator.boto3.client")
    def test_unrelated_validation_error_reraises(self, mock_boto_client):
        """A ValidationException not naming a known param is not retried."""
        generator = BedrockGenerator(
            model_name="anthropic.claude-opus-4-8",
            bedrock_generate_params={"temperature": 0},
            aws_region="us-west-2",
        )
        err = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "malformed request body"}},
            "InvokeModel",
        )
        generator.client.invoke_model = MagicMock(side_effect=err)

        with self.assertRaises(ClientError):
            generator.generate_response("hello")
        self.assertEqual(generator.client.invoke_model.call_count, 1)


if __name__ == "__main__":
    unittest.main()

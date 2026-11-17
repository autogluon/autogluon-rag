import unittest

from botocore.exceptions import ClientError

from agrag.modules.bedrock_retry import _is_throttling_error


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "InvokeModel")


class TestBedrockRetryPredicate(unittest.TestCase):
    def test_throttling_codes_are_retried(self):
        for code in ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"):
            self.assertTrue(_is_throttling_error(_client_error(code)), code)

    def test_validation_exception_is_not_retried(self):
        self.assertFalse(_is_throttling_error(_client_error("ValidationException")))

    def test_non_clienterror_is_not_retried(self):
        self.assertFalse(_is_throttling_error(ValueError("boom")))


if __name__ == "__main__":
    unittest.main()

import json
import logging
import re
from typing import Dict

import boto3
from botocore.exceptions import ClientError

from agrag.constants import LOGGER_NAME
from agrag.modules.bedrock_retry import bedrock_throttle_retry

logger = logging.getLogger(LOGGER_NAME)

# Inference params some newer models (e.g. Claude Opus 4.8) reject as deprecated /
# unsupported. When Bedrock raises a ValidationException naming one of these, the
# generator drops it and retries. Restricting the parse to known param names keeps
# us from stripping arbitrary tokens out of an unrelated validation message.
_KNOWN_INFERENCE_PARAMS = ("temperature", "top_p", "top_k")


class BedrockGenerator:
    """
    A class used to generate responses based on a query and a given context using AWS Bedrock.
    Refer to https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html for supported models in Bedrock.

    Attributes:
    ----------
    model_name : str
        The name of the Bedrock model to use for response generation.
    bedrock_generate_params : dict, optional
        Additional parameters to pass to the Bedrock generate API method.

    Methods:
    -------
    generate_response(query: str, context: List[str]) -> str:
        Generates a response based on the query and context.
    """

    def __init__(
        self,
        model_name: str,
        aws_region: str = None,
        bedrock_generate_params: Dict = None,
    ):
        self.model_name = model_name
        self.bedrock_generate_params = bedrock_generate_params or {}
        # Inference params the model has told us (via ValidationException) it does
        # not accept; dropped from the request body and remembered for the session.
        self._unsupported_params = set()
        self.client = boto3.client("bedrock-runtime", region_name=aws_region)

        logger.info(f"Using AWS Bedrock Model {self.model_name} for Generator Module")

    def generate_response(self, query: str, temperature: float = None) -> str:
        """
        Generates a response based on the query.

        Parameters:
        ----------
        query : str
            The user query.
        temperature : float, optional
            Per-call sampling temperature override (used by self-consistency
            sampling). ``None`` -> use the configured ``bedrock_generate_params``.
            Silently ignored if the model has already reported it unsupported (e.g.
            Opus 4.8), consistent with the drop-and-retry logic below.

        Returns:
        -------
        str
            The generated response.
        """

        try:
            output = self._invoke_model(self._build_body(query, temperature=temperature))
        except ClientError as err:
            # A model may reject a supplied inference param (e.g. temperature is
            # deprecated for Claude Opus 4.8). Drop the offending param, remember
            # it for subsequent calls, and retry once. Any other ValidationException
            # (or throttling that already exhausted its backoff) reraises.
            dropped = self._handle_unsupported_param(err)
            if not dropped:
                raise
            output = self._invoke_model(self._build_body(query, temperature=temperature))

        output = json.loads(output.get("body").read())
        response = self.extract_response(output)
        return response

    def _build_body(self, query: str, temperature: float = None) -> str:
        """Build the InvokeModel JSON body, omitting params the model has rejected."""
        merged = dict(self.bedrock_generate_params)
        # Per-call temperature override takes precedence over the configured value.
        if temperature is not None:
            merged["temperature"] = temperature
        params = {k: v for k, v in merged.items() if k not in self._unsupported_params}
        if "claude" in self.model_name:
            messages = [{"role": "user", "content": query}]
            return json.dumps({"messages": messages, **params})
        return json.dumps({"prompt": query, **params})

    def _handle_unsupported_param(self, err: ClientError) -> bool:
        """If ``err`` is a ValidationException naming a known inference param, record
        it in ``self._unsupported_params`` so it is dropped on retry. Returns True if
        a param was newly dropped (caller should retry), False otherwise."""
        if err.response.get("Error", {}).get("Code") != "ValidationException":
            return False
        message = err.response.get("Error", {}).get("Message", "") or str(err)
        lowered = message.lower()
        if "deprecated" not in lowered and "not supported" not in lowered and "unsupported" not in lowered:
            return False
        for param in _KNOWN_INFERENCE_PARAMS:
            if re.search(rf"\b{re.escape(param)}\b", lowered) and param not in self._unsupported_params:
                self._unsupported_params.add(param)
                logger.warning(
                    "Bedrock model %s rejected inference param %r (%s); dropping it and retrying.",
                    self.model_name,
                    param,
                    message.strip(),
                )
                return True
        return False

    @bedrock_throttle_retry
    def _invoke_model(self, body: str):
        """Call Bedrock ``invoke_model`` with throttling retry/backoff.

        Scoped to just the API call (not response parsing) so only throttling --
        not a malformed-output ``ValueError`` -- is retried. See
        ``agrag.modules.bedrock_retry`` for the policy.
        """
        return self.client.invoke_model(
            body=body,
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )

    def extract_response(self, output: Dict) -> str:
        """
        Extracts the response text from the model output.

        Parameters:
        ----------
        output : Dict
            The output dictionary from the Bedrock model.

        Returns:
        -------
        str
            The extracted response text.
        """
        # Used for Mistral response
        if "outputs" in output and isinstance(output["outputs"], list) and "text" in output["outputs"][0]:
            return output["outputs"][0]["text"].strip()
        # Used for Anthropic response
        elif "content" in output and output["type"] == "message":
            return output["content"][0]["text"].strip()
        # Used for Llama response
        elif "generation" in output:
            return output["generation"].strip()
        else:
            raise ValueError("Unknown output structure: %s", output)

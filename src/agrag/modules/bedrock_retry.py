"""Shared Bedrock throttling-retry policy for the generator and embedding clients.

Bedrock enforces per-account TPM/RPM limits; under parallel load (e.g. the
1,000-question MuSiQue benchmark running N worker processes) ``invoke_model`` calls
raise ``botocore.exceptions.ClientError`` with a ``ThrottlingException`` code. This
module centralizes the retry/backoff policy so every throttled Bedrock call --
generation and embedding alike -- backs off identically.

The retry is scoped to throttling/transient Bedrock error codes only (see
``_THROTTLE_CODES``) and uses ``reraise=True`` so that after the attempts are
exhausted, or for any other exception -- a permanent ``ValidationException``
(e.g. a deprecated inference param), a malformed response, or a non-Bedrock
error -- the original error propagates immediately, unchanged. This is important:
retrying a permanent error just stalls every call through the full backoff
schedule (~8 min) before failing anyway. Behavior on a successful first call is
completely unaffected.
"""

import logging

from botocore.exceptions import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Transient Bedrock error codes worth backing off on. Everything else (notably
# ``ValidationException``, which is also a ``ClientError``) is permanent and must
# reraise immediately rather than be retried as if it were throttling.
_THROTTLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ProvisionedThroughputExceededException",
    "RequestLimitExceeded",
    "ServiceUnavailableException",
    "ServiceQuotaExceededException",
    "ModelTimeoutException",
}


def _is_throttling_error(exc: BaseException) -> bool:
    """Return True only for transient Bedrock throttle/availability ``ClientError``s."""
    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code") if getattr(exc, "response", None) else None
    return code in _THROTTLE_CODES


def log_retry_attempt(retry_state):
    """tenacity ``after`` callback: log each retry with its attempt number + cause."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Bedrock throttling retry: attempt %d failed after %.1fs (%r); backing off.",
        retry_state.attempt_number,
        retry_state.seconds_since_start or 0.0,
        exc,
    )


# Exponential backoff tuned for tight Bedrock quotas: waits 32s, 64s, then capped at
# 128s, up to 6 attempts total. Only transient throttle/availability codes (see
# _THROTTLE_CODES) are retried; anything else -- including ValidationException --
# reraises immediately so real errors are not masked by long backoff waits.
bedrock_throttle_retry = retry(
    retry=retry_if_exception(_is_throttling_error),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=32, min=32, max=128),
    after=log_retry_attempt,
    reraise=True,
)

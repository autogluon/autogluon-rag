"""Strands-backed reasoning for the agentic RAG path.

This module provides an optional third backend for the ``QueryPlanner`` and
``DecisionPolicy``, alongside the existing rule-based and (raw-Bedrock)
LLM-backed modes. It uses the `Strands Agents <https://github.com/strands-agents>`_
SDK driving Bedrock Claude Sonnet 4.6 to emit only the plan or the action,
while Python keeps deriving every tool/action argument deterministically.

Why this is safe to bolt on:

* The LLM emits only the plan/action, never arguments. Planning returns a
  list of subquery strings; action selection returns a single value constrained
  by a JSON-schema enum to the legal-action set the caller computed. The
  planner's normalization (original query first, dedup, cap) and the policy's
  ``_build_args`` still run in Python, so a model response can never inject a
  tool argument or an illegal action.
* Structured output, not free text. Strands' ``Agent.structured_output``
  forces a Bedrock tool-use response validated against a Pydantic schema, so we
  get a typed object back instead of parsing prose. An Enum field becomes a
  JSON-schema ``enum``, which is exactly how "choose one of these actions" is
  expressed to the model.
* Never breaks the loop. ``strands`` is imported lazily and every call is
  wrapped so any import error, credential/backend failure, or validation error
  returns ``None``; the caller then falls back to its LLM or rule-based path.

The SDK and its Pydantic dependency are optional; nothing here is imported at
module load unless a backend is actually constructed.
"""

import logging
from enum import Enum
from typing import List, Optional, Sequence

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Conservative default cap on generated tokens; plan/action outputs are tiny.
_DEFAULT_MAX_TOKENS = 512

# Instruction anchoring the planner's structured output. The model returns only
# subqueries; Python normalizes them back into the executor's list contract.
_PLAN_SYSTEM_PROMPT = (
    "You are a retrieval query planner for a RAG system. Decompose the user's "
    "question into focused search subqueries (subgoals) that, retrieved "
    "together, would surface the evidence needed to answer it. Do NOT answer the "
    "question. Prefer a single subquery for simple questions; only split when the "
    "question bundles distinct information needs (e.g. multi-hop or comparison). "
    "When a later subquery needs the ANSWER to an earlier one (a bridge entity the "
    "question never names -- e.g. the band a person formed, then that band's home "
    "city), order the earlier subquery first and refer to its answer with a 1-based "
    "back-reference '#n' (n = the earlier subquery's position). Example: for 'In what "
    "city was the band Danko Jones formed?' -> ['What band did Danko Jones form?', "
    "'In what city was #1 formed?']. Use '#n' ONLY for a genuine dependency."
)


class StrandsReasoner:
    """Bedrock/Strands reasoning helper shared by the planner and policy.

    A single instance holds one Strands ``Agent`` bound to a ``BedrockModel``
    (Claude Sonnet 4.6 by default) and exposes two narrow methods:

    * :meth:`plan_subqueries` — returns a list of subquery strings (or ``None``).
    * :meth:`choose_action` — returns one action value from a supplied legal set
      (or ``None``), enforced by a per-call Enum-constrained schema so the model
      cannot return anything outside the set.

    Neither method returns tool arguments; the callers derive those
    deterministically. Any failure (missing SDK, Bedrock error, invalid output)
    yields ``None`` so the caller can fall back.

    Parameters
    ----------
    model_id : str
        Bedrock model id / inference-profile id (e.g.
        ``us.anthropic.claude-sonnet-4-6``).
    region_name : Optional[str]
        AWS region for the Bedrock runtime client. When ``None`` the SDK/boto3
        default resolution applies.
    temperature : float
        Sampling temperature; ``0.0`` for deterministic-as-possible planning and
        action selection.
    max_tokens : int
        Upper bound on generated tokens (outputs are small).
    """

    def __init__(
        self,
        model_id: str,
        region_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._agent = None  # lazily built on first use
        self._unavailable = False  # sticky flag once construction is known to fail

    def _get_agent(self):
        """Build (once) and return the Strands agent, or ``None`` if unavailable.

        The import and model construction are deferred to first use and cached.
        A failure is remembered so we do not retry a broken import on every call.
        """
        if self._agent is not None:
            return self._agent
        if self._unavailable:
            return None
        try:
            from strands import Agent
            from strands.models import BedrockModel

            model = BedrockModel(
                model_id=self.model_id,
                region_name=self.region_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # strict_tools makes the model adhere to the tool/JSON schema,
                # which is what enforces "emit only the enum action".
                strict_tools=True,
            )
            self._agent = Agent(model=model)
            return self._agent
        except Exception as exc:  # SDK missing / bad config / no creds
            logger.warning("Strands backend unavailable (%s); callers will fall back", exc)
            self._unavailable = True
            return None

    @property
    def available(self) -> bool:
        """Whether the Strands agent can be constructed (best-effort, cached)."""
        return self._get_agent() is not None

    def plan_subqueries(self, query: str, max_subqueries: int) -> Optional[List[str]]:
        """Decompose ``query`` into up to ``max_subqueries`` retrieval subqueries.

        Returns the raw list of subquery strings from the model (unnormalized),
        or ``None`` on any failure. The caller is responsible for prepending the
        original query, deduping, and capping.
        """
        agent = self._get_agent()
        if agent is None:
            return None

        # Local Pydantic schema so import stays optional and the field docstring
        # steers the model. Defined per call to keep module import dependency-free.
        try:
            from pydantic import BaseModel, Field
        except Exception as exc:  # pydantic ships with strands, but be defensive
            logger.debug("pydantic unavailable for Strands planner (%s)", exc)
            return None

        class _Plan(BaseModel):
            subqueries: List[str] = Field(
                default_factory=list,
                description=(
                    f"Between 1 and {max_subqueries} focused retrieval subqueries. "
                    "Each is a search query, not an answer."
                ),
            )

        prompt = f"{_PLAN_SYSTEM_PROMPT}\n\nQuestion: {query}"
        try:
            result = agent.structured_output(_Plan, prompt)
        except Exception as exc:
            logger.debug("Strands planner call failed (%s); falling back", exc)
            return None

        subqueries = getattr(result, "subqueries", None)
        if not isinstance(subqueries, list):
            return None
        # Return only string items; normalization/capping happens in the planner.
        return [s for s in subqueries if isinstance(s, str)]

    def choose_action(self, prompt: str, legal_values: Sequence[str]) -> Optional[str]:
        """Choose one action value from ``legal_values`` via an Enum-constrained schema.

        ``prompt`` is the caller's state summary + allowed-action description.
        The returned value is guaranteed to be one of ``legal_values`` (the
        JSON-schema enum enforces this); returns ``None`` on any failure so the
        caller falls back to its deterministic first-legal choice.
        """
        agent = self._get_agent()
        if agent is None:
            return None
        if not legal_values:
            return None

        try:
            from pydantic import BaseModel, Field
        except Exception as exc:
            logger.debug("pydantic unavailable for Strands policy (%s)", exc)
            return None

        # Build a dynamic str-Enum restricted to exactly the legal actions, so the
        # model's only valid outputs are those values (JSON-schema enum).
        action_enum = Enum("LegalAction", {v: v for v in legal_values}, type=str)

        class _Choice(BaseModel):
            action: action_enum = Field(  # type: ignore[valid-type]
                description="The single best next action; must be one of the allowed values.",
            )

        try:
            result = agent.structured_output(_Choice, prompt)
        except Exception as exc:
            logger.debug("Strands policy call failed (%s); falling back", exc)
            return None

        chosen = getattr(result, "action", None)
        if chosen is None:
            return None
        # ``chosen`` is a str-Enum member; normalize to its plain string value.
        value = getattr(chosen, "value", chosen)
        if not isinstance(value, str) or value not in legal_values:
            return None
        return value

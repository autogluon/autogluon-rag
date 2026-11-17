"""Query planning for the agentic RAG path.

The ``QueryPlanner`` prepares retrieval queries: it does not answer the user's
question directly, it only produces one or more retrieval queries (subqueries /
subgoals) from the input query.

Three modes are supported, in precedence order Strands > LLM > rule-based:

* Rule-based (default): a simple regex that splits the query on conjunctions
  and punctuation. Deterministic and dependency-free.
* LLM-backed (opt-in via ``use_llm`` + a ``generator_module``): the query is
  decomposed by the configured generator (the same shared LLM used elsewhere in
  the agentic path). The model's output is parsed and validated back into the
  same ``List[str]`` contract — original query first, deduped, with at most
  ``max_subqueries`` subqueries appended after it — and any malformed output falls
  back to the rule-based plan.
  There is no pydantic dependency; validation is stdlib JSON + type checks, in
  the same "constrained output + tolerant parse + safe fallback" spirit as
  ``AnswerVerifier``.
* Strands-backed (opt-in via a ``strands_backend``): the query is decomposed
  by a Strands agent driving Bedrock Sonnet 4.6, which returns only subquery
  strings (Pydantic-validated structured output). Those strings go through the
  same normalization as the LLM path, so the ``List[str]`` contract is identical.
  Any failure falls back to the LLM path (if configured) and then to rules.

In every mode the LLM emits only the subqueries; Python owns the normalization
(original query first, dedup, cap), so the executor's contract can't be broken.
"""

import json
import logging
import re
from typing import List, Optional

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Conjunctions / cues that often separate distinct information needs in a query.
_SPLIT_PATTERN = re.compile(
    r"\b(?:and|versus|vs\.?|compared to|as well as)\b|[;?]",
    flags=re.IGNORECASE,
)

# Prompt for the LLM planner. It must decompose (not answer) and emit strict JSON
# so the output can be validated without a schema library.
_PLAN_INSTRUCTION = (
    "You are a retrieval query planner for a RAG system. Decompose the user's "
    "question into focused search subqueries (subgoals) that, retrieved "
    "together, would surface the evidence needed to answer it. Do NOT answer the "
    "question. Prefer a single subquery for simple questions; only split when the "
    "question bundles distinct information needs (e.g. multi-hop or comparison). "
    "Preserve every named entity, place, date, and constraint from the question in "
    "the subqueries. When a single step depends on two or more constraints joined by "
    "'and' (e.g. 'north of X and where Y happened'), keep ALL of them together in one "
    "subquery -- never drop or split off one of the anchors. "
    "When a later subquery needs the ANSWER to an earlier one (a bridge entity the "
    "question never names -- e.g. the band a person formed, then that band's home "
    "city), order the subqueries so the earlier one comes first and refer to its "
    "answer with a 1-based back-reference '#n' (n = the earlier subquery's position). "
    'Example: for "In what city was the band Danko Jones formed?" -> '
    '["What band did Danko Jones form?", "In what city was #1 formed?"]. '
    "A single subquery may depend on MORE THAN ONE earlier subquery -- reference each "
    "with its own '#n'. Example: for \"Who was president when the director of Film X "
    'and the director of Film Y were both alive?" -> ["Who directed Film X?", "Who '
    'directed Film Y?", "During what period were #1 and #2 both alive?", "Who was '
    'president during #3?"]. '
    "Use '#n' ONLY for a genuine dependency; do not invent one for independent steps. "
    "Return at most {max_subqueries} subqueries.\n\n"
    'Return ONLY a JSON object of the form {{"subqueries": ["...", "..."]}} with '
    "no markdown, no code fences, and no explanation.\n\n"
    "Question: "
)

# Extra directive prepended to the planner prompt on a recovery re-plan. The
# first decomposition produced an answer the verifier rejected, so we ask for a
# genuinely different, finer-grained breakdown rather than a paraphrase of the
# same (failed) plan. The failed plan and a short diagnosis are interpolated in so
# the model can steer away from the specific mistake (e.g. a dropped constraint or
# a wrong bridge entity).
_REPLAN_DIRECTIVE = (
    "This is a RETRY. A previous decomposition of this question led to an answer "
    "that FAILED verification, so it was wrong or unsupported. Produce a DIFFERENT, "
    "MORE GRANULAR decomposition -- do not repeat the previous plan. Split each "
    "bundled constraint into its own hop, make every bridge dependency explicit with "
    "a '#n' back-reference (a hop that depends on two or more earlier hops must "
    "reference EACH of them, e.g. '#2' and '#3'), and keep EVERY named entity, place, "
    "date, and constraint from the question. Be careful not to drop a constraint or "
    "resolve a bridge entity to the wrong (e.g. more famous) referent.\n"
    "Previous (failed) plan: {previous_plan}\n"
    "{feedback}"
    "\n"
)


class QueryPlanner:
    """Planner that derives retrieval subqueries from a query.

    Attributes:
    ----------
    max_subqueries : int
        Upper bound on the number of subqueries produced beyond the original
        query. The original query is always the first plan entry and does not count
        against this bound, so a plan holds up to ``max_subqueries + 1`` entries.
    generator_module : Optional[GeneratorModule]
        The shared generator used when ``use_llm`` is enabled. When ``None`` the
        planner is always rule-based regardless of ``use_llm``.
    use_llm : bool
        Whether to use the LLM to decompose the query. Effective only when a
        ``generator_module`` is also provided.
    strands_backend : Optional[StrandsReasoner]
        A Strands/Bedrock reasoning helper. When provided it takes precedence
        over the raw-LLM path: the model emits only subquery strings, which are
        normalized the same way. ``None`` disables the Strands path.
    """

    def __init__(self, max_subqueries: int = 4, generator_module=None, use_llm: bool = False, strands_backend=None):
        self.max_subqueries = max_subqueries
        self.generator_module = generator_module
        self.use_llm = use_llm
        self.strands_backend = strands_backend

    @property
    def _llm_enabled(self) -> bool:
        return self.use_llm and self.generator_module is not None

    def create_plan(self, query: str) -> List[str]:
        """Return a list of retrieval queries for the given user query.

        The original query is always the first entry. Precedence is
        Strands > raw-LLM > rule-based: each configured mode is tried in turn and
        any failure falls through to the next, so the loop never breaks.
        """
        if self.strands_backend is not None:
            strands_plan = self._strands_plan(query)
            if strands_plan:
                logger.debug("Strands planner produced %d subqueries for %r", len(strands_plan), query)
                return strands_plan
            logger.debug("Strands planner produced no usable plan for %r; falling back", query)
        if self._llm_enabled:
            llm_plan = self._llm_plan(query)
            if llm_plan:
                logger.debug("LLM planner produced %d subqueries for %r", len(llm_plan), query)
                return llm_plan
            logger.debug("LLM planner produced no usable plan for %r; falling back to rules", query)
        return self._rule_based_plan(query)

    def replan(
        self,
        query: str,
        previous_plan: Optional[List[str]] = None,
        feedback: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Produce an alternative, finer-grained decomposition for verify-recovery.

        Called only after a first plan's answer failed verification. It reuses the
        LLM / Strands decomposition machinery but (1) appends ``_REPLAN_DIRECTIVE``
        so the model breaks the question down differently and keeps every
        constraint, and (2) raises the effective subquery cap (``max_subqueries + 2``)
        so a genuine 4-hop chain has headroom the first pass' cap does not leave.
        ``previous_plan`` is the plan to avoid; ``feedback`` is an optional short
        diagnosis (e.g. the rejected answer / verification label). Returns a fresh
        ``List[str]`` (original query first) or ``None`` when no usable alternative
        plan could be produced, so the caller keeps its current plan.
        """
        cap = self.max_subqueries + 2
        prev = ", ".join(previous_plan[1:]) if previous_plan and len(previous_plan) > 1 else "(none)"
        directive = _REPLAN_DIRECTIVE.format(
            previous_plan=prev,
            feedback=(f"Diagnosis: {feedback}\n" if feedback else ""),
        )
        if self.strands_backend is not None:
            # The Strands backend takes only (query, max); fold the directive into
            # the query text so the retry still steers away from the failed plan.
            plan = self._strands_plan(f"{directive}\nQuestion: {query}", query, max_subqueries=cap)
            if plan:
                return plan
        if self._llm_enabled:
            prompt = _PLAN_INSTRUCTION.format(max_subqueries=cap) + directive + query
            try:
                raw = self.generator_module.generate_response(prompt)
            except Exception as exc:  # generation must never break recovery
                logger.debug("Replan generation failed (%s)", exc)
                return None
            return self._parse_subqueries(raw, query, max_subqueries=cap)
        return None

    def _strands_plan(
        self, query: str, original_query: Optional[str] = None, max_subqueries: Optional[int] = None
    ) -> Optional[List[str]]:
        """Decompose the query via the Strands backend; normalize its output.

        The backend returns only subquery strings; normalization (original query
        first, dedup, cap) is applied here so the ``List[str]`` contract matches
        the other modes. ``original_query`` is the string used as the plan's first
        entry (defaults to ``query``); it lets the replan path pass a
        directive-augmented ``query`` to the backend while still leading the plan
        with the clean user question. Returns ``None`` on any failure so the caller
        falls back.
        """
        cap = max_subqueries if max_subqueries is not None else self.max_subqueries
        lead = original_query if original_query is not None else query
        try:
            subqueries = self.strands_backend.plan_subqueries(query, cap)
        except Exception as exc:  # backend must never break the loop
            logger.debug("Strands planner call failed (%s); falling back", exc)
            return None
        if not subqueries:
            return None
        return self._normalize_subqueries(subqueries, lead, max_subqueries=cap)

    def _rule_based_plan(self, query: str) -> List[str]:
        """Derive subqueries by splitting on conjunctions / punctuation.

        The original query is always included first and does not count against the
        cap. If the query appears to bundle multiple information needs (e.g.
        contains "and", "versus", "?"), the parts are added as additional
        subqueries, up to ``max_subqueries`` subqueries beyond the original.
        """
        query = query.strip()
        plan: List[str] = [query] if query else []

        parts = [p.strip() for p in _SPLIT_PATTERN.split(query) if p and p.strip()]
        # Only treat as multi-part when splitting actually produced >1 meaningful
        # part and each part is a reasonable length (avoids splitting on stray
        # punctuation into tiny fragments).
        meaningful = [p for p in parts if len(p.split()) >= 2]
        if len(meaningful) > 1:
            for part in meaningful:
                # Cap the number of subqueries (entries beyond the original),
                # not the whole plan, so the original never consumes a slot.
                if len(plan) - 1 >= self.max_subqueries:
                    break
                if part not in plan:
                    plan.append(part)

        logger.debug("Rule-based planner produced %d subqueries for %r", len(plan) - 1, query)
        return plan

    def _llm_plan(self, query: str) -> Optional[List[str]]:
        """Ask the generator to decompose the query; validate its output.

        Returns a validated ``List[str]`` (original query first) or ``None`` when
        the model output cannot be parsed/validated, so the caller falls back.
        """
        prompt = _PLAN_INSTRUCTION.format(max_subqueries=self.max_subqueries) + query
        try:
            raw = self.generator_module.generate_response(prompt)
        except Exception as exc:  # generator/backend failure must not break the loop
            logger.debug("LLM planner generation failed (%s); falling back to rules", exc)
            return None
        return self._parse_subqueries(raw, query)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Best-effort parse of a JSON object embedded in model output.

        Slices from the first ``{`` to the last ``}`` so leading/trailing prose or
        code fences do not defeat parsing. Returns the parsed dict, or ``None`` if
        no JSON object can be recovered.
        """
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None

    def _parse_subqueries(
        self, raw: str, original_query: str, max_subqueries: Optional[int] = None
    ) -> Optional[List[str]]:
        """Validate the model output into a capped, deduped ``List[str]``.

        Delegates normalization to ``_normalize_subqueries``. ``max_subqueries``
        overrides the default cap (used by the replan path). Returns ``None`` on
        any parse/validation failure so the caller falls back.
        """
        obj = self._extract_json(raw)
        if obj is None:
            logger.debug("LLM planner output was not valid JSON: %r", raw)
            return None
        subqueries = obj.get("subqueries")
        if not isinstance(subqueries, list):
            logger.debug("LLM planner JSON missing a 'subqueries' list: %r", obj)
            return None
        return self._normalize_subqueries(subqueries, original_query, max_subqueries=max_subqueries)

    def _normalize_subqueries(
        self, subqueries: List, original_query: str, max_subqueries: Optional[int] = None
    ) -> Optional[List[str]]:
        """Normalize raw model subqueries into the executor's ``List[str]`` contract.

        The plan always leads with the original query, which does not count against
        the cap; only non-empty string subqueries not already present are appended,
        capped at ``max_subqueries`` subqueries (so the plan holds up to
        ``max_subqueries + 1`` entries). This matches the prompts, which tell the
        model to return "at most ``max_subqueries`` subqueries". Shared by the LLM
        and Strands paths so both yield identical shapes. Returns ``None`` when
        nothing usable remains so the caller falls back.
        """
        cap = max_subqueries if max_subqueries is not None else self.max_subqueries
        original = original_query.strip()
        plan: List[str] = [original] if original else []
        for item in subqueries:
            # Cap the number of subqueries (entries beyond the original).
            if len(plan) - (1 if original else 0) >= cap:
                break
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned and cleaned not in plan:
                plan.append(cleaned)

        return plan or None

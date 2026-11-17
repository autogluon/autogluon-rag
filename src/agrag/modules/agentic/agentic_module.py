"""Top-level controller for the agentic RAG path.

``AgenticRAGModule`` wires together the planner, tools, policy, synthesizer, and
verifier, runs the bounded loop via ``AgentExecutor``, and returns either a
supported answer or an abstention message (optionally with a structured trace).

It reuses the already-initialized ``RetrieverModule`` and ``GeneratorModule`` from
the RAG pipeline; it never re-ingests, re-chunks, or re-embeds data.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.executor import AgentExecutor
from agrag.modules.agentic.planner import QueryPlanner
from agrag.modules.agentic.policy import DecisionPolicy
from agrag.modules.agentic.synthesizer import AnswerSynthesizer
from agrag.modules.agentic.tools.llm_tools import ContextCompressionTool, QueryRewriteTool
from agrag.modules.agentic.tools.registry import ToolRegistry
from agrag.modules.agentic.tools.retrieve_tools import MultiQueryRetrieveTool, RetrieveTool
from agrag.modules.agentic.trace import AgentTrace
from agrag.modules.agentic.verifier import AnswerVerifier

logger = logging.getLogger(LOGGER_NAME)

DEFAULT_ABSTENTION = (
    "I don't have enough supporting evidence in the indexed documents to answer " "this question confidently."
)


class AgenticRAGModule:
    """Controller for agentic answering.

    Attributes:
    ----------
    retriever_module : RetrieverModule
        Existing retriever, wrapped by the retrieval tools.
    generator_module : GeneratorModule
        Existing generator, used for synthesis, verification, and (optionally)
        query rewrite / context compression.
    config : dict
        Agent configuration (see configs/agent/default.yaml). Recognized keys:
        max_iterations, max_subqueries, retrieve_top_k_per_query,
        use_query_rewrite, use_context_compression, use_verification,
        allow_abstention, max_rewrites, min_evidence_count, max_context_tokens,
        min_subgoal_coverage, min_relevance, query_prefix, use_llm_planner,
        use_llm_policy, use_strands_planner, use_strands_policy,
        use_iterative_planner.
    """

    def __init__(
        self,
        retriever_module,
        generator_module,
        config: Optional[Dict[str, Any]] = None,
        verifier_generator_module=None,
    ):
        self.retriever_module = retriever_module
        self.generator_module = generator_module
        # Optional independent verifier model (see agent_verifier_model). None ->
        # the verifier reuses ``generator_module`` (prior behavior).
        self.verifier_generator_module = verifier_generator_module
        cfg = config or {}

        self.max_iterations = cfg.get("max_iterations", 5)
        self.max_subqueries = cfg.get("max_subqueries", 4)
        # None here is a deliberate "no agent-level override" sentinel: the retrieve
        # tools pass top_k=None straight through so the retriever uses its own
        # configured top_k. This differs from AGENT_RETRIEVE_TOP_K_PER_QUERY (8) in
        # configs/agent/default.yaml, which the normal args path always supplies; the
        # 8 is the pipeline default, while None is the module-level "unset" behavior.
        self.retrieve_top_k_per_query = cfg.get("retrieve_top_k_per_query", None)
        self.use_query_rewrite = cfg.get("use_query_rewrite", True)
        self.use_context_compression = cfg.get("use_context_compression", False)
        self.use_verification = cfg.get("use_verification", True)
        # Never-refuse is the default: the agent iterates (rewrite + retrieve more)
        # on weak/failed verification and, when iteration is exhausted, returns a
        # best-effort evidence-grounded answer instead of the canned abstention.
        # Verification still runs and steers those rewrites. Set allow_abstention
        # True to restore the ability to refuse (abstain) when a question is
        # unanswerable from the corpus.
        self.allow_abstention = cfg.get("allow_abstention", False)
        # How many query rewrites a single run may issue. This is the main lever on
        # how hard the agent iterates for new evidence (re-retrieving the same query
        # adds nothing), so the never-refuse default benefits from more than one.
        self.max_rewrites = cfg.get("max_rewrites", 2)
        self.min_evidence_count = cfg.get("min_evidence_count", 2)
        self.max_context_tokens = cfg.get("max_context_tokens", 6000)
        self.min_subgoal_coverage = cfg.get("min_subgoal_coverage", 0.5)
        self.min_relevance = cfg.get("min_relevance", None)
        self.query_prefix = cfg.get("query_prefix", "")
        # Distill verbose/reasoning synthesis drafts into a short answer span (one
        # extra generator call, only when the draft looks verbose). Default off ->
        # raw draft returned unchanged.
        self.extract_answer = cfg.get("extract_answer", False)
        # On failed verification (sequential path), do bounded rewrite+retrieve+
        # re-draft passes so a discriminative verifier can correct a wrong answer.
        self.verify_retry = cfg.get("verify_retry", False)
        self.use_fused_retrieval = cfg.get("use_fused_retrieval", False)
        self.rrf_k = cfg.get("rrf_k", 60)
        self.use_llm_planner = cfg.get("use_llm_planner", False)
        self.use_llm_policy = cfg.get("use_llm_policy", False)
        self.use_strands_planner = cfg.get("use_strands_planner", False)
        self.use_strands_policy = cfg.get("use_strands_policy", False)
        # Sequential-hop execution: resolve subqueries in order, threading each
        # hop's answer into the next hop's retrieval query. Default off -> the
        # existing single up-front parallel MULTI_RETRIEVE behavior is unchanged.
        self.use_iterative_planner = cfg.get("use_iterative_planner", False)
        # Multi-hop recovery (sequential path). use_hop_recovery: on an UNKNOWN hop,
        # hypothesize a candidate and use it only as an extra search query, then
        # re-extract. use_replan_recovery: on a failed final verification, re-decompose
        # the whole question and re-run the hops. Both default off.
        self.use_hop_recovery = cfg.get("use_hop_recovery", False)
        self.use_replan_recovery = cfg.get("use_replan_recovery", False)
        self.max_recovery_attempts = cfg.get("max_recovery_attempts", 2)
        # Global cap on total hop retrievals across all passes (initial plan +
        # replans) for one sequential run. None -> unbounded (prior behavior).
        self.max_total_hops = cfg.get("max_total_hops", None)
        # Deep-hop retrieval widening (sequential path): a hop that depends on an
        # earlier hop, or whose index >= deep_hop_threshold, requests deep_hop_top_k
        # chunks instead of the base per-query top_k. None -> disabled.
        self.deep_hop_top_k = cfg.get("deep_hop_top_k", None)
        self.deep_hop_threshold = cfg.get("deep_hop_threshold", 2)
        # Reject a hop answer that shares no content token with that hop's evidence
        # (a parametric leak), routing it through recovery/UNKNOWN. None/False off.
        self.use_entity_grounding = cfg.get("use_entity_grounding", False)
        # Drop the single-word query_prefix on the final multi-hop synthesis so
        # answer-bearing qualifiers survive. None/False -> prefix applied.
        self.final_answer_drop_prefix = cfg.get("final_answer_drop_prefix", False)
        self.use_direct_answer_fallback = cfg.get("use_direct_answer_fallback", False)
        # Dual-candidate synthesis + arbitration: always compute a chain-free
        # candidate alongside the chain draft and reconcile them (supersedes the
        # abstention-only direct-answer fallback). Default off.
        self.dual_synthesis = cfg.get("dual_synthesis", False)
        # Self-consistency: draft the final answer K times at a sampling temperature
        # and keep the majority answer. 1 -> disabled (single greedy draft).
        self.self_consistency = cfg.get("self_consistency", 1)
        # Real cost budgets for one run: wall-clock seconds, total generator (LLM)
        # calls, and total retrieval-tool calls. Each None -> unbounded. When any
        # trips, the run returns its best-effort answer tagged ANSWERED_UNVERIFIED
        # with reason ``budget_exhausted`` (never-refuse contract preserved).
        self.max_wall_clock_s = cfg.get("max_wall_clock_s", None)
        self.max_llm_calls = cfg.get("max_llm_calls", None)
        self.max_retrieval_calls = cfg.get("max_retrieval_calls", None)

        self._build_components()

    def _build_strands_backend(self):
        """Build a shared Strands reasoning backend, or return ``None``.

        A single ``StrandsReasoner`` is shared by the planner and policy when
        either Strands flag is on. It reuses the generator's Bedrock model id and
        region so no new model configuration is introduced. Construction is
        deferred/guarded inside the reasoner, so a missing SDK or bad credentials
        degrade gracefully to the existing LLM/rule-based paths rather than
        raising here.
        """
        if not (self.use_strands_planner or self.use_strands_policy):
            return None

        model_id = getattr(self.generator_module, "model_name", None)
        # Best-effort region discovery from the underlying Bedrock client; falls
        # back to boto3/SDK default resolution when unavailable.
        region_name = None
        generator = getattr(self.generator_module, "generator", None)
        client = getattr(generator, "client", None)
        if client is not None:
            region_name = getattr(getattr(client, "meta", None), "region_name", None)

        if not model_id:
            logger.warning("Strands backend requested but generator has no model_name; disabling it")
            return None

        try:
            from agrag.modules.agentic.strands_backend import StrandsReasoner

            return StrandsReasoner(model_id=model_id, region_name=region_name)
        except Exception as exc:  # import-time failure must not break construction
            logger.warning("Could not create Strands backend (%s); using LLM/rule paths", exc)
            return None


    def _build_components(self) -> None:
        # Shared Strands backend (None unless a Strands flag is on and it builds).
        strands_backend = self._build_strands_backend()

        tools = [
            RetrieveTool(self.retriever_module, top_k=self.retrieve_top_k_per_query),
            MultiQueryRetrieveTool(
                self.retriever_module,
                top_k=self.retrieve_top_k_per_query,
                use_fused_retrieval=self.use_fused_retrieval,
                rrf_k=self.rrf_k,
            ),
        ]
        if self.use_query_rewrite:
            tools.append(QueryRewriteTool(self.generator_module))
        if self.use_context_compression:
            tools.append(ContextCompressionTool(self.generator_module))
        self.tool_registry = ToolRegistry(tools)

        self.planner = QueryPlanner(
            max_subqueries=self.max_subqueries,
            generator_module=self.generator_module,
            use_llm=self.use_llm_planner,
            strands_backend=strands_backend if self.use_strands_planner else None,
        )
        self.synthesizer = AnswerSynthesizer(
            self.generator_module,
            max_context_tokens=self.max_context_tokens,
            query_prefix=self.query_prefix,
            extract_answer=self.extract_answer,
        )
        self.verifier = (
            AnswerVerifier(
                # Judge with the independent verifier model when configured, else
                # reuse the drafting generator (prior behavior).
                self.verifier_generator_module or self.generator_module,
                # The verifier only needs >=1 evidence item to render a judgment;
                # gating it on the policy's ``min_evidence_count`` (the "retrieve
                # more" threshold, default 2) auto-rejected correct answers grounded
                # in a single strong passage without ever calling the judge. Decouple
                # them: the policy keeps the config value, the verifier requires only
                # non-empty evidence.
                min_evidence_count=1,
                max_context_tokens=self.max_context_tokens,
            )
            if self.use_verification
            else None
        )
        self.policy = DecisionPolicy(
            min_evidence_count=self.min_evidence_count,
            use_query_rewrite=self.use_query_rewrite,
            use_context_compression=self.use_context_compression,
            max_context_tokens=self.max_context_tokens,
            min_subgoal_coverage=self.min_subgoal_coverage,
            min_relevance=self.min_relevance,
            max_rewrites=self.max_rewrites,
            max_iterations=self.max_iterations,
            generator_module=self.generator_module,
            use_llm=self.use_llm_policy,
            strands_backend=strands_backend if self.use_strands_policy else None,
            allow_abstention=self.allow_abstention,
        )
        self.executor = AgentExecutor(
            tool_registry=self.tool_registry,
            policy=self.policy,
            planner=self.planner,
            synthesizer=self.synthesizer,
            verifier=self.verifier,
            max_iterations=self.max_iterations,
            allow_abstention=self.allow_abstention,
            use_iterative_planner=self.use_iterative_planner,
            verify_retry=self.verify_retry,
            use_hop_recovery=self.use_hop_recovery,
            use_replan_recovery=self.use_replan_recovery,
            max_recovery_attempts=self.max_recovery_attempts,
            max_total_hops=self.max_total_hops,
            deep_hop_top_k=self.deep_hop_top_k,
            deep_hop_threshold=self.deep_hop_threshold,
            use_entity_grounding=self.use_entity_grounding,
            final_answer_drop_prefix=self.final_answer_drop_prefix,
            use_direct_answer_fallback=self.use_direct_answer_fallback,
            dual_synthesis=self.dual_synthesis,
            self_consistency=self.self_consistency,
            max_wall_clock_s=self.max_wall_clock_s,
            max_llm_calls=self.max_llm_calls,
            max_retrieval_calls=self.max_retrieval_calls,
        )

    def answer(self, query: str, return_trace: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Answer a query via the agentic loop.

        Returns the answer string (or an abstention message). When
        ``return_trace`` is True, returns ``(answer, trace_dict)``.
        """
        state, evidence_store, final_answer = self.executor.run(query)

        answer = final_answer if final_answer is not None else DEFAULT_ABSTENTION
        logger.info("Agentic run finished with status=%s", state.status.value)

        if return_trace:
            trace = AgentTrace.from_run(state, evidence_store, final_answer)
            return answer, trace.to_dict()
        return answer

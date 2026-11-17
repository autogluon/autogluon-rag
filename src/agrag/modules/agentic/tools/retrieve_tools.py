"""Retrieval tools that wrap the existing RetrieverModule.

These tools do not re-ingest, re-chunk, or re-embed anything. They call the
existing ``RetrieverModule.retrieve(..., return_metadata=True)`` at query time and
convert the structured records into ``Evidence`` objects.
"""

import logging
from typing import List

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import Evidence
from agrag.modules.agentic.tools.base import Tool, ToolResult

logger = logging.getLogger(LOGGER_NAME)


def _records_to_evidence(records, retrieval_query: str, tool_name: str) -> List[Evidence]:
    """Convert retriever records into Evidence objects.

    ``records`` may be ``None`` (no valid indices) or a list of dicts. When the
    retriever is configured to return text only (a list of strings), each string
    is wrapped as text-only evidence so the tool still degrades gracefully.
    """
    if not records:
        return []
    evidence = []
    for rank, record in enumerate(records):
        if isinstance(record, str):
            evidence.append(Evidence(text=record, retrieval_query=retrieval_query, rank=rank, tool_name=tool_name))
        else:
            item = Evidence.from_retrieval_record(
                record, retrieval_query=retrieval_query, rank=record.get("rank", rank), tool_name=tool_name
            )
            evidence.append(item)
    return evidence


class RetrieveTool(Tool):
    """Run a single retrieval query against the existing retriever.

    Attributes:
    ----------
    top_k : Optional[int]
        Per-query number of chunks to retrieve. If None, the retriever's own
        ``top_k`` is used.
    """

    name = "RetrieveTool"

    def __init__(self, retriever_module, top_k: int = None):
        self.retriever_module = retriever_module
        self.top_k = top_k

    def run(self, query: str, top_k: int = None, **kwargs) -> ToolResult:
        # A per-call ``top_k`` overrides the tool's default so callers can widen
        # retrieval for specific queries (e.g. the executor pulls a larger window
        # for deep, dependency-bearing hops). ``RetrieverModule.retrieve`` already
        # treats ``top_k=None`` as "use my configured top_k", so this is a pure
        # pass-through when the caller does not override.
        effective_top_k = top_k if top_k is not None else self.top_k
        records = self.retriever_module.retrieve(query, return_metadata=True, top_k=effective_top_k)
        evidence = _records_to_evidence(records, retrieval_query=query, tool_name=self.name)
        logger.debug("%s retrieved %d chunks for %r", self.name, len(evidence), query)
        return self._result(
            evidence=evidence,
            summary=f"retrieved {len(evidence)} chunks for query",
        )


class MultiQueryRetrieveTool(Tool):
    """Run retrieval over several subqueries and merge the results.

    Deduplication across subqueries is left to the ``EvidenceStore``; this tool
    simply returns all evidence it produced, tagged with the subquery that found
    each item.

    Attributes:
    ----------
    top_k : Optional[int]
        Per-subquery number of chunks to retrieve. If None, the retriever's own
        ``top_k`` is used.
    """

    name = "MultiQueryRetrieveTool"

    def __init__(self, retriever_module, top_k: int = None, use_fused_retrieval: bool = False, rrf_k: int = 60):
        self.retriever_module = retriever_module
        self.top_k = top_k
        self.use_fused_retrieval = use_fused_retrieval
        self.rrf_k = rrf_k

    def run(self, queries: List[str], original_query: str = None, **kwargs) -> ToolResult:
        if self.use_fused_retrieval:
            return self._run_fused(queries, original_query)

        all_evidence: List[Evidence] = []
        for query in queries:
            records = self.retriever_module.retrieve(query, return_metadata=True, top_k=self.top_k)
            all_evidence.extend(_records_to_evidence(records, retrieval_query=query, tool_name=self.name))
        logger.debug("%s retrieved %d chunks across %d subqueries", self.name, len(all_evidence), len(queries))
        return self._result(
            evidence=all_evidence,
            summary=f"retrieved {len(all_evidence)} chunks across {len(queries)} subqueries",
        )

    def _run_fused(self, queries: List[str], original_query: str = None) -> ToolResult:
        """Globally fuse retrieval across all subqueries via the retriever.

        Delegates to ``RetrieverModule.retrieve_fused``, which owns the genuinely
        global pipeline: raw per-``(subquery x signal)`` candidates, one RRF, one
        provenance-preserving dedup, one cross-encoder rerank against the original
        user query, one MMR pass, final top-k, then context expansion. This tool
        only turns the resulting records into ``Evidence`` — it performs no
        fusion or reranking of its own, so nothing is reranked twice.
        """
        retrieve_fused = getattr(self.retriever_module, "retrieve_fused", None)
        if retrieve_fused is None:
            # Retriever predates global fusion; fall back to plain concatenation
            # so the tool still works rather than raising.
            logger.debug("%s: retriever has no retrieve_fused; falling back to concat", self.name)
            return self.__class__(self.retriever_module, top_k=self.top_k).run(queries=queries)

        records = (
            retrieve_fused(
                subqueries=queries,
                original_query=original_query if original_query is not None else (queries[0] if queries else None),
                return_metadata=True,
                top_k=self.top_k,
                rrf_k=self.rrf_k,
            )
            or []
        )

        if not records:
            return self._result(evidence=[], summary=f"retrieved 0 chunks across {len(queries)} subqueries (fused)")

        evidence: List[Evidence] = []
        for rank, record in enumerate(records):
            # ``retrieval_queries`` already holds every subquery that surfaced the
            # chunk (from the retriever's provenance-preserving dedup); the first
            # is used as the representative retrieval_query.
            queries_for_record = record.get("retrieval_queries") or []
            primary_query = queries_for_record[0] if queries_for_record else (queries[0] if queries else "")
            item = Evidence.from_retrieval_record(
                record, retrieval_query=primary_query, rank=rank, tool_name=self.name
            )
            evidence.append(item)

        logger.debug("%s fused %d chunks across %d subqueries", self.name, len(evidence), len(queries))
        return self._result(
            evidence=evidence,
            summary=f"fused {len(evidence)} chunks across {len(queries)} subqueries",
        )

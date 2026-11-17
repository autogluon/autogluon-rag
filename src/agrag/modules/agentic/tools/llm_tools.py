"""LLM-backed tools that reuse the existing GeneratorModule.

Both tools call the single configured generator. They are gated behind config
flags and are off or optional by default. Neither produces evidence; they
transform queries/context.
"""

import logging
import re
from typing import List

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.tools.base import Tool, ToolResult

logger = logging.getLogger(LOGGER_NAME)

_REWRITE_INSTRUCTION = (
    "Rewrite the following search query to improve document retrieval. "
    "Return only the rewritten query on a single line -- a plain keyword search "
    "query, with no preamble, no reasoning, no markdown, and no explanation.\n\nQuery: "
)

# Markers that a reasoning-prone model leaks into a "rewrite this query" reply:
# a horizontal-rule separator, or a reasoning/afterthought header. The rewritten
# query is meant to be a single search line, so anything from these markers on is
# chain-of-thought, not query text, and must never reach the retriever (it drags
# newlines and prose into the dense/sparse query and wrecks retrieval).
_REWRITE_CUTOFF_RE = re.compile(
    r"\n\s*-{3,}|^\s*(?:reasoning|thought|note|wait|explanation|let me|actually)\b[:\s]",
    re.IGNORECASE | re.MULTILINE,
)


def sanitize_rewrite(text: str) -> str:
    """Reduce a query-rewrite reply to a single clean search line.

    The model is asked for one search line but (on hard multi-hop questions)
    sometimes returns the line followed by ``---`` / ``**Reasoning:**`` / "Wait,
    let me reconsider..." narration. Passing that verbatim to the retriever
    pollutes the dense/sparse query with prose and newlines. Keep only the text
    before the first reasoning marker, then the first non-empty line, and strip
    surrounding markdown/quote punctuation. Returns "" when nothing usable
    remains so the caller can fall back to the original query.
    """
    if not text:
        return ""
    cut = _REWRITE_CUTOFF_RE.search(text)
    if cut:
        text = text[: cut.start()]
    for line in text.splitlines():
        line = line.strip().strip("`\"'*").strip()
        # Skip a leading label like "Rewritten query:" the model may echo.
        line = re.sub(r"(?i)^(rewritten|search)?\s*query[:\-]\s*", "", line).strip()
        if line:
            return line
    return ""

_COMPRESS_INSTRUCTION = (
    "Compress the following context into a concise, self-contained summary that "
    "preserves all facts needed to answer the query. Return only the summary.\n\n"
)


class QueryRewriteTool(Tool):
    """Improve a query after weak retrieval, using the generator."""

    name = "QueryRewriteTool"

    def __init__(self, generator_module):
        self.generator_module = generator_module

    def run(self, query: str, **kwargs) -> ToolResult:
        prompt = f"{_REWRITE_INSTRUCTION}{query}"
        raw = self.generator_module.generate_response(prompt)
        rewritten = sanitize_rewrite(raw) or (query or "").strip()
        logger.debug("%s rewrote %r -> %r (raw=%r)", self.name, query, rewritten, raw)
        return self._result(output=rewritten, summary="rewrote query")


class ContextCompressionTool(Tool):
    """Compress many chunks into smaller context using the generator."""

    name = "ContextCompressionTool"

    def __init__(self, generator_module):
        self.generator_module = generator_module

    def run(self, query: str, texts: List[str], **kwargs) -> ToolResult:
        joined = "\n\n".join(texts)
        prompt = f"{_COMPRESS_INSTRUCTION}Query: {query}\n\nContext:\n{joined}"
        compressed = self.generator_module.generate_response(prompt).strip()
        logger.debug("%s compressed %d chunks", self.name, len(texts))
        return self._result(output=compressed, summary=f"compressed {len(texts)} chunks")

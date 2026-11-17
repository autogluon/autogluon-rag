"""Agentic RAG components for AutoGluon-RAG.

This package adds an optional agentic answering path on top of the standard RAG
pipeline. The standard path (``generate_response(query)``) remains the default;
agentic behavior only runs when explicitly enabled via ``mode="agentic"`` or the
``agent`` configuration block.
"""

from agrag.modules.agentic.agentic_module import AgenticRAGModule

__all__ = ["AgenticRAGModule"]

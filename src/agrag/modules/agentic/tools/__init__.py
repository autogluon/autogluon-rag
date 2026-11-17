"""Tools available to the agentic RAG executor.

Tools are the only way the agent interacts with the outside world (retriever,
generator). The ``ToolRegistry`` controls which tools the agent may call, keeping
the toolbox small and predictable for the MVP.
"""

from agrag.modules.agentic.tools.base import Tool, ToolResult
from agrag.modules.agentic.tools.llm_tools import ContextCompressionTool, QueryRewriteTool
from agrag.modules.agentic.tools.registry import ToolRegistry
from agrag.modules.agentic.tools.retrieve_tools import MultiQueryRetrieveTool, RetrieveTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "RetrieveTool",
    "MultiQueryRetrieveTool",
    "QueryRewriteTool",
    "ContextCompressionTool",
]

"""Base classes for agentic tools."""

from dataclasses import dataclass, field
from typing import Any, List

from agrag.modules.agentic.evidence import Evidence


@dataclass
class ToolResult:
    """The outcome of running a tool.

    Attributes:
    ----------
    tool_name : str
        Name of the tool that produced this result.
    evidence : List[Evidence]
        Any evidence produced (empty for tools that do not retrieve).
    output : Any
        A tool-specific payload (e.g. a rewritten query string, compressed
        context). ``None`` when the tool only produces evidence.
    summary : str
        Short, serializable description of what happened (for the trace).
    """

    tool_name: str
    evidence: List[Evidence] = field(default_factory=list)
    output: Any = None
    summary: str = ""

    @property
    def contains_evidence(self) -> bool:
        return bool(self.evidence)


class Tool:
    """Base class for a tool the agent can call.

    Subclasses set ``name`` and implement ``run``. Keeping a common base lets the
    ``ToolRegistry`` validate and dispatch uniformly.
    """

    name: str = "tool"

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def _result(self, **kwargs) -> ToolResult:
        """Convenience for building a ToolResult tagged with this tool's name."""
        kwargs.setdefault("tool_name", self.name)
        return ToolResult(**kwargs)

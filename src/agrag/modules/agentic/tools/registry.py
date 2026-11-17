"""Registry controlling which tools the agent may call."""

import logging
from typing import Dict, List

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.tools.base import Tool, ToolResult

logger = logging.getLogger(LOGGER_NAME)


class ToolRegistry:
    """Holds the allowed tools and dispatches calls by name.

    Keeping dispatch here (rather than letting the executor call tools directly)
    means the set of callable tools is explicit and validated: an action naming a
    tool that was not registered raises rather than silently doing nothing.
    """

    def __init__(self, tools: List[Tool] = None):
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not getattr(tool, "name", None):
            raise ValueError("Tool must define a non-empty 'name'")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered. Available: {self.names()}")
        return self._tools[name]

    def run(self, name: str, **kwargs) -> ToolResult:
        """Validate the tool name and run it with the given arguments."""
        tool = self.get(name)
        logger.debug("Running tool %s with args %s", name, list(kwargs.keys()))
        return tool.run(**kwargs)

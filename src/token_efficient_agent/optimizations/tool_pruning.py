from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool

from token_efficient_agent.tools.registry import ToolRegistry


def get_tool_descriptions(
    registry: ToolRegistry,
    pruning_enabled: bool,
) -> list[BaseTool]:
    """Return tools with verbose or pruned descriptions based on the flag."""
    mode: Literal["verbose", "pruned"] = "pruned" if pruning_enabled else "verbose"
    return registry.get_tools(mode=mode)

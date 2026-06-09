from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool


@dataclass
class ToolEntry:
    tool: BaseTool
    pruned_description: str
    routing_description: str


class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        pruned_description: str,
        routing_description: str,
    ) -> None:
        self._entries[tool.name] = ToolEntry(
            tool=tool,
            pruned_description=pruned_description,
            routing_description=routing_description,
        )

    @property
    def tool_names(self) -> list[str]:
        return list(self._entries.keys())

    def get_tools(
        self, mode: Literal["verbose", "pruned", "routing"] = "verbose"
    ) -> list[BaseTool]:
        tools = []
        for entry in self._entries.values():
            if mode == "verbose":
                tools.append(entry.tool)
            elif mode == "pruned":
                clone = _clone_tool_with_description(entry.tool, entry.pruned_description)
                tools.append(clone)
            else:
                clone = _clone_tool_with_description(entry.tool, entry.routing_description)
                tools.append(clone)
        return tools

    def get_tool(self, name: str, mode: Literal["verbose", "pruned", "routing"] = "verbose") -> BaseTool:
        entry = self._entries[name]
        if mode == "verbose":
            return entry.tool
        desc = entry.pruned_description if mode == "pruned" else entry.routing_description
        return _clone_tool_with_description(entry.tool, desc)


def _clone_tool_with_description(tool: BaseTool, description: str) -> BaseTool:
    clone = tool.model_copy()
    clone.description = description
    return clone


def create_default_registry() -> ToolRegistry:
    from token_efficient_agent.tools.web_search import web_search_tool, WEB_SEARCH_PRUNED, WEB_SEARCH_ROUTING
    from token_efficient_agent.tools.database import sql_query_tool, SQL_QUERY_PRUNED, SQL_QUERY_ROUTING
    from token_efficient_agent.tools.calculator import calculator_tool, CALCULATOR_PRUNED, CALCULATOR_ROUTING
    from token_efficient_agent.tools.file_ops import file_read_tool, file_write_tool, FILE_READ_PRUNED, FILE_READ_ROUTING, FILE_WRITE_PRUNED, FILE_WRITE_ROUTING
    from token_efficient_agent.tools.email import send_email_tool, SEND_EMAIL_PRUNED, SEND_EMAIL_ROUTING
    from token_efficient_agent.tools.code_exec import run_code_tool, RUN_CODE_PRUNED, RUN_CODE_ROUTING

    registry = ToolRegistry()

    registry.register(web_search_tool, pruned_description=WEB_SEARCH_PRUNED, routing_description=WEB_SEARCH_ROUTING)
    registry.register(sql_query_tool, pruned_description=SQL_QUERY_PRUNED, routing_description=SQL_QUERY_ROUTING)
    registry.register(calculator_tool, pruned_description=CALCULATOR_PRUNED, routing_description=CALCULATOR_ROUTING)
    registry.register(file_read_tool, pruned_description=FILE_READ_PRUNED, routing_description=FILE_READ_ROUTING)
    registry.register(file_write_tool, pruned_description=FILE_WRITE_PRUNED, routing_description=FILE_WRITE_ROUTING)
    registry.register(send_email_tool, pruned_description=SEND_EMAIL_PRUNED, routing_description=SEND_EMAIL_ROUTING)
    registry.register(run_code_tool, pruned_description=RUN_CODE_PRUNED, routing_description=RUN_CODE_ROUTING)

    return registry

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from token_efficient_agent.tools.registry import ToolRegistry


class ToolRouter:
    """Manages progressive tool disclosure — two-phase tool loading.

    Phase 1: LLM sees only short routing descriptions (name + one line).
             It picks which tool to use via a special "select_tool" tool.
    Phase 2: Full tool definition is loaded and bound for the actual call.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._routing_tools: list[BaseTool] | None = None

    def get_routing_tool(self) -> BaseTool:
        """Return a meta-tool that lets the agent select which tool to use.

        The tool's description lists all available tools with one-line summaries.
        The agent calls select_tool(name="tool_name") to load the full definition.
        """
        tool_list = "\n".join(
            f"  - {t.name}: {t.description}"
            for t in self.registry.get_tools(mode="routing")
        )

        @StructuredTool.from_function
        def select_tool(name: str) -> str:
            """Select a tool to use. Call this first, then use the loaded tool.

            Available tools:
            {tool_list}
            """
            if name not in self.registry.tool_names:
                return f"Unknown tool '{name}'. Available: {', '.join(self.registry.tool_names)}"
            return f"Tool '{name}' loaded. You can now use it."

        select_tool.description = (
            f"Select a tool to use by name. Call this first to load the full tool definition, "
            f"then use the loaded tool in your next step.\n\nAvailable tools:\n{tool_list}"
        )
        return select_tool

    def get_full_tool(self, name: str) -> BaseTool:
        """Load the full (verbose) tool definition for execution."""
        return self.registry.get_tool(name, mode="verbose")

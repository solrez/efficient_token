from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from token_efficient_agent.config import AgentConfig, ModelTier
from token_efficient_agent.state import AgentState
from token_efficient_agent.provider import create_chat_model
from token_efficient_agent.tools.registry import ToolRegistry, create_default_registry
from token_efficient_agent.optimizations.observation_masking import apply_observation_masking
from token_efficient_agent.optimizations.prompt_caching import apply_cache_layout
from token_efficient_agent.optimizations.tool_pruning import get_tool_descriptions
from token_efficient_agent.optimizations.progressive_disclosure import ToolRouter
from token_efficient_agent.optimizations.model_routing import classify_complexity

SYSTEM_PROMPT = (
    "You are a helpful research and analysis assistant. You have access to various tools "
    "to search the web, query databases, perform calculations, read/write files, "
    "send emails, and execute code. Use the appropriate tools to complete tasks thoroughly. "
    "Always verify your work and provide clear, actionable outputs."
)


def build_agent_graph(
    config: AgentConfig,
    registry: ToolRegistry | None = None,
    callbacks: list | None = None,
) -> StateGraph:
    """Build a LangGraph agent with optimizations applied based on config flags."""
    if registry is None:
        registry = create_default_registry()

    opts = config.optimizations
    tool_router = ToolRouter(registry) if opts.progressive_disclosure else None

    real_tools = get_tool_descriptions(registry, pruning_enabled=False)
    tool_node = ToolNode(real_tools)

    async def agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        if opts.observation_masking:
            messages = apply_observation_masking(messages, opts.observation_masking_window)

        if opts.prompt_caching:
            messages = apply_cache_layout(messages, config.provider)

        tier = ModelTier(state.get("current_model_tier", "strong"))
        llm = create_chat_model(config, tier=tier)

        bound_tools = _get_bound_tools(state, config, registry, tool_router, real_tools)
        if bound_tools:
            llm = llm.bind_tools(bound_tools)

        response = await llm.ainvoke(messages, config={"callbacks": callbacks or []})

        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
            "selected_tool": _extract_selected_tool(response),
        }

    async def handle_select_tool(state: AgentState) -> dict:
        """Handle the select_tool call by synthesizing a ToolMessage response."""
        last = state["messages"][-1]
        tool_messages = []
        for tc in last.tool_calls:
            if tc["name"] == "select_tool":
                name = tc["args"].get("name", "")
                if name in registry.tool_names:
                    content = f"Tool '{name}' loaded. You can now call it directly."
                else:
                    content = f"Unknown tool '{name}'. Available: {', '.join(registry.tool_names)}"
                tool_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=tc["id"],
                    name="select_tool",
                ))
        return {"messages": tool_messages, "selected_tool": _extract_selected_tool(last)}

    async def routing_classifier_node(state: AgentState) -> dict:
        if not config.optimizations.model_routing:
            return {"current_model_tier": ModelTier.STRONG.value}

        if state.get("step_count", 0) == 0:
            return {"current_model_tier": ModelTier.STRONG.value}

        tier = await classify_complexity(state["messages"], config)
        return {"current_model_tier": tier.value}

    def should_continue(state: AgentState) -> Literal["tools", "select", "end"]:
        last = state["messages"][-1]

        if state.get("step_count", 0) >= config.max_steps:
            return "end"

        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "end"

        if opts.progressive_disclosure and _is_select_tool_call(last):
            return "select"

        return "tools"

    graph = StateGraph(AgentState)

    if opts.model_routing:
        graph.add_node("routing_classifier", routing_classifier_node)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        if opts.progressive_disclosure:
            graph.add_node("handle_select", handle_select_tool)

        graph.add_edge(START, "routing_classifier")
        graph.add_edge("routing_classifier", "agent")

        edges = {"tools": "tools", "end": END}
        if opts.progressive_disclosure:
            edges["select"] = "handle_select"
            graph.add_edge("handle_select", "agent")
        graph.add_conditional_edges("agent", should_continue, edges)
        graph.add_edge("tools", "routing_classifier")
    else:
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        if opts.progressive_disclosure:
            graph.add_node("handle_select", handle_select_tool)

        graph.add_edge(START, "agent")

        edges = {"tools": "tools", "end": END}
        if opts.progressive_disclosure:
            edges["select"] = "handle_select"
            graph.add_edge("handle_select", "agent")
        graph.add_conditional_edges("agent", should_continue, edges)
        graph.add_edge("tools", "agent")

    return graph.compile()


def _get_bound_tools(
    state: AgentState,
    config: AgentConfig,
    registry: ToolRegistry,
    tool_router: ToolRouter | None,
    all_tools: list[BaseTool],
) -> list[BaseTool]:
    """Determine which tools to bind for this step."""
    opts = config.optimizations

    if opts.progressive_disclosure and tool_router:
        selected = state.get("selected_tool")
        if selected and selected in registry.tool_names:
            full_tool = tool_router.get_full_tool(selected)
            return [full_tool]
        return [tool_router.get_routing_tool()]

    if opts.tool_pruning:
        return get_tool_descriptions(registry, pruning_enabled=True)

    return all_tools


def _is_select_tool_call(message: AIMessage) -> bool:
    if not message.tool_calls:
        return False
    return any(tc["name"] == "select_tool" for tc in message.tool_calls)


def _extract_selected_tool(response: AIMessage) -> str | None:
    if not isinstance(response, AIMessage) or not response.tool_calls:
        return None
    for tc in response.tool_calls:
        if tc["name"] == "select_tool":
            return tc["args"].get("name")
    return None

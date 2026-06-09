from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from token_efficient_agent.config import ProviderType


def apply_cache_layout(
    messages: list[BaseMessage],
    provider: ProviderType,
) -> list[BaseMessage]:
    """Reorder messages so static content forms a stable cacheable prefix.

    Layout:
      1. System message(s) — with cache_control for Anthropic
      2. Conversation history (human/AI/tool messages)
      3. Dynamic working-memory content at the tail

    For Anthropic, adds cache_control breakpoints on the system message
    and at the boundary between stable history and recent dynamic content.
    OpenAI uses automatic prefix caching, so ordering alone is sufficient.
    """
    system_msgs = []
    conversation_msgs = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_msgs.append(msg)
        else:
            conversation_msgs.append(msg)

    if provider == ProviderType.ANTHROPIC:
        system_msgs = _add_anthropic_cache_control(system_msgs, conversation_msgs)

    return system_msgs + conversation_msgs


def _add_anthropic_cache_control(
    system_msgs: list[BaseMessage],
    conversation_msgs: list[BaseMessage],
) -> list[BaseMessage]:
    """Add cache_control markers to Anthropic messages for prompt caching."""
    result = []
    for msg in system_msgs:
        patched = msg.model_copy()
        patched.additional_kwargs = {
            **patched.additional_kwargs,
            "cache_control": {"type": "ephemeral"},
        }
        result.append(patched)
    return result

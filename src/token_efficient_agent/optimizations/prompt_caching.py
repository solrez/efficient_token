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

    For Anthropic, adds cache_control on system message content blocks.
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
        system_msgs = _add_anthropic_cache_control(system_msgs)

    return system_msgs + conversation_msgs


def _add_anthropic_cache_control(
    system_msgs: list[BaseMessage],
) -> list[BaseMessage]:
    """Add cache_control markers to Anthropic system messages.

    Restructures system message content as a list of text blocks with
    cache_control on the last block, which is how langchain-anthropic
    passes it through to the Anthropic API.
    """
    result = []
    for msg in system_msgs:
        content = msg.content
        if isinstance(content, str):
            patched = SystemMessage(content=[
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ])
        elif isinstance(content, list):
            blocks = list(content)
            if blocks:
                last = blocks[-1]
                if isinstance(last, dict):
                    blocks[-1] = {**last, "cache_control": {"type": "ephemeral"}}
                elif isinstance(last, str):
                    blocks[-1] = {"type": "text", "text": last, "cache_control": {"type": "ephemeral"}}
            patched = SystemMessage(content=blocks)
        else:
            patched = msg
        result.append(patched)
    return result

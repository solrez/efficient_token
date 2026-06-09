from __future__ import annotations

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage


def apply_observation_masking(
    messages: list[BaseMessage],
    window_k: int = 3,
) -> list[BaseMessage]:
    """Replace tool output content older than the last K tool exchanges with placeholders.

    Preserves the action trace (tool name + arguments in AIMessage tool_calls)
    but drops raw output data from ToolMessages outside the window.
    """
    tool_msg_indices = [
        i for i, m in enumerate(messages) if isinstance(m, ToolMessage)
    ]

    if len(tool_msg_indices) <= window_k:
        return messages

    indices_to_mask = set(tool_msg_indices[:-window_k])

    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_mask:
            masked = ToolMessage(
                content=f"[output omitted — step {i}]",
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, "name", None),
            )
            result.append(masked)
        else:
            result.append(msg)

    return result

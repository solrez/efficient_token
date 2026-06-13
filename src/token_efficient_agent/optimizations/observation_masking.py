from __future__ import annotations

import os
import re

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage


def apply_observation_masking(
    messages: list[BaseMessage],
    window_k: int = 3,
    use_llm: bool = False,
    llm_model: str = "openrouter/qwen/qwen3.5-9b",
) -> list[BaseMessage]:
    """Content-aware masking: replace old tool outputs with compact summaries.

    Two modes:
    - regex (default): Fast, free, extracts IDs/statuses/names via patterns
    - llm: Uses a cheap small model to summarize, better for unstructured content
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
            tool_name = getattr(msg, "name", None) or "tool"
            original = msg.content if isinstance(msg.content, str) else str(msg.content)
            if use_llm:
                summary = _llm_summarize(tool_name, original, llm_model)
            else:
                summary = _extract_key_info(tool_name, original)
            masked = ToolMessage(
                content=summary,
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, "name", None),
            )
            result.append(masked)
        else:
            result.append(msg)

    return result


def _extract_key_info(tool_name: str, content: str) -> str:
    """Regex-based extraction: free, instant, good for structured data."""
    if not content or len(content) < 100:
        return content

    key_items = []

    ids = re.findall(r'["\']?(?:id|order_id|user_id|product_id)["\']?\s*[:=]\s*["\']?([#\w-]+)', content, re.IGNORECASE)
    if ids:
        key_items.append(f"IDs: {', '.join(dict.fromkeys(ids))}")

    statuses = re.findall(r'["\']?(?:status|state)["\']?\s*[:=]\s*["\']?(\w+)', content, re.IGNORECASE)
    if statuses:
        key_items.append(f"status: {', '.join(dict.fromkeys(statuses))}")

    names = re.findall(r'["\']?(?:name|first_name|last_name)["\']?\s*[:=]\s*["\']?([A-Z][\w\s]+?)(?:["\',}])', content)
    if names:
        key_items.append(f"name: {', '.join(dict.fromkeys(names[:3]))}")

    amounts = re.findall(r'["\']?(?:amount|total|price)["\']?\s*[:=]\s*["\']?([\d.]+)', content, re.IGNORECASE)
    if amounts:
        key_items.append(f"amounts: {', '.join(dict.fromkeys(amounts[:3]))}")

    if key_items:
        return f"[{tool_name} summary: {'; '.join(key_items)}]"

    first_line = content.split('\n')[0][:150]
    return f"[{tool_name}: {first_line}...]"


def _llm_summarize(tool_name: str, content: str, model: str) -> str:
    """LLM-based summarization: uses a cheap small model for better quality."""
    if not content or len(content) < 100:
        return content

    try:
        from litellm import completion

        res = completion(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this {tool_name} tool output in ONE line. "
                    f"Keep all IDs, statuses, names, amounts, and key facts. "
                    f"Drop addresses, descriptions, and verbose details.\n\n"
                    f"{content}"
                ),
            }],
            max_tokens=100,
            temperature=0.0,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            api_base="https://openrouter.ai/api/v1",
        )
        summary = res.choices[0].message.content.strip()
        return f"[{tool_name} summary: {summary}]"
    except Exception:
        return _extract_key_info(tool_name, content)

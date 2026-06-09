from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from token_efficient_agent.config import AgentConfig, ModelTier
from token_efficient_agent.provider import create_chat_model

CLASSIFIER_PROMPT = """Classify the complexity of the next action the AI agent needs to take.

SIMPLE tasks (use small model): summarization, data extraction, formatting, unit conversion,
simple lookups, template filling, classification into known categories.

COMPLEX tasks (use large model): multi-step reasoning, planning, code generation,
SQL query construction, analysis requiring judgment, novel problem solving,
ambiguous or open-ended requests.

Based on the conversation so far, is the NEXT agent action simple or complex?
Respond with exactly one word: SIMPLE or COMPLEX"""


async def classify_complexity(
    messages: list[BaseMessage],
    config: AgentConfig,
) -> ModelTier:
    """Use a cheap model to classify whether the next step needs the strong model."""
    llm = create_chat_model(config, tier=ModelTier.WEAK, max_tokens=10)

    classification_messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=_summarize_recent_context(messages)),
    ]

    response = await llm.ainvoke(classification_messages)
    text = response.content.strip().upper()

    if "SIMPLE" in text:
        return ModelTier.WEAK
    return ModelTier.STRONG


def _summarize_recent_context(messages: list[BaseMessage]) -> str:
    """Extract the last few messages as context for the classifier."""
    recent = messages[-4:]
    parts = []
    for msg in recent:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        truncated = content[:300] + "..." if len(content) > 300 else content
        parts.append(f"{role}: {truncated}")
    return "\n".join(parts)

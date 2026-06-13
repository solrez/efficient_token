from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from token_efficient_agent.config import AgentConfig, ModelTier, ProviderType


def create_chat_model(
    config: AgentConfig,
    tier: ModelTier = ModelTier.STRONG,
    **kwargs,
) -> BaseChatModel:
    model_name = config.get_model(tier)

    if config.provider == ProviderType.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, **kwargs)

    if config.provider == ProviderType.OPENROUTER:
        from langchain_openai import ChatOpenAI
        import os

        return ChatOpenAI(
            model=model_name,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/solrez/efficient_token",
                "X-Title": "Token Efficiency Research",
            },
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, **kwargs)

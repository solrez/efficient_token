from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    DEEPINFRA = "deepinfra"
    GROQ = "groq"
    DEEPSEEK = "deepseek"


class ModelTier(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


MODELS = {
    ProviderType.ANTHROPIC: {
        ModelTier.STRONG: "claude-sonnet-4-6",
        ModelTier.WEAK: "claude-haiku-4-5",
    },
    ProviderType.OPENAI: {
        ModelTier.STRONG: "gpt-4o",
        ModelTier.WEAK: "gpt-4o-mini",
    },
    ProviderType.OPENROUTER: {
        ModelTier.STRONG: "deepseek/deepseek-v4-flash",
        ModelTier.WEAK: "deepseek/deepseek-v4-flash",
    },
}

OPENROUTER_MODEL_PRESETS = {
    "deepseek-v4-flash": {
        ModelTier.STRONG: "deepseek/deepseek-v4-flash",
        ModelTier.WEAK: "google/gemma-4-26b-a4b-it",
    },
    "deepseek-v4-pro": {
        ModelTier.STRONG: "deepseek/deepseek-v4-pro",
        ModelTier.WEAK: "deepseek/deepseek-v4-flash",
    },
    "qwen3-flash": {
        ModelTier.STRONG: "qwen/qwen3.6-flash",
        ModelTier.WEAK: "qwen/qwen3.5-9b",
    },
    "qwen3-plus": {
        ModelTier.STRONG: "qwen/qwen3.7-plus",
        ModelTier.WEAK: "qwen/qwen3.6-flash",
    },
    "llama4-scout": {
        ModelTier.STRONG: "meta-llama/llama-4-scout",
        ModelTier.WEAK: "qwen/qwen3.5-9b",
    },
    "mistral-small": {
        ModelTier.STRONG: "mistralai/mistral-small-4-2506",
        ModelTier.WEAK: "qwen/qwen3.5-9b",
    },
}

INPUT_COST_PER_MTOK = {
    # Anthropic
    "claude-sonnet-4-6": 3.0,
    "claude-haiku-4-5": 1.0,
    # OpenAI
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    # OpenRouter models
    "deepseek/deepseek-v4-flash": 0.098,
    "deepseek/deepseek-v4-pro": 0.435,
    "qwen/qwen3.6-flash": 0.188,
    "qwen/qwen3.7-plus": 0.32,
    "qwen/qwen3.7-max": 1.25,
    "meta-llama/llama-4-scout": 0.10,
    "qwen/qwen3.5-9b": 0.10,
    "google/gemma-4-26b-a4b-it": 0.06,
    # DeepInfra models
    "Qwen/Qwen3-14B": 0.12,
    "Qwen/Qwen3-30B-A3B": 0.12,
    "Qwen/Qwen3-32B": 0.08,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 0.09,
    "google/gemma-4-26B-A4B-it": 0.07,
    "google/gemma-3-27b-it": 0.08,
    "google/gemma-3-12b-it": 0.05,
    "google/gemma-3-4b-it": 0.05,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.02,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 0.075,
    # DeepSeek direct API (cache miss price)
    "deepseek-v4-flash": 0.14,
    "deepseek-v4-pro": 0.435,
    "deepseek-reasoner": 0.14,
}

OUTPUT_COST_PER_MTOK = {
    # Anthropic
    "claude-sonnet-4-6": 15.0,
    "claude-haiku-4-5": 5.0,
    # OpenAI
    "gpt-4o": 10.0,
    "gpt-4o-mini": 0.60,
    # OpenRouter models
    "deepseek/deepseek-v4-flash": 0.196,
    "deepseek/deepseek-v4-pro": 0.870,
    "qwen/qwen3.6-flash": 1.125,
    "qwen/qwen3.7-plus": 1.28,
    "qwen/qwen3.7-max": 3.75,
    "meta-llama/llama-4-scout": 0.30,
    "qwen/qwen3.5-9b": 0.15,
    "google/gemma-4-26b-a4b-it": 0.33,
    # DeepInfra models
    "Qwen/Qwen3-14B": 0.24,
    "Qwen/Qwen3-30B-A3B": 0.50,
    "Qwen/Qwen3-32B": 0.28,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 0.10,
    "google/gemma-4-26B-A4B-it": 0.34,
    "google/gemma-3-27b-it": 0.16,
    "google/gemma-3-12b-it": 0.15,
    "google/gemma-3-4b-it": 0.10,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.03,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 0.20,
    # DeepSeek direct API
    "deepseek-v4-flash": 0.28,
    "deepseek-v4-pro": 0.87,
    "deepseek-reasoner": 0.28,
}

CONTEXT_WINDOW = {
    # Anthropic
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    # OpenRouter models
    "deepseek/deepseek-v4-flash": 1_048_576,
    "deepseek/deepseek-v4-pro": 163_840,
    "qwen/qwen3.6-flash": 131_072,
    "qwen/qwen3.7-plus": 131_072,
    "qwen/qwen3.7-max": 131_072,
    "meta-llama/llama-4-scout": 131_072,
    "mistralai/mistral-small-4-2506": 128_000,
    "qwen/qwen3.5-9b": 40_960,
    "google/gemma-4-26b-a4b-it": 262_144,
    # DeepInfra models
    "Qwen/Qwen3-14B": 40_960,
    "Qwen/Qwen3-30B-A3B": 40_960,
    "Qwen/Qwen3-32B": 40_960,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 262_144,
    "google/gemma-4-26B-A4B-it": 262_144,
    "google/gemma-3-27b-it": 131_072,
    "google/gemma-3-12b-it": 131_072,
    "google/gemma-3-4b-it": 131_072,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 131_072,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 128_000,
    # Groq models
    "qwen/qwen3-32b": 131_072,
    # DeepSeek direct API
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4-pro": 163_840,
    "deepseek-reasoner": 163_840,
}

CACHE_READ_COST_PER_MTOK = {
    "claude-sonnet-4-6": 0.30,
    "claude-haiku-4-5": 0.10,
    "gpt-4o": 1.25,
    "gpt-4o-mini": 0.075,
}

CACHE_WRITE_COST_PER_MTOK = {
    "claude-sonnet-4-6": 3.75,
    "claude-haiku-4-5": 1.25,
}


class OptimizationFlags(BaseModel):
    prompt_caching: bool = False
    tool_pruning: bool = False
    progressive_disclosure: bool = False
    observation_masking: bool = False
    observation_masking_window: int = Field(default=3, ge=1)
    observation_masking_llm: bool = False
    model_routing: bool = False

    def label(self) -> str:
        parts = []
        if self.prompt_caching:
            parts.append("cache")
        if self.tool_pruning:
            parts.append("prune")
        if self.progressive_disclosure:
            parts.append("progressive")
        if self.observation_masking:
            parts.append(f"mask(k={self.observation_masking_window})")
        if self.model_routing:
            parts.append("route")
        return "+".join(parts) if parts else "baseline"


class TracingConfig(BaseModel):
    enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


class AgentConfig(BaseModel):
    provider: ProviderType = ProviderType.ANTHROPIC
    openrouter_model: str = ""
    optimizations: OptimizationFlags = Field(default_factory=OptimizationFlags)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    max_steps: int = 15

    def get_context_window(self, tier: ModelTier = ModelTier.STRONG) -> int:
        model = self.get_model(tier)
        return CONTEXT_WINDOW.get(model, 131_072)

    def get_model(self, tier: ModelTier = ModelTier.STRONG) -> str:
        if self.provider in (ProviderType.DEEPINFRA, ProviderType.GROQ, ProviderType.DEEPSEEK):
            return self.openrouter_model
        if self.provider == ProviderType.OPENROUTER and self.openrouter_model:
            preset = OPENROUTER_MODEL_PRESETS.get(self.openrouter_model)
            if preset:
                return preset[tier]
            return self.openrouter_model
        return MODELS[self.provider][tier]

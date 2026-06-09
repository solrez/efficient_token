from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTier(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


MODELS = {
    ProviderType.ANTHROPIC: {
        ModelTier.STRONG: "claude-sonnet-4-20250514",
        ModelTier.WEAK: "claude-haiku-4-5-20251001",
    },
    ProviderType.OPENAI: {
        ModelTier.STRONG: "gpt-4o",
        ModelTier.WEAK: "gpt-4o-mini",
    },
}

INPUT_COST_PER_MTOK = {
    "claude-sonnet-4-20250514": 3.0,
    "claude-haiku-4-5-20251001": 0.80,
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
}

OUTPUT_COST_PER_MTOK = {
    "claude-sonnet-4-20250514": 15.0,
    "claude-haiku-4-5-20251001": 4.0,
    "gpt-4o": 10.0,
    "gpt-4o-mini": 0.60,
}

CACHE_READ_COST_PER_MTOK = {
    "claude-sonnet-4-20250514": 0.30,
    "claude-haiku-4-5-20251001": 0.08,
}

CACHE_WRITE_COST_PER_MTOK = {
    "claude-sonnet-4-20250514": 3.75,
    "claude-haiku-4-5-20251001": 1.0,
}


class OptimizationFlags(BaseModel):
    prompt_caching: bool = False
    tool_pruning: bool = False
    progressive_disclosure: bool = False
    observation_masking: bool = False
    observation_masking_window: int = Field(default=3, ge=1)
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
    optimizations: OptimizationFlags = Field(default_factory=OptimizationFlags)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    max_steps: int = 15

    def get_model(self, tier: ModelTier = ModelTier.STRONG) -> str:
        return MODELS[self.provider][tier]

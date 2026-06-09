from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, AIMessage

from token_efficient_agent.config import (
    AgentConfig,
    OptimizationFlags,
    INPUT_COST_PER_MTOK,
    OUTPUT_COST_PER_MTOK,
    CACHE_READ_COST_PER_MTOK,
    CACHE_WRITE_COST_PER_MTOK,
)


@dataclass
class StepMetrics:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class RunMetrics:
    scenario: str = ""
    optimization_label: str = ""
    provider: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    num_llm_calls: int = 0
    latency_ms: float = 0.0
    steps: list[StepMetrics] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        if not self.steps:
            return 0.0
        cost = 0.0
        for step in self.steps:
            input_rate = INPUT_COST_PER_MTOK.get(step.model, 3.0)
            output_rate = OUTPUT_COST_PER_MTOK.get(step.model, 15.0)
            cache_read_rate = CACHE_READ_COST_PER_MTOK.get(step.model, input_rate)
            cache_write_rate = CACHE_WRITE_COST_PER_MTOK.get(step.model, input_rate)

            non_cached_input = step.input_tokens - step.cache_read_tokens - step.cache_write_tokens
            cost += non_cached_input / 1_000_000 * input_rate
            cost += step.output_tokens / 1_000_000 * output_rate
            cost += step.cache_read_tokens / 1_000_000 * cache_read_rate
            cost += step.cache_write_tokens / 1_000_000 * cache_write_rate
        return round(cost, 6)


def extract_metrics_from_messages(
    messages: list[BaseMessage],
    scenario: str,
    optimization_label: str,
    provider: str,
    model_name: str,
    latency_ms: float,
) -> RunMetrics:
    """Extract token usage directly from LLM response message metadata."""
    metrics = RunMetrics(
        scenario=scenario,
        optimization_label=optimization_label,
        provider=provider,
        latency_ms=latency_ms,
    )

    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue

        cache_info = usage.get("input_token_details", {}) or {}
        step = StepMetrics(
            model=model_name,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=cache_info.get("cache_read", 0) or 0,
            cache_write_tokens=cache_info.get("cache_creation", 0) or 0,
        )
        metrics.steps.append(step)
        metrics.total_input_tokens += step.input_tokens
        metrics.total_output_tokens += step.output_tokens
        metrics.total_cache_read_tokens += step.cache_read_tokens
        metrics.total_cache_write_tokens += step.cache_write_tokens
        metrics.num_llm_calls += 1

    return metrics


class TracingManager:
    """Manages Langfuse tracing with optimization-aware metadata."""

    def __init__(self, config: AgentConfig, session_id: str | None = None) -> None:
        self.config = config
        self.session_id = session_id
        self._langfuse = None

        if config.tracing.enabled and config.tracing.langfuse_secret_key:
            import os
            os.environ["LANGFUSE_PUBLIC_KEY"] = config.tracing.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = config.tracing.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = config.tracing.langfuse_host

            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse()
                self._langfuse.auth_check()
                print(f"  [Langfuse] Connected to {config.tracing.langfuse_host}")
            except Exception as e:
                print(f"  [Langfuse] Connection failed: {e}")
                self._langfuse = None

    def create_callback(
        self,
        scenario_name: str,
        optimization_flags: OptimizationFlags,
    ):
        """Create a Langfuse callback handler for a single agent run."""
        if not self._langfuse:
            return None

        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler(
            trace_context={
                "name": f"{scenario_name}/{optimization_flags.label()}",
                "session_id": self.session_id,
                "metadata": {
                    "prompt_caching": optimization_flags.prompt_caching,
                    "tool_pruning": optimization_flags.tool_pruning,
                    "progressive_disclosure": optimization_flags.progressive_disclosure,
                    "observation_masking": optimization_flags.observation_masking,
                    "observation_masking_window": optimization_flags.observation_masking_window,
                    "model_routing": optimization_flags.model_routing,
                    "provider": self.config.provider.value,
                },
                "tags": [
                    f"scenario:{scenario_name}",
                    f"opt:{optimization_flags.label()}",
                    f"provider:{self.config.provider.value}",
                ],
            },
        )
        return handler

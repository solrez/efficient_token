from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from rich.console import Console

from token_efficient_agent.config import AgentConfig, OptimizationFlags
from token_efficient_agent.graph import build_agent_graph
from token_efficient_agent.tools.registry import create_default_registry
from token_efficient_agent.tracing import TracingManager, RunMetrics, extract_metrics_from_messages
from token_efficient_agent.benchmark.scenarios import BenchmarkScenario, SCENARIOS

console = Console()

DEFAULT_PROFILES: list[OptimizationFlags] = [
    OptimizationFlags(),
    OptimizationFlags(prompt_caching=True),
    OptimizationFlags(tool_pruning=True),
    OptimizationFlags(observation_masking=True),
    OptimizationFlags(progressive_disclosure=True),
    OptimizationFlags(model_routing=True),
    OptimizationFlags(prompt_caching=True, tool_pruning=True),
    OptimizationFlags(prompt_caching=True, tool_pruning=True, observation_masking=True),
    OptimizationFlags(
        prompt_caching=True,
        tool_pruning=True,
        observation_masking=True,
        progressive_disclosure=True,
        model_routing=True,
    ),
]


@dataclass
class BenchmarkResults:
    runs: list[RunMetrics] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def get_runs_for_scenario(self, scenario: str) -> list[RunMetrics]:
        return [r for r in self.runs if r.scenario == scenario]


class BenchmarkRunner:
    def __init__(
        self,
        base_config: AgentConfig,
        scenarios: list[str] | None = None,
        profiles: list[OptimizationFlags] | None = None,
    ) -> None:
        self.base_config = base_config
        self.scenario_names = scenarios or list(SCENARIOS.keys())
        self.profiles = profiles or DEFAULT_PROFILES
        self.results = BenchmarkResults()

    async def run_all(self) -> BenchmarkResults:
        total = len(self.scenario_names) * len(self.profiles)
        idx = 0
        for scenario_name in self.scenario_names:
            scenario = SCENARIOS[scenario_name]
            for profile in self.profiles:
                idx += 1
                console.print(f"  [{idx}/{total}] {scenario.name} / {profile.label()}")
                metrics = await self._run_single(scenario, profile)
                self.results.runs.append(metrics)
                console.print(
                    f"         -> {metrics.total_input_tokens:,} in / "
                    f"{metrics.total_output_tokens:,} out / "
                    f"${metrics.total_cost:.4f} / "
                    f"{metrics.num_llm_calls} calls / "
                    f"{metrics.latency_ms / 1000:.1f}s"
                )
        return self.results

    async def _run_single(
        self,
        scenario: BenchmarkScenario,
        flags: OptimizationFlags,
    ) -> RunMetrics:
        config = self.base_config.model_copy()
        config.optimizations = flags

        registry = create_default_registry()
        tracing = TracingManager(config, session_id=self.results.session_id)

        callback = tracing.create_callback()
        callbacks = [callback] if callback else []

        graph = build_agent_graph(config, registry=registry, callbacks=callbacks)

        initial_state = {
            "messages": [HumanMessage(content=scenario.initial_message)],
            "step_count": 0,
            "selected_tool": None,
            "current_model_tier": "strong",
        }

        start = time.perf_counter()
        try:
            with tracing.trace_context(scenario.name, flags):
                result = await graph.ainvoke(
                    initial_state,
                    config={"callbacks": callbacks, "recursion_limit": config.max_steps * 3},
                )
        except Exception as e:
            console.print(f"         [red]Error: {e}[/red]")
            return RunMetrics(
                scenario=scenario.name,
                optimization_label=flags.label(),
                provider=config.provider.value,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        tracing.flush()

        metrics = extract_metrics_from_messages(
            messages=result["messages"],
            scenario=scenario.name,
            optimization_label=flags.label(),
            provider=config.provider.value,
            model_name=config.get_model(),
            latency_ms=elapsed_ms,
        )

        return metrics

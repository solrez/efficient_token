"""Phase 2 experiment: compare adaptive controller vs static configurations.

Runs the same tasks with:
  1. baseline (no optimization)
  2. static-aggressive (mask=1, prune, progressive)
  3. static-moderate (mask=3, prune)
  4. adaptive (uncertainty-driven controller)

Logs per-step decisions for the adaptive config and compares cost-of-pass.
Results are saved incrementally after each task completes.

Usage:
    python -m adaptive_compression experiment \
        --domain retail --num-tasks 50 --provider deepseek \
        --model deepseek-v4-flash --output results/adaptive_experiment/
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm
litellm.suppress_debug_info = True

from rich.console import Console
from rich.table import Table

from token_efficient_agent.config import (
    AgentConfig, OptimizationFlags, ModelTier, ProviderType,
    INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK,
)
from adaptive_compression.adaptive_agent import AdaptiveTauAgent, AdaptiveTaskResult
from adaptive_compression.controller import ThresholdController

console = Console()

USER_SIM_MODEL = "openrouter/openai/gpt-4o-mini"
USER_SIM_PROVIDER = "openrouter"

STATIC_CONFIGS = {
    "baseline": OptimizationFlags(),
    # Masking only
    "static-mask1": OptimizationFlags(
        observation_masking=True,
        observation_masking_window=1,
    ),
    "static-mask3": OptimizationFlags(
        observation_masking=True,
        observation_masking_window=3,
    ),
    "static-mask5": OptimizationFlags(
        observation_masking=True,
        observation_masking_window=5,
    ),
    # Pruning + masking
    "static-mask1-prune": OptimizationFlags(
        tool_pruning=True,
        observation_masking=True,
        observation_masking_window=1,
    ),
    # Progressive only
    "static-progressive": OptimizationFlags(
        progressive_disclosure=True,
    ),
    # Progressive + masking combinations
    "static-prog-mask3": OptimizationFlags(
        progressive_disclosure=True,
        observation_masking=True,
        observation_masking_window=3,
    ),
    "static-prog-mask5": OptimizationFlags(
        progressive_disclosure=True,
        observation_masking=True,
        observation_masking_window=5,
    ),
}


@dataclass
class ConfigResult:
    config_name: str
    task_results: list[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        valid = [r for r in self.task_results if r["total_steps"] > 0]
        if not valid:
            return 0.0
        return sum(r["reward"] for r in valid) / len(valid)

    @property
    def avg_cost(self) -> float:
        valid = [r for r in self.task_results if r["total_steps"] > 0]
        if not valid:
            return 0.0
        return sum(r["total_cost"] for r in valid) / len(valid)

    @property
    def avg_input_tokens(self) -> float:
        valid = [r for r in self.task_results if r["total_steps"] > 0]
        if not valid:
            return 0.0
        return sum(r["total_input_tokens"] for r in valid) / len(valid)

    @property
    def cost_of_pass(self) -> float:
        pr = self.pass_rate
        if pr <= 0:
            return float("inf")
        return self.avg_cost / pr


def _compute_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    rate_in = INPUT_COST_PER_MTOK.get(model_name, 0)
    rate_out = OUTPUT_COST_PER_MTOK.get(model_name, 0)
    return (input_tokens / 1_000_000 * rate_in) + (output_tokens / 1_000_000 * rate_out)


class AdaptiveExperiment:
    """Run adaptive vs static comparison experiment."""

    def __init__(
        self,
        config: AgentConfig,
        domain: str = "retail",
        task_ids: list[int] | None = None,
        num_tasks: int | None = None,
        temperature: float = 0.0,
        concurrency: int = 1,
        output_dir: str = "results/adaptive_experiment",
        only_configs: list[str] | None = None,
    ):
        self.config = config
        self.domain = domain
        self.task_ids = task_ids
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.only_configs = only_configs
        self._per_task_csv_ready = False
        self._adaptive_csv_ready = False

    def run(self):
        return asyncio.run(self._run_async())

    async def _run_async(self):
        import logging
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        if self.domain == "retail":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as _tasks
        else:
            from tau_bench.envs.airline.tasks_test import TASKS_TEST as _tasks

        task_ids = self.task_ids
        if task_ids is None:
            total = len(_tasks)
            n = self.num_tasks or total
            task_ids = list(range(min(n, total)))

        model_name = self.config.get_model()
        console.print(f"\n[bold]Adaptive Compression Experiment[/bold]")
        console.print(f"  Domain: {self.domain}")
        console.print(f"  Tasks: {len(task_ids)}")
        console.print(f"  Provider: {self.config.provider.value}")
        console.print(f"  Model: {model_name}")
        console.print(f"  Configs: {', '.join(list(STATIC_CONFIGS.keys()) + ['adaptive'])}")
        console.print()

        self._save_params(task_ids)

        all_config_results = {}

        # Run each config sequentially (filter by --configs if specified)
        all_configs = list(STATIC_CONFIGS.items()) + [("adaptive", None)]
        if self.only_configs:
            all_configs = [(n, f) for n, f in all_configs if n in self.only_configs]

        for config_name, flags in all_configs:
            console.print(f"\n  [bold]{'=' * 50}[/bold]")
            console.print(f"  [bold]Config: {config_name}[/bold]")
            console.print(f"  [bold]{'=' * 50}[/bold]")

            cr = ConfigResult(config_name=config_name)
            semaphore = asyncio.Semaphore(self.concurrency)
            completed = 0
            passed = 0

            async def run_one(tid, _config_name=config_name, _flags=flags):
                async with semaphore:
                    if _config_name == "adaptive":
                        return await asyncio.to_thread(self._run_single_adaptive, tid)
                    else:
                        return await asyncio.to_thread(
                            self._run_single_static, tid, _flags, _config_name
                        )

            for tid in task_ids:
                result = await run_one(tid)

                if config_name == "adaptive" and isinstance(result, AdaptiveTaskResult):
                    task_dict = {
                        "task_id": result.task_id,
                        "config": "adaptive",
                        "reward": result.reward,
                        "total_cost": result.total_cost,
                        "total_input_tokens": result.total_input_tokens,
                        "total_output_tokens": result.total_output_tokens,
                        "total_steps": result.total_steps,
                    }
                    cr.task_results.append(task_dict)
                    self._append_per_task(task_dict, model_name)
                    for sl in result.step_logs:
                        self._append_adaptive_step(sl)
                else:
                    if isinstance(result, dict):
                        # Fix cost if litellm didn't compute it
                        if not result.get("total_cost"):
                            result["total_cost"] = _compute_cost(
                                result["total_input_tokens"],
                                result["total_output_tokens"],
                                model_name,
                            )
                        cr.task_results.append(result)
                        self._append_per_task(result, model_name)

                completed += 1
                reward = task_dict["reward"] if config_name == "adaptive" else result.get("reward", 0)
                if reward > 0:
                    passed += 1
                status = "[green]PASS[/green]" if reward > 0 else "[red]FAIL[/red]"
                tid_val = result.task_id if isinstance(result, AdaptiveTaskResult) else result.get("task_id", "?")
                tokens = result.total_input_tokens if isinstance(result, AdaptiveTaskResult) else result.get("total_input_tokens", 0)
                console.print(
                    f"    Task {tid_val:>3}: {status}  "
                    f"tokens={tokens:>7,}  "
                    f"running: {passed}/{completed} ({passed/completed:.0%})"
                )

            all_config_results[config_name] = cr
            console.print(
                f"  [bold]{config_name} done:[/bold] "
                f"pass={cr.pass_rate:.1%}, "
                f"avg_tokens={cr.avg_input_tokens:,.0f}, "
                f"avg_cost=${cr.avg_cost:.4f}"
            )

        self._save_summary(all_config_results)
        self._print_report(all_config_results)

        return all_config_results

    def _run_single_static(
        self, tid: int, flags: OptimizationFlags, config_name: str, max_retries: int = 3
    ) -> dict:
        from tau_bench.envs import get_env
        from token_efficient_agent.accuracy.tau_runner import OptimizedTauAgent

        for attempt in range(1, max_retries + 1):
            try:
                env = get_env(
                    env_name=self.domain,
                    user_strategy="llm",
                    user_model=USER_SIM_MODEL,
                    user_provider=USER_SIM_PROVIDER,
                    task_split="test",
                )
                agent = OptimizedTauAgent(
                    config=self.config,
                    flags=flags,
                    tools_info=env.tools_info,
                    wiki=env.wiki,
                    temperature=self.temperature,
                )
                solve_result, metrics = agent.solve(env, task_index=tid)
                return {
                    "task_id": tid,
                    "config": config_name,
                    "reward": solve_result.reward,
                    "total_cost": solve_result.total_cost,
                    "total_input_tokens": metrics["total_input_tokens"],
                    "total_output_tokens": metrics["total_output_tokens"],
                    "total_steps": metrics["num_steps"],
                }
            except Exception as e:
                is_transient = any(kw in str(e).lower() for kw in [
                    "429", "rate", "timeout", "connection", "model busy", "retry",
                    "overloaded", "502", "503",
                ])
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt * 5
                    console.print(f"    [yellow]Task {tid} ({config_name}) retry {attempt}: {str(e)[:50]}...[/yellow]")
                    time.sleep(wait)
                    continue
                console.print(f"    [red]Task {tid} ({config_name}) failed: {str(e)[:80]}[/red]")
                return {
                    "task_id": tid, "config": config_name, "reward": 0.0,
                    "total_cost": 0.0, "total_input_tokens": 0,
                    "total_output_tokens": 0, "total_steps": 0,
                }

    def _run_single_adaptive(self, tid: int, max_retries: int = 3) -> AdaptiveTaskResult:
        from tau_bench.envs import get_env

        for attempt in range(1, max_retries + 1):
            try:
                env = get_env(
                    env_name=self.domain,
                    user_strategy="llm",
                    user_model=USER_SIM_MODEL,
                    user_provider=USER_SIM_PROVIDER,
                    task_split="test",
                )
                agent = AdaptiveTauAgent(
                    config=self.config,
                    tools_info=env.tools_info,
                    wiki=env.wiki,
                    temperature=self.temperature,
                )
                return agent.solve(env, task_index=tid)
            except Exception as e:
                is_transient = any(kw in str(e).lower() for kw in [
                    "429", "rate", "timeout", "connection", "model busy", "retry",
                    "overloaded", "502", "503",
                ])
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt * 5
                    console.print(f"    [yellow]Task {tid} (adaptive) retry {attempt}: {str(e)[:50]}...[/yellow]")
                    time.sleep(wait)
                    continue
                console.print(f"    [red]Task {tid} (adaptive) failed: {str(e)[:80]}[/red]")
                return AdaptiveTaskResult(
                    task_id=tid, reward=0.0, total_steps=0,
                    total_cost=0.0, total_input_tokens=0, total_output_tokens=0,
                )

    # --- Incremental CSV writing ---

    def _append_per_task(self, result: dict, model_name: str):
        path = self.output_dir / "per_task.csv"
        if not self._per_task_csv_ready:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "config", "task_id", "reward", "total_cost",
                        "total_input_tokens", "total_output_tokens", "total_steps",
                    ])
            self._per_task_csv_ready = True

        cost = result.get("total_cost", 0)
        if not cost:
            cost = _compute_cost(
                result.get("total_input_tokens", 0),
                result.get("total_output_tokens", 0),
                model_name,
            )

        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                result["config"], result["task_id"], result["reward"],
                f"{cost:.6f}", result["total_input_tokens"],
                result["total_output_tokens"], result["total_steps"],
            ])

    def _append_adaptive_step(self, sl):
        path = self.output_dir / "adaptive_steps.csv"
        if not self._adaptive_csv_ready:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "task_id", "step", "tool_called", "confidence",
                        "profile_selected", "mask_window", "pruning_used",
                        "progressive_used", "input_tokens", "output_tokens", "cost",
                    ])
            self._adaptive_csv_ready = True

        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                sl.task_id, sl.step, sl.tool_called, sl.confidence,
                sl.profile_selected, sl.mask_window, int(sl.pruning_used),
                int(sl.progressive_used), sl.input_tokens, sl.output_tokens,
                f"{sl.cost:.6f}",
            ])

    def _save_params(self, task_ids: list[int]):
        with open(self.output_dir / "params.json", "w") as f:
            json.dump({
                "model": self.config.get_model(),
                "provider": self.config.provider.value,
                "domain": self.domain,
                "temperature": self.temperature,
                "num_tasks": len(task_ids),
                "task_ids": task_ids,
                "configs": list(STATIC_CONFIGS.keys()) + ["adaptive"],
            }, f, indent=2)

    def _save_summary(self, all_results: dict[str, ConfigResult]):
        with open(self.output_dir / "summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "pass_rate", "avg_cost", "avg_input_tokens",
                "cost_of_pass", "num_tasks",
            ])
            for name, cr in all_results.items():
                valid = [r for r in cr.task_results if r["total_steps"] > 0]
                writer.writerow([
                    name, f"{cr.pass_rate:.4f}", f"{cr.avg_cost:.6f}",
                    f"{cr.avg_input_tokens:.0f}", f"{cr.cost_of_pass:.6f}",
                    len(valid),
                ])

    def _print_report(self, all_results: dict[str, ConfigResult]):
        table = Table(
            title="Adaptive vs Static Compression",
            show_header=True, header_style="bold cyan",
        )
        table.add_column("Config", style="bold", min_width=20)
        table.add_column("Pass Rate", justify="center")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Avg Input Tok", justify="right")
        table.add_column("Cost-of-Pass", justify="right")
        table.add_column("Token Savings", justify="right", style="green")

        baseline_cop = all_results.get("baseline", ConfigResult("")).cost_of_pass
        baseline_tokens = all_results.get("baseline", ConfigResult("")).avg_input_tokens

        for name, cr in all_results.items():
            cop = cr.cost_of_pass
            token_savings = ""
            if name != "baseline" and baseline_tokens > 0:
                pct = (1 - cr.avg_input_tokens / baseline_tokens) * 100
                token_savings = f"{pct:+.1f}%"

            table.add_row(
                name,
                f"{cr.pass_rate:.1%}",
                f"${cr.avg_cost:.4f}",
                f"{cr.avg_input_tokens:,.0f}",
                f"${cop:.4f}" if cop < float("inf") else "N/A",
                token_savings,
            )

        console.print()
        console.print(table)
        console.print(f"\n  Results saved to: {self.output_dir}/")

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
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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
from adaptive_compression.ablation_variants import ABLATION_CONFIGS

console = Console()
log = logging.getLogger("adaptive_experiment")

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
        self._log_file = None

    def _setup_logging(self):
        """Set up file + console logging. All output goes to experiment.log."""
        log_path = self.output_dir / "experiment.log"
        self._log_file = log_path

        log.setLevel(logging.DEBUG)
        log.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)

        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    def _log(self, msg: str, level: str = "info"):
        """Log to both file and rich console."""
        getattr(log, level)(msg)

    def run(self):
        return asyncio.run(self._run_async())

    async def _run_async(self):
        if self.domain == "retail":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as _tasks
        else:
            from tau_bench.envs.airline.tasks_test import TASKS as _tasks

        task_ids = self.task_ids
        if task_ids is None:
            total = len(_tasks)
            n = self.num_tasks or total
            task_ids = list(range(min(n, total)))

        model_name = self.config.get_model()

        self._save_params(task_ids)
        self._setup_logging()

        self._log("=" * 70)
        self._log("ADAPTIVE COMPRESSION EXPERIMENT")
        self._log("=" * 70)
        self._log(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Domain:    {self.domain}")
        self._log(f"Tasks:     {len(task_ids)} (IDs: {task_ids[0]}..{task_ids[-1]})")
        self._log(f"Provider:  {self.config.provider.value}")
        self._log(f"Model:     {model_name}")
        self._log(f"User sim:  {USER_SIM_MODEL}")
        self._log(f"Temp:      {self.temperature}")
        self._log(f"Output:    {self.output_dir}")

        configs_to_run = list(STATIC_CONFIGS.keys()) + ["adaptive"]
        if self.only_configs:
            configs_to_run = [c for c in configs_to_run if c in self.only_configs]
        self._log(f"Configs:   {', '.join(configs_to_run)}")
        self._log(f"Log file:  {self._log_file}")
        self._log("")

        console.print(f"\n[bold]Adaptive Compression Experiment[/bold]")
        console.print(f"  Domain: {self.domain}")
        console.print(f"  Tasks: {len(task_ids)}")
        console.print(f"  Provider: {self.config.provider.value}")
        console.print(f"  Model: {model_name}")
        console.print(f"  Configs: {', '.join(configs_to_run)}")
        console.print(f"  Log file: {self._log_file}")
        console.print()

        all_config_results = {}

        # Run each config sequentially (filter by --configs if specified)
        # "adaptive" and "ablation-*" are special — flags=None signals non-static
        all_configs = list(STATIC_CONFIGS.items()) + [("adaptive", None)]
        for abl_name in ABLATION_CONFIGS:
            all_configs.append((abl_name, None))
        if self.only_configs:
            all_configs = [(n, f) for n, f in all_configs if n in self.only_configs]

        experiment_start = time.time()

        for config_name, flags in all_configs:
            config_start = time.time()
            self._log("-" * 60)
            self._log(f"CONFIG: {config_name}")
            self._log("-" * 60)
            console.print(f"\n  [bold]{'=' * 50}[/bold]")
            console.print(f"  [bold]Config: {config_name}[/bold]")
            console.print(f"  [bold]{'=' * 50}[/bold]")

            cr = ConfigResult(config_name=config_name)
            semaphore = asyncio.Semaphore(self.concurrency)
            completed = 0
            passed = 0
            total_tokens_so_far = 0
            total_cost_so_far = 0.0

            async def run_one(tid, _config_name=config_name, _flags=flags):
                async with semaphore:
                    if _config_name == "adaptive":
                        return await asyncio.to_thread(self._run_single_adaptive, tid)
                    elif _config_name in ABLATION_CONFIGS:
                        return await asyncio.to_thread(
                            self._run_single_ablation, tid, _config_name
                        )
                    else:
                        return await asyncio.to_thread(
                            self._run_single_static, tid, _flags, _config_name
                        )

            for tid in task_ids:
                task_start = time.time()
                result = await run_one(tid)

                is_adaptive_type = (config_name == "adaptive" or config_name in ABLATION_CONFIGS)
                if is_adaptive_type and isinstance(result, AdaptiveTaskResult):
                    task_dict = {
                        "task_id": result.task_id,
                        "config": config_name,
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
                task_elapsed = time.time() - task_start
                reward = task_dict["reward"] if is_adaptive_type and isinstance(result, AdaptiveTaskResult) else result.get("reward", 0)
                if reward > 0:
                    passed += 1
                tid_val = result.task_id if isinstance(result, AdaptiveTaskResult) else result.get("task_id", "?")
                tokens = result.total_input_tokens if isinstance(result, AdaptiveTaskResult) else result.get("total_input_tokens", 0)
                cost = result.total_cost if isinstance(result, AdaptiveTaskResult) else result.get("total_cost", 0)
                steps = result.total_steps if isinstance(result, AdaptiveTaskResult) else result.get("total_steps", 0)
                total_tokens_so_far += tokens
                total_cost_so_far += cost

                status_str = "PASS" if reward > 0 else "FAIL"
                status_rich = "[green]PASS[/green]" if reward > 0 else "[red]FAIL[/red]"

                # Adaptive step-level detail in log file
                if config_name == "adaptive" and isinstance(result, AdaptiveTaskResult):
                    for sl in result.step_logs:
                        self._log(
                            f"  step {sl.step:>2}: {sl.tool_called:<25} "
                            f"conf={sl.confidence:.3f}  profile={sl.profile_selected:<14} "
                            f"mask={sl.mask_window}  prog={int(sl.progressive_used)}  prune={int(sl.pruning_used)}  "
                            f"in={sl.input_tokens:>6,}  out={sl.output_tokens:>5,}",
                            level="debug",
                        )
                    stats = result.controller_stats
                    if stats:
                        dist = stats.get("profile_distribution", {})
                        dist_str = " | ".join(f"{k}={v}" for k, v in sorted(dist.items()))
                        self._log(
                            f"  controller: mean_conf={stats.get('mean_confidence', 0):.3f}  "
                            f"profiles: {dist_str}",
                            level="debug",
                        )

                log_line = (
                    f"Task {tid_val:>3}: {status_str}  "
                    f"steps={steps:>2}  tokens={tokens:>7,}  cost=${cost:.4f}  "
                    f"time={task_elapsed:.1f}s  "
                    f"running: {passed}/{completed} = {passed/completed:.1%}"
                )
                self._log(log_line)

                console.print(
                    f"    Task {tid_val:>3}: {status_rich}  "
                    f"steps={steps:>2}  tokens={tokens:>7,}  cost=${cost:.4f}  "
                    f"time={task_elapsed:.1f}s  "
                    f"running: {passed}/{completed} ({passed/completed:.0%})"
                )

            config_elapsed = time.time() - config_start
            avg_tokens = total_tokens_so_far / completed if completed else 0
            self._log("")
            self._log(f"CONFIG SUMMARY: {config_name}")
            self._log(f"  Pass rate:      {cr.pass_rate:.1%} ({passed}/{completed})")
            self._log(f"  Avg tokens:     {cr.avg_input_tokens:,.0f}")
            self._log(f"  Avg cost:       ${cr.avg_cost:.4f}")
            self._log(f"  Cost-of-pass:   ${cr.cost_of_pass:.4f}" if cr.cost_of_pass < float("inf") else "  Cost-of-pass:   N/A")
            self._log(f"  Total cost:     ${total_cost_so_far:.4f}")
            self._log(f"  Elapsed:        {config_elapsed:.0f}s ({config_elapsed/60:.1f}min)")
            self._log("")

            all_config_results[config_name] = cr
            console.print(
                f"  [bold]{config_name} done:[/bold] "
                f"pass={cr.pass_rate:.1%}, "
                f"avg_tokens={cr.avg_input_tokens:,.0f}, "
                f"avg_cost=${cr.avg_cost:.4f}, "
                f"elapsed={config_elapsed:.0f}s"
            )

        total_elapsed = time.time() - experiment_start
        self._save_summary(all_config_results)
        self._print_report(all_config_results)
        self._log_final_report(all_config_results, total_elapsed)

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
                    self._log(f"Task {tid} ({config_name}) retry {attempt}/{max_retries}: {str(e)[:100]}", level="warning")
                    time.sleep(wait)
                    continue
                self._log(f"Task {tid} ({config_name}) FAILED permanently: {str(e)[:200]}", level="error")
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
                    self._log(f"Task {tid} (adaptive) retry {attempt}/{max_retries}: {str(e)[:100]}", level="warning")
                    time.sleep(wait)
                    continue
                self._log(f"Task {tid} (adaptive) FAILED permanently: {str(e)[:200]}", level="error")
                return AdaptiveTaskResult(
                    task_id=tid, reward=0.0, total_steps=0,
                    total_cost=0.0, total_input_tokens=0, total_output_tokens=0,
                )

    def _run_single_ablation(
        self, tid: int, ablation_name: str, max_retries: int = 3
    ) -> AdaptiveTaskResult:
        from tau_bench.envs import get_env

        abl_config = ABLATION_CONFIGS[ablation_name]
        agent_class = abl_config["agent_class"]
        agent_kwargs = abl_config["agent_kwargs"]

        for attempt in range(1, max_retries + 1):
            try:
                env = get_env(
                    env_name=self.domain,
                    user_strategy="llm",
                    user_model=USER_SIM_MODEL,
                    user_provider=USER_SIM_PROVIDER,
                    task_split="test",
                )
                agent = agent_class(
                    config=self.config,
                    tools_info=env.tools_info,
                    wiki=env.wiki,
                    temperature=self.temperature,
                    **agent_kwargs,
                )
                return agent.solve(env, task_index=tid)
            except Exception as e:
                is_transient = any(kw in str(e).lower() for kw in [
                    "429", "rate", "timeout", "connection", "model busy", "retry",
                    "overloaded", "502", "503",
                ])
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt * 5
                    self._log(f"Task {tid} ({ablation_name}) retry {attempt}/{max_retries}: {str(e)[:100]}", level="warning")
                    time.sleep(wait)
                    continue
                self._log(f"Task {tid} ({ablation_name}) FAILED permanently: {str(e)[:200]}", level="error")
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

    def _log_final_report(self, all_results: dict[str, ConfigResult], total_elapsed: float):
        """Write a comprehensive final report to the log file."""
        self._log("=" * 70)
        self._log("FINAL REPORT")
        self._log("=" * 70)
        self._log(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        self._log("")

        baseline_tokens = all_results.get("baseline", ConfigResult("")).avg_input_tokens
        baseline_cop = all_results.get("baseline", ConfigResult("")).cost_of_pass

        header = f"{'Config':<22} {'Pass%':>7} {'AvgTok':>10} {'AvgCost':>10} {'CoP':>10} {'TokSave':>9}"
        self._log(header)
        self._log("-" * len(header))

        for name, cr in all_results.items():
            cop = cr.cost_of_pass
            cop_str = f"${cop:.4f}" if cop < float("inf") else "N/A"
            tok_save = ""
            if name != "baseline" and baseline_tokens > 0:
                pct = (1 - cr.avg_input_tokens / baseline_tokens) * 100
                tok_save = f"{pct:+.1f}%"

            self._log(
                f"{name:<22} {cr.pass_rate:>6.1%} {cr.avg_input_tokens:>10,.0f} "
                f"${cr.avg_cost:>9.4f} {cop_str:>10} {tok_save:>9}"
            )

        self._log("")
        self._log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 70)

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

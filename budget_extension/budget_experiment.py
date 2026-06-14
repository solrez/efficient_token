"""Budget-constrained experiment runner.

Compares baseline (no compression, hard budget cutoff) vs budget-aware
adaptive compression across multiple token budget levels.

Budget levels are derived from baseline token usage percentiles.

Usage:
    python -m budget_extension experiment \
        --domain retail --num-tasks 50 --provider deepseek \
        --model deepseek-v4-flash --output results/budget_experiment/ \
        --budgets auto --baseline-dir results/adaptive_experiment_retail
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

import numpy as np

import litellm
litellm.suppress_debug_info = True

from rich.console import Console
from rich.table import Table

from token_efficient_agent.config import (
    AgentConfig, OptimizationFlags, ModelTier, ProviderType,
    INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK,
)
from adaptive_compression.adaptive_agent import AdaptiveTaskResult, AdaptiveStepLog
from adaptive_compression.controller import ThresholdController
from budget_extension.budget_agent import BudgetTauAgent

console = Console()
log = logging.getLogger("budget_experiment")

USER_SIM_MODEL = "openrouter/openai/gpt-4o-mini"
USER_SIM_PROVIDER = "openrouter"


def derive_budgets(baseline_dir: str) -> dict[str, int]:
    """Derive budget levels from baseline per_task.csv token usage."""
    per_task = Path(baseline_dir) / "per_task.csv"
    if not per_task.exists():
        raise FileNotFoundError(f"No per_task.csv in {baseline_dir}")

    tokens = []
    with open(per_task) as f:
        for row in csv.DictReader(f):
            if row["config"] == "baseline" and int(row["total_steps"]) > 0:
                tokens.append(int(row["total_input_tokens"]))

    if not tokens:
        raise ValueError("No baseline results found")

    arr = np.array(tokens)
    budgets = {
        "p10": int(np.percentile(arr, 10)),
        "p25": int(np.percentile(arr, 25)),
        "p50": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "unlimited": 0,
    }
    return budgets


@dataclass
class BudgetConfigResult:
    config_name: str
    budget_level: str
    budget_tokens: int
    task_results: list[dict] = field(default_factory=list)
    budget_exceeded_count: int = 0

    @property
    def pass_rate(self) -> float:
        valid = [r for r in self.task_results if r["total_steps"] > 0]
        if not valid:
            return 0.0
        return sum(r["reward"] for r in valid) / len(valid)

    @property
    def avg_input_tokens(self) -> float:
        valid = [r for r in self.task_results if r["total_steps"] > 0]
        if not valid:
            return 0.0
        return sum(r["total_input_tokens"] for r in valid) / len(valid)

    @property
    def avg_cost(self) -> float:
        valid = [r for r in self.task_results if r["total_steps"] > 0]
        if not valid:
            return 0.0
        return sum(r["total_cost"] for r in valid) / len(valid)


def _compute_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    rate_in = INPUT_COST_PER_MTOK.get(model_name, 0)
    rate_out = OUTPUT_COST_PER_MTOK.get(model_name, 0)
    return (input_tokens / 1_000_000 * rate_in) + (output_tokens / 1_000_000 * rate_out)


class BudgetExperiment:
    """Run budget-constrained comparison experiment."""

    def __init__(
        self,
        config: AgentConfig,
        domain: str = "retail",
        task_ids: list[int] | None = None,
        num_tasks: int | None = None,
        temperature: float = 0.0,
        concurrency: int = 1,
        output_dir: str = "results/budget_experiment",
        budgets: dict[str, int] | None = None,
        baseline_dir: str | None = None,
    ):
        self.config = config
        self.domain = domain
        self.task_ids = task_ids
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._per_task_csv_ready = False
        self._steps_csv_ready = False

        if budgets:
            self.budgets = budgets
        elif baseline_dir:
            self.budgets = derive_budgets(baseline_dir)
        else:
            raise ValueError("Provide --budgets or --baseline-dir")

    def _setup_logging(self):
        log_path = self.output_dir / "experiment.log"
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

        self._setup_logging()

        self._log("=" * 70)
        self._log("BUDGET-CONSTRAINED EXPERIMENT")
        self._log("=" * 70)
        self._log(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Domain:    {self.domain}")
        self._log(f"Tasks:     {len(task_ids)}")
        self._log(f"Provider:  {self.config.provider.value}")
        self._log(f"Model:     {model_name}")
        self._log(f"User sim:  {USER_SIM_MODEL}")
        self._log(f"Budgets:   {json.dumps(self.budgets)}")
        self._log("")

        self._save_params(task_ids)

        all_results = {}
        experiment_start = time.time()

        configs = ["baseline", "budget_aware"]

        for budget_level, budget_tokens in sorted(
            self.budgets.items(), key=lambda x: x[1] if x[1] > 0 else float("inf")
        ):
            for config_name in configs:
                run_key = f"{config_name}@{budget_level}"
                config_start = time.time()
                self._log("-" * 60)
                self._log(f"CONFIG: {run_key} (budget={budget_tokens:,} tokens)")
                self._log("-" * 60)

                cr = BudgetConfigResult(
                    config_name=config_name,
                    budget_level=budget_level,
                    budget_tokens=budget_tokens,
                )
                completed = 0
                passed = 0

                for tid in task_ids:
                    task_start = time.time()

                    if config_name == "baseline":
                        result = await asyncio.to_thread(
                            self._run_baseline, tid, budget_tokens
                        )
                    else:
                        result = await asyncio.to_thread(
                            self._run_budget_aware, tid, budget_tokens
                        )

                    task_dict = self._result_to_dict(result, config_name, budget_level, budget_tokens)
                    cr.task_results.append(task_dict)
                    if task_dict.get("budget_exceeded"):
                        cr.budget_exceeded_count += 1

                    self._append_per_task(task_dict, model_name)
                    if isinstance(result, AdaptiveTaskResult):
                        for sl in result.step_logs:
                            self._append_step(sl, budget_level)

                    completed += 1
                    task_elapsed = time.time() - task_start
                    reward = task_dict["reward"]
                    if reward > 0:
                        passed += 1

                    status = "PASS" if reward > 0 else "FAIL"
                    exceeded = " BUDGET_HIT" if task_dict.get("budget_exceeded") else ""
                    self._log(
                        f"Task {tid:>3}: {status}  "
                        f"steps={task_dict['total_steps']:>2}  "
                        f"tokens={task_dict['total_input_tokens']:>7,}  "
                        f"cost=${task_dict['total_cost']:.4f}  "
                        f"time={task_elapsed:.1f}s  "
                        f"running: {passed}/{completed} = {passed/completed:.1%}"
                        f"{exceeded}"
                    )

                config_elapsed = time.time() - config_start
                self._log("")
                self._log(f"SUMMARY: {run_key}")
                self._log(f"  Pass rate:        {cr.pass_rate:.1%} ({passed}/{completed})")
                self._log(f"  Avg tokens:       {cr.avg_input_tokens:,.0f}")
                self._log(f"  Avg cost:         ${cr.avg_cost:.4f}")
                self._log(f"  Budget exceeded:  {cr.budget_exceeded_count}/{completed}")
                self._log(f"  Elapsed:          {config_elapsed:.0f}s")
                self._log("")

                all_results[run_key] = cr

        total_elapsed = time.time() - experiment_start
        self._save_summary(all_results)
        self._print_report(all_results)
        self._log_final_report(all_results, total_elapsed)

        return all_results

    def _run_baseline(self, tid: int, budget_tokens: int, max_retries: int = 3) -> AdaptiveTaskResult:
        """Baseline: no compression, hard budget cutoff."""
        from tau_bench.envs import get_env
        from adaptive_compression.adaptive_agent import AdaptiveTauAgent
        from adaptive_compression.controller import ThresholdController

        for attempt in range(1, max_retries + 1):
            try:
                env = get_env(
                    env_name=self.domain,
                    user_strategy="llm",
                    user_model=USER_SIM_MODEL,
                    user_provider=USER_SIM_PROVIDER,
                    task_split="test",
                )
                if budget_tokens > 0:
                    from budget_extension.budget_controller import BudgetAwareController

                    class PreserveOnlyController(BudgetAwareController):
                        """Always preserve — no compression, just budget tracking."""
                        def select_profile(self, confidence):
                            from adaptive_compression.controller import ProfileName, PROFILES
                            self.state.confidence_history.append(confidence)
                            self.state.step_count += 1
                            self.state.profile_history.append(ProfileName.PRESERVE)
                            return PROFILES[ProfileName.PRESERVE]

                    controller = PreserveOnlyController(token_budget=budget_tokens)
                    agent = BudgetTauAgent.__new__(BudgetTauAgent)
                    AdaptiveTauAgent.__init__(
                        agent,
                        config=self.config,
                        tools_info=env.tools_info,
                        wiki=env.wiki,
                        temperature=self.temperature,
                        controller=controller,
                    )
                    agent.token_budget = budget_tokens
                else:
                    agent = AdaptiveTauAgent(
                        config=self.config,
                        tools_info=env.tools_info,
                        wiki=env.wiki,
                        temperature=self.temperature,
                        controller=ThresholdController(),
                    )
                    agent.controller = _NoCompressionController()

                return agent.solve(env, task_index=tid)
            except Exception as e:
                is_transient = any(kw in str(e).lower() for kw in [
                    "429", "rate", "timeout", "connection", "model busy",
                    "retry", "overloaded", "502", "503",
                ])
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt * 5
                    self._log(f"Task {tid} (baseline) retry {attempt}: {str(e)[:100]}", level="warning")
                    time.sleep(wait)
                    continue
                self._log(f"Task {tid} (baseline) FAILED: {str(e)[:200]}", level="error")
                return AdaptiveTaskResult(
                    task_id=tid, reward=0.0, total_steps=0,
                    total_cost=0.0, total_input_tokens=0, total_output_tokens=0,
                )

    def _run_budget_aware(self, tid: int, budget_tokens: int, max_retries: int = 3) -> AdaptiveTaskResult:
        """Budget-aware adaptive compression."""
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
                effective_budget = budget_tokens if budget_tokens > 0 else 10_000_000
                agent = BudgetTauAgent(
                    config=self.config,
                    tools_info=env.tools_info,
                    wiki=env.wiki,
                    token_budget=effective_budget,
                    temperature=self.temperature,
                )
                return agent.solve(env, task_index=tid)
            except Exception as e:
                is_transient = any(kw in str(e).lower() for kw in [
                    "429", "rate", "timeout", "connection", "model busy",
                    "retry", "overloaded", "502", "503",
                ])
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt * 5
                    self._log(f"Task {tid} (budget_aware) retry {attempt}: {str(e)[:100]}", level="warning")
                    time.sleep(wait)
                    continue
                self._log(f"Task {tid} (budget_aware) FAILED: {str(e)[:200]}", level="error")
                return AdaptiveTaskResult(
                    task_id=tid, reward=0.0, total_steps=0,
                    total_cost=0.0, total_input_tokens=0, total_output_tokens=0,
                )

    def _result_to_dict(self, result, config_name, budget_level, budget_tokens):
        model_name = self.config.get_model()
        cost = result.total_cost
        if not cost:
            cost = _compute_cost(result.total_input_tokens, result.total_output_tokens, model_name)
        return {
            "config": config_name,
            "budget_level": budget_level,
            "budget_tokens": budget_tokens,
            "task_id": result.task_id,
            "reward": result.reward,
            "total_cost": cost,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "total_steps": result.total_steps,
            "budget_exceeded": result.controller_stats.get("budget_exceeded", False),
        }

    # --- CSV writing ---

    def _append_per_task(self, result: dict, model_name: str):
        path = self.output_dir / "per_task.csv"
        if not self._per_task_csv_ready:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "config", "budget_level", "budget_tokens", "task_id",
                        "reward", "total_cost", "total_input_tokens",
                        "total_output_tokens", "total_steps", "budget_exceeded",
                    ])
            self._per_task_csv_ready = True

        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                result["config"], result["budget_level"], result["budget_tokens"],
                result["task_id"], result["reward"], f"{result['total_cost']:.6f}",
                result["total_input_tokens"], result["total_output_tokens"],
                result["total_steps"], int(result.get("budget_exceeded", False)),
            ])

    def _append_step(self, sl: AdaptiveStepLog, budget_level: str):
        path = self.output_dir / "budget_steps.csv"
        if not self._steps_csv_ready:
            if not path.exists():
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "budget_level", "task_id", "step", "tool_called",
                        "confidence", "profile_selected", "mask_window",
                        "pruning_used", "progressive_used",
                        "input_tokens", "output_tokens", "cost",
                    ])
            self._steps_csv_ready = True

        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                budget_level, sl.task_id, sl.step, sl.tool_called,
                sl.confidence, sl.profile_selected, sl.mask_window,
                int(sl.pruning_used), int(sl.progressive_used),
                sl.input_tokens, sl.output_tokens, f"{sl.cost:.6f}",
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
                "budgets": self.budgets,
            }, f, indent=2)

    def _save_summary(self, all_results: dict[str, BudgetConfigResult]):
        with open(self.output_dir / "summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "budget_level", "budget_tokens", "pass_rate",
                "avg_input_tokens", "avg_cost", "budget_exceeded_count", "num_tasks",
            ])
            for key, cr in all_results.items():
                valid = [r for r in cr.task_results if r["total_steps"] > 0]
                writer.writerow([
                    cr.config_name, cr.budget_level, cr.budget_tokens,
                    f"{cr.pass_rate:.4f}", f"{cr.avg_input_tokens:.0f}",
                    f"{cr.avg_cost:.6f}", cr.budget_exceeded_count, len(valid),
                ])

    def _print_report(self, all_results: dict[str, BudgetConfigResult]):
        table = Table(title="Budget-Constrained Experiment", show_header=True, header_style="bold cyan")
        table.add_column("Config", style="bold", min_width=25)
        table.add_column("Budget", justify="right")
        table.add_column("Pass Rate", justify="center")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Budget Hit", justify="center")

        for key, cr in all_results.items():
            valid = len([r for r in cr.task_results if r["total_steps"] > 0])
            table.add_row(
                key,
                f"{cr.budget_tokens:,}" if cr.budget_tokens > 0 else "unlimited",
                f"{cr.pass_rate:.1%}",
                f"{cr.avg_input_tokens:,.0f}",
                f"${cr.avg_cost:.4f}",
                f"{cr.budget_exceeded_count}/{valid}",
            )

        console.print()
        console.print(table)

    def _log_final_report(self, all_results: dict[str, BudgetConfigResult], total_elapsed: float):
        self._log("=" * 70)
        self._log("FINAL REPORT")
        self._log("=" * 70)
        self._log(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        self._log("")

        header = f"{'Config':<30} {'Budget':>10} {'Pass%':>7} {'AvgTok':>10} {'AvgCost':>10} {'BudgetHit':>10}"
        self._log(header)
        self._log("-" * len(header))

        for key, cr in all_results.items():
            valid = len([r for r in cr.task_results if r["total_steps"] > 0])
            budget_str = f"{cr.budget_tokens:,}" if cr.budget_tokens > 0 else "unlimited"
            self._log(
                f"{key:<30} {budget_str:>10} {cr.pass_rate:>6.1%} "
                f"{cr.avg_input_tokens:>10,.0f} ${cr.avg_cost:>9.4f} "
                f"{cr.budget_exceeded_count:>5}/{valid}"
            )

        self._log("")
        self._log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 70)


class _NoCompressionController(ThresholdController):
    """Always returns preserve profile — used for unlimited baseline."""
    def select_profile(self, confidence):
        from adaptive_compression.controller import ProfileName, PROFILES
        self.state.confidence_history.append(confidence)
        self.state.step_count += 1
        self.state.profile_history.append(ProfileName.PRESERVE)
        return PROFILES[ProfileName.PRESERVE]

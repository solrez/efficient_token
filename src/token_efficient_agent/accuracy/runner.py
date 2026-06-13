"""Run BFCL accuracy benchmark with concurrency, checkpointing, retries, and cost tracking."""
from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.table import Table

from token_efficient_agent.config import (
    AgentConfig,
    OptimizationFlags,
    ModelTier,
    INPUT_COST_PER_MTOK,
    OUTPUT_COST_PER_MTOK,
)
from token_efficient_agent.provider import create_chat_model
from token_efficient_agent.accuracy.bfcl_loader import BFCLTestCase, load_all, load_category, load_stratified_sample
from token_efficient_agent.accuracy.evaluator import evaluate_tool_calls, EvalResult

console = Console()

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the provided functions. "
    "Call the appropriate function to answer the user's question. "
    "If no function is relevant, respond without calling any function."
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
DEFAULT_CONCURRENCY = 5


@dataclass
class CategoryResult:
    category: str
    optimization_label: str
    run_id: int = 0
    total: int = 0
    passed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    case_results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def total_cost(self) -> float:
        rate_in = 3.0
        rate_out = 15.0
        return (self.total_input_tokens / 1_000_000 * rate_in
                + self.total_output_tokens / 1_000_000 * rate_out)


@dataclass
class AccuracyBenchmarkResults:
    results: list[CategoryResult] = field(default_factory=list)

    def get(self, category: str, opt_label: str) -> CategoryResult | None:
        matching = [r for r in self.results if r.category == category and r.optimization_label == opt_label]
        if not matching:
            return None
        if len(matching) == 1:
            return matching[0]
        merged = CategoryResult(category=category, optimization_label=opt_label)
        for r in matching:
            merged.total += r.total
            merged.passed += r.passed
            merged.failed += r.failed
            merged.total_input_tokens += r.total_input_tokens
            merged.total_output_tokens += r.total_output_tokens
            merged.case_results.extend(r.case_results)
        n = len(matching)
        merged.total //= n
        merged.passed //= n
        merged.failed //= n
        merged.total_input_tokens //= n
        merged.total_output_tokens //= n
        return merged

    def get_all_runs(self, category: str, opt_label: str) -> list[CategoryResult]:
        return [r for r in self.results if r.category == category and r.optimization_label == opt_label]

    @property
    def categories(self) -> list[str]:
        return sorted(set(r.category for r in self.results))

    @property
    def optimization_labels(self) -> list[str]:
        seen = []
        for r in self.results:
            if r.optimization_label not in seen:
                seen.append(r.optimization_label)
        return seen


class CostTracker:
    """Track running cost across the experiment."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    @property
    def total_cost(self) -> float:
        rate_in = INPUT_COST_PER_MTOK.get(self.model_name, 1.0)
        rate_out = OUTPUT_COST_PER_MTOK.get(self.model_name, 5.0)
        return (self.total_input_tokens / 1_000_000 * rate_in
                + self.total_output_tokens / 1_000_000 * rate_out)

    def summary(self) -> str:
        return (f"${self.total_cost:.4f} spent "
                f"({self.total_input_tokens:,} in / {self.total_output_tokens:,} out / "
                f"{self.total_calls} calls)")


class CheckpointManager:
    """Save and load experiment progress for resume capability."""

    def __init__(self, checkpoint_path: str | None):
        self.path = checkpoint_path
        self._completed: set[str] = set()
        if self.path and os.path.exists(self.path):
            self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = self._key(row["category"], row["optimization"], int(row["run_id"]))
                    self._completed.add(key)
            console.print(f"  [dim]Resumed from checkpoint: {len(self._completed)} configs already done[/dim]")
        except Exception:
            pass

    def _key(self, category: str, opt_label: str, run_id: int) -> str:
        return f"{category}|{opt_label}|{run_id}"

    def is_done(self, category: str, opt_label: str, run_id: int) -> bool:
        return self._key(category, opt_label, run_id) in self._completed

    def mark_done(self, result: CategoryResult):
        key = self._key(result.category, result.optimization_label, result.run_id)
        self._completed.add(key)
        if self.path:
            self._append(result)

    def _append(self, result: CategoryResult):
        exists = os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow([
                    "category", "optimization", "run_id", "total", "passed", "failed",
                    "accuracy", "input_tokens", "output_tokens", "cost_usd",
                ])
            writer.writerow([
                result.category, result.optimization_label, result.run_id,
                result.total, result.passed, result.failed,
                f"{result.accuracy:.4f}", result.total_input_tokens, result.total_output_tokens,
                f"{result.total_cost:.6f}",
            ])


class AccuracyBenchmarkRunner:
    def __init__(
        self,
        config: AgentConfig,
        categories: list[str] | None = None,
        profiles: list[OptimizationFlags] | None = None,
        limit_per_category: int | None = None,
        num_runs: int = 1,
        temperature: float = 0.0,
        concurrency: int = DEFAULT_CONCURRENCY,
        checkpoint_path: str | None = None,
    ) -> None:
        self.config = config
        self.categories = categories or ["simple", "multiple", "parallel", "irrelevance"]
        self.profiles = profiles or _default_profiles()
        self.limit = limit_per_category
        self.num_runs = num_runs
        self.temperature = temperature
        self.concurrency = concurrency
        self.all_results = AccuracyBenchmarkResults()
        self.cost_tracker = CostTracker(config.get_model())
        self.checkpoint = CheckpointManager(checkpoint_path)
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run_all(self) -> AccuracyBenchmarkResults:
        for category in self.categories:
            if self.limit:
                cases = load_stratified_sample(
                    categories=[category],
                    n_per_category=self.limit,
                    seed=42,
                )
            else:
                cases = load_category(category)
            console.print(f"\n[bold]Category: {category}[/bold] ({len(cases)} cases)")

            for profile in self.profiles:
                label = profile.label()

                for run_id in range(1, self.num_runs + 1):
                    if self.checkpoint.is_done(category, label, run_id):
                        console.print(f"  {label} run {run_id}/{self.num_runs} [dim]skipped (checkpoint)[/dim]")
                        continue

                    run_tag = f" run {run_id}/{self.num_runs}" if self.num_runs > 1 else ""
                    console.print(f"  {label}{run_tag}...", end=" ")

                    cat_result = await self._run_config(cases, category, label, run_id, profile)
                    self.all_results.results.append(cat_result)
                    self.checkpoint.mark_done(cat_result)

                    console.print(
                        f"[green]{cat_result.accuracy:.1%}[/green] "
                        f"({cat_result.passed}/{cat_result.total}) "
                        f"| {cat_result.total_input_tokens:,} tok "
                        f"| ${cat_result.total_cost:.4f} "
                        f"| [dim]{self.cost_tracker.summary()}[/dim]"
                    )

        return self.all_results

    async def _run_config(
        self,
        cases: list[BFCLTestCase],
        category: str,
        label: str,
        run_id: int,
        flags: OptimizationFlags,
    ) -> CategoryResult:
        cat_result = CategoryResult(
            category=category,
            optimization_label=label,
            run_id=run_id,
        )

        tasks = [self._run_single_with_semaphore(case, flags) for case in cases]
        results = await asyncio.gather(*tasks)

        for result in results:
            cat_result.total += 1
            cat_result.total_input_tokens += result.input_tokens
            cat_result.total_output_tokens += result.output_tokens
            cat_result.case_results.append(result)
            self.cost_tracker.add(result.input_tokens, result.output_tokens)
            if result.passed:
                cat_result.passed += 1
            else:
                cat_result.failed += 1

        return cat_result

    async def _run_single_with_semaphore(
        self,
        case: BFCLTestCase,
        flags: OptimizationFlags,
    ) -> EvalResult:
        async with self._semaphore:
            return await self._run_single_with_retry(case, flags)

    async def _run_single_with_retry(
        self,
        case: BFCLTestCase,
        flags: OptimizationFlags,
    ) -> EvalResult:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return await self._run_single(case, flags)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_retryable = any(s in err_str for s in ["rate_limit", "429", "500", "502", "503", "timeout", "connection"])
                if not is_retryable:
                    break
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

        return EvalResult(
            test_id=case.id,
            category=case.category,
            passed=False,
            error=f"Failed after {MAX_RETRIES} retries: {last_error}",
        )

    async def _run_single(
        self,
        case: BFCLTestCase,
        flags: OptimizationFlags,
    ) -> EvalResult:
        tools = case.langchain_tools

        if flags.tool_pruning:
            tools = [_prune_tool(t) for t in tools]

        if flags.progressive_disclosure:
            tools = [_progressive_tool(t) for t in tools]

        tier = ModelTier.STRONG
        if flags.model_routing and case.category == "simple":
            tier = ModelTier.WEAK

        llm = create_chat_model(self.config, tier=tier, temperature=self.temperature)
        llm = llm.bind_tools(tools)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in case.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))

        response = await llm.ainvoke(messages)

        return evaluate_tool_calls(
            response=response,
            ground_truth=case.ground_truth,
            test_id=case.id,
            category=case.category,
        )


def _prune_tool(tool) -> object:
    clone = tool.model_copy()
    desc = clone.description
    if len(desc) > 80:
        clone.description = desc.split(".")[0] + "."
    return clone


def _progressive_tool(tool) -> object:
    clone = tool.model_copy()
    clone.description = clone.description.split(".")[0] + "."
    return clone


def _default_profiles() -> list[OptimizationFlags]:
    return [
        OptimizationFlags(),
        OptimizationFlags(tool_pruning=True),
        OptimizationFlags(progressive_disclosure=True),
        OptimizationFlags(tool_pruning=True, progressive_disclosure=True),
        OptimizationFlags(
            prompt_caching=True,
            tool_pruning=True,
            progressive_disclosure=True,
            observation_masking=True,
            model_routing=True,
        ),
    ]


def print_accuracy_report(results: AccuracyBenchmarkResults) -> None:
    categories = results.categories
    opt_labels = results.optimization_labels

    table = Table(title="BFCL Accuracy vs Token Efficiency", show_header=True, header_style="bold cyan")
    table.add_column("Optimization", style="bold", min_width=35)

    for cat in categories:
        table.add_column(f"{cat}", justify="center")
    table.add_column("Avg Acc", justify="center", style="bold")
    table.add_column("Total Tok", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Savings", justify="right", style="green")

    baseline_cost = None
    for label in opt_labels:
        row = [label]
        accuracies = []
        total_input = 0
        total_cost = 0.0

        for cat in categories:
            runs = results.get_all_runs(cat, label)
            if runs:
                accs = [r.accuracy for r in runs]
                mean_acc = sum(accs) / len(accs)
                if len(accs) > 1:
                    std = (sum((a - mean_acc) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
                    row.append(f"{mean_acc:.1%}±{std:.1%}")
                else:
                    row.append(f"{mean_acc:.1%}")
                accuracies.append(mean_acc)
                total_input += sum(r.total_input_tokens for r in runs) // len(runs)
                total_cost += sum(r.total_cost for r in runs) / len(runs)
            else:
                row.append("-")

        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
        row.append(f"{avg_acc:.1%}")
        row.append(f"{total_input:,}")
        row.append(f"${total_cost:.4f}")

        if label == "baseline":
            baseline_cost = total_cost
            row.append("-")
        elif baseline_cost and baseline_cost > 0:
            savings = (1 - total_cost / baseline_cost) * 100
            row.append(f"{savings:+.1f}%")
        else:
            row.append("-")

        table.add_row(*row)

    console.print()
    console.print(table)


def export_accuracy_csv(results: AccuracyBenchmarkResults, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "optimization", "run_id", "total", "passed", "failed",
            "accuracy", "input_tokens", "output_tokens", "cost_usd",
        ])
        for r in results.results:
            writer.writerow([
                r.category, r.optimization_label, r.run_id,
                r.total, r.passed, r.failed,
                f"{r.accuracy:.4f}", r.total_input_tokens, r.total_output_tokens,
                f"{r.total_cost:.6f}",
            ])


def export_per_case_csv(results: AccuracyBenchmarkResults, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "optimization", "run_id", "test_id",
            "passed", "input_tokens", "output_tokens", "error",
        ])
        for cat_result in results.results:
            for case in cat_result.case_results:
                writer.writerow([
                    cat_result.category, cat_result.optimization_label,
                    cat_result.run_id, case.test_id,
                    int(case.passed), case.input_tokens, case.output_tokens,
                    case.error,
                ])

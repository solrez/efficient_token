from __future__ import annotations

import csv
import io

from rich.console import Console
from rich.table import Table

from token_efficient_agent.benchmark.runner import BenchmarkResults
from token_efficient_agent.tracing import RunMetrics


class BenchmarkReporter:
    def __init__(self, results: BenchmarkResults) -> None:
        self.results = results
        self.console = Console()

    def print_comparison_table(self) -> None:
        scenarios = sorted(set(r.scenario for r in self.results.runs))

        for scenario in scenarios:
            runs = self.results.get_runs_for_scenario(scenario)
            if not runs:
                continue

            table = Table(
                title=f"Scenario: {scenario}",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Optimizations", style="bold", min_width=30)
            table.add_column("Input Tok", justify="right")
            table.add_column("Output Tok", justify="right")
            table.add_column("Cache Read", justify="right")
            table.add_column("Cost ($)", justify="right")
            table.add_column("LLM Calls", justify="right")
            table.add_column("Latency (s)", justify="right")
            table.add_column("Savings", justify="right", style="green")

            baseline_cost = None
            for run in runs:
                if run.optimization_label == "baseline":
                    baseline_cost = run.total_cost
                    break

            for run in runs:
                savings = ""
                if baseline_cost and baseline_cost > 0 and run.optimization_label != "baseline":
                    pct = (1 - run.total_cost / baseline_cost) * 100
                    savings = f"{pct:+.1f}%"

                table.add_row(
                    run.optimization_label,
                    f"{run.total_input_tokens:,}",
                    f"{run.total_output_tokens:,}",
                    f"{run.total_cache_read_tokens:,}",
                    f"${run.total_cost:.4f}",
                    str(run.num_llm_calls),
                    f"{run.latency_ms / 1000:.1f}",
                    savings,
                )

            self.console.print()
            self.console.print(table)

    def print_per_optimization_impact(self) -> None:
        table = Table(
            title="Per-Optimization Marginal Impact (averaged across scenarios)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Optimization", style="bold")
        table.add_column("Avg Input Token Change", justify="right")
        table.add_column("Avg Cost Change", justify="right")
        table.add_column("Avg LLM Call Change", justify="right")

        opt_names = ["cache", "prune", "mask", "progressive", "route"]

        for opt_name in opt_names:
            opt_runs = [r for r in self.results.runs if r.optimization_label == opt_name]
            baseline_runs = [r for r in self.results.runs if r.optimization_label == "baseline"]

            if not opt_runs or not baseline_runs:
                continue

            scenarios = set(r.scenario for r in opt_runs) & set(r.scenario for r in baseline_runs)
            if not scenarios:
                continue

            token_diffs = []
            cost_diffs = []
            call_diffs = []

            for s in scenarios:
                opt = next((r for r in opt_runs if r.scenario == s), None)
                base = next((r for r in baseline_runs if r.scenario == s), None)
                if opt and base:
                    if base.total_input_tokens > 0:
                        token_diffs.append((opt.total_input_tokens - base.total_input_tokens) / base.total_input_tokens * 100)
                    if base.total_cost > 0:
                        cost_diffs.append((opt.total_cost - base.total_cost) / base.total_cost * 100)
                    call_diffs.append(opt.num_llm_calls - base.num_llm_calls)

            avg_tok = sum(token_diffs) / len(token_diffs) if token_diffs else 0
            avg_cost = sum(cost_diffs) / len(cost_diffs) if cost_diffs else 0
            avg_calls = sum(call_diffs) / len(call_diffs) if call_diffs else 0

            table.add_row(
                opt_name,
                f"{avg_tok:+.1f}%",
                f"{avg_cost:+.1f}%",
                f"{avg_calls:+.1f}",
            )

        self.console.print()
        self.console.print(table)

    def export_csv(self, path: str) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "scenario", "optimizations", "provider",
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "cost_usd", "llm_calls", "latency_ms",
        ])

        for run in self.results.runs:
            writer.writerow([
                run.scenario, run.optimization_label, run.provider,
                run.total_input_tokens, run.total_output_tokens,
                run.total_cache_read_tokens, run.total_cache_write_tokens,
                f"{run.total_cost:.6f}", run.num_llm_calls, f"{run.latency_ms:.0f}",
            ])

        csv_content = output.getvalue()
        with open(path, "w") as f:
            f.write(csv_content)
        return path

"""Statistical analysis for the fractional factorial experiment."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from itertools import combinations

from rich.console import Console
from rich.table import Table

from token_efficient_agent.accuracy.experiment import (
    BFCL_FACTOR_NAMES,
    BFCL_FACTOR_MATRIX,
    BFCL_CONFIGS,
    TAU_FACTOR_NAMES,
    TAU_FACTOR_MATRIX,
    TAU_CONFIGS,
)

console = Console()


@dataclass
class InteractionEffect:
    factor_a: str
    factor_b: str
    effect: float
    p_value: float | None = None

    @property
    def direction(self) -> str:
        if abs(self.effect) < 0.01:
            return "neutral"
        return "synergistic" if self.effect > 0 else "antagonistic"

    @property
    def significant(self) -> bool:
        if self.p_value is None:
            return abs(self.effect) > 0.02
        return self.p_value < 0.05


@dataclass
class MainEffect:
    factor: str
    effect: float
    p_value: float | None = None


@dataclass
class AnalysisResults:
    main_effects: list[MainEffect] = field(default_factory=list)
    interactions: list[InteractionEffect] = field(default_factory=list)
    config_accuracies: dict[str, float] = field(default_factory=dict)
    config_costs: dict[str, float] = field(default_factory=dict)
    config_cost_of_pass: dict[str, float] = field(default_factory=dict)


def analyze_experiment(csv_path: str, category: str | None = None) -> AnalysisResults:
    """Run full factorial analysis on experiment results.

    Args:
        csv_path: Path to the per-run aggregate CSV (from export_accuracy_csv)
        category: Filter to specific BFCL category, or None for aggregate
    """
    rows = _load_csv(csv_path)
    if category:
        rows = [r for r in rows if r["category"] == category]

    config_runs = _group_by_config(rows)

    results = AnalysisResults()

    for label, runs in config_runs.items():
        accs = [float(r["accuracy"]) for r in runs]
        costs = [float(r["cost_usd"]) for r in runs]
        mean_acc = sum(accs) / len(accs) if accs else 0
        mean_cost = sum(costs) / len(costs) if costs else 0
        results.config_accuracies[label] = mean_acc
        results.config_costs[label] = mean_cost
        results.config_cost_of_pass[label] = cost_of_pass(mean_acc, mean_cost)

    config_labels = [cfg.label() for cfg in BFCL_CONFIGS]
    config_acc_vector = []
    for i, label in enumerate(config_labels):
        runs = config_runs.get(label, [])
        if runs:
            config_acc_vector.append(sum(float(r["accuracy"]) for r in runs) / len(runs))
        else:
            config_acc_vector.append(None)

    results.main_effects = _compute_main_effects(config_acc_vector)
    results.interactions = _compute_interaction_effects(config_acc_vector)

    return results


def _compute_main_effects(acc_vector: list[float | None]) -> list[MainEffect]:
    """Compute main effects from the factorial design."""
    effects = []
    for j, factor in enumerate(BFCL_FACTOR_NAMES):
        on_vals = []
        off_vals = []
        for i, acc in enumerate(acc_vector):
            if acc is None or i >= len(BFCL_FACTOR_MATRIX):
                continue
            if BFCL_FACTOR_MATRIX[i][j] == 1:
                on_vals.append(acc)
            else:
                off_vals.append(acc)
        if on_vals and off_vals:
            effect = sum(on_vals) / len(on_vals) - sum(off_vals) / len(off_vals)
            effects.append(MainEffect(factor=factor, effect=effect))
    return effects


def _compute_interaction_effects(acc_vector: list[float | None]) -> list[InteractionEffect]:
    """Compute all pairwise interaction effects.

    IE(A,B) = mean(A=1,B=1) + mean(A=0,B=0) - mean(A=1,B=0) - mean(A=0,B=1)
    divided by 2 (half the difference of differences).
    """
    interactions = []
    for (j1, f1), (j2, f2) in combinations(enumerate(BFCL_FACTOR_NAMES), 2):
        groups = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
        for i, acc in enumerate(acc_vector):
            if acc is None or i >= len(BFCL_FACTOR_MATRIX):
                continue
            key = (BFCL_FACTOR_MATRIX[i][j1], BFCL_FACTOR_MATRIX[i][j2])
            groups[key].append(acc)

        if all(len(v) > 0 for v in groups.values()):
            m00 = sum(groups[(0, 0)]) / len(groups[(0, 0)])
            m01 = sum(groups[(0, 1)]) / len(groups[(0, 1)])
            m10 = sum(groups[(1, 0)]) / len(groups[(1, 0)])
            m11 = sum(groups[(1, 1)]) / len(groups[(1, 1)])
            ie = ((m11 - m10) - (m01 - m00)) / 2
            interactions.append(InteractionEffect(factor_a=f1, factor_b=f2, effect=ie))

    return interactions


def cost_of_pass(accuracy: float, mean_cost: float) -> float:
    """Expected cost to produce one correct answer."""
    if accuracy <= 0:
        return float("inf")
    return mean_cost / accuracy


def bootstrap_ci(
    binary_outcomes: list[bool],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for accuracy.

    Returns (mean, lower, upper).
    """
    import random
    rng = random.Random(seed)
    n = len(binary_outcomes)
    if n == 0:
        return (0.0, 0.0, 0.0)

    means = []
    for _ in range(n_resamples):
        sample = rng.choices(binary_outcomes, k=n)
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lower_idx = int(n_resamples * alpha / 2)
    upper_idx = int(n_resamples * (1 - alpha / 2))
    mean_acc = sum(binary_outcomes) / n

    return (mean_acc, means[lower_idx], means[upper_idx])


def mcnemar_test(
    outcomes_a: list[bool],
    outcomes_b: list[bool],
) -> tuple[float, float]:
    """McNemar's test for paired binary outcomes.

    Returns (chi2_statistic, p_value).
    Both lists must be the same length, aligned by test case.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("Outcome lists must be the same length")

    b_count = 0  # a correct, b wrong
    c_count = 0  # a wrong, b correct
    for a, b in zip(outcomes_a, outcomes_b):
        if a and not b:
            b_count += 1
        elif not a and b:
            c_count += 1

    if b_count + c_count == 0:
        return (0.0, 1.0)

    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)

    p_value = _chi2_sf(chi2, df=1)
    return (chi2, p_value)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function for chi-squared distribution (1 - CDF). Approximation for df=1."""
    if x <= 0:
        return 1.0
    from math import erf, sqrt, exp
    z = sqrt(x)
    return 1 - erf(z / sqrt(2))


def print_analysis_report(results: AnalysisResults) -> None:
    """Print the main effects and interaction heatmap."""
    table = Table(title="Main Effects on Accuracy", show_header=True, header_style="bold cyan")
    table.add_column("Factor", style="bold")
    table.add_column("Effect", justify="right")
    table.add_column("Direction", justify="center")

    for me in sorted(results.main_effects, key=lambda x: abs(x.effect), reverse=True):
        direction = "positive" if me.effect > 0 else "negative" if me.effect < 0 else "neutral"
        style = "green" if me.effect > 0 else "red" if me.effect < -0.01 else ""
        table.add_row(me.factor, f"{me.effect:+.3f}", direction, style=style)

    console.print()
    console.print(table)

    table2 = Table(title="Pairwise Interaction Effects", show_header=True, header_style="bold magenta")
    table2.add_column("Pair", style="bold")
    table2.add_column("IE", justify="right")
    table2.add_column("Type", justify="center")

    for ie in sorted(results.interactions, key=lambda x: abs(x.effect), reverse=True):
        style = "green" if ie.direction == "synergistic" else "red" if ie.direction == "antagonistic" else ""
        sig = " *" if ie.significant else ""
        table2.add_row(
            f"{ie.factor_a} × {ie.factor_b}",
            f"{ie.effect:+.4f}{sig}",
            ie.direction,
            style=style,
        )

    console.print()
    console.print(table2)

    if results.config_cost_of_pass:
        table3 = Table(title="Cost-of-Pass Ranking (lower is better)", show_header=True, header_style="bold yellow")
        table3.add_column("Config", style="bold")
        table3.add_column("Accuracy", justify="right")
        table3.add_column("Cost", justify="right")
        table3.add_column("Cost-of-Pass", justify="right")

        ranked = sorted(results.config_cost_of_pass.items(), key=lambda x: x[1])
        for label, cop in ranked[:10]:
            acc = results.config_accuracies.get(label, 0)
            cost = results.config_costs.get(label, 0)
            table3.add_row(label, f"{acc:.1%}", f"${cost:.4f}", f"${cop:.4f}")

        console.print()
        console.print(table3)


def export_interaction_csv(results: AnalysisResults, path: str) -> None:
    """Export interaction effects for external plotting."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["factor_a", "factor_b", "effect", "direction", "significant"])
        for ie in results.interactions:
            writer.writerow([
                ie.factor_a, ie.factor_b, f"{ie.effect:.6f}",
                ie.direction, int(ie.significant),
            ])


def _load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _group_by_config(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        label = row["optimization"]
        groups.setdefault(label, []).append(row)
    return groups

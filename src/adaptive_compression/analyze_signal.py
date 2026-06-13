"""Analyze uncertainty signal quality from validation run data.

Answers the key Phase 1 question: is the uncertainty signal predictive enough
to drive compression decisions?

Usage:
    python -m adaptive_compression.analyze_signal results/signal_validation/
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class StepRow:
    task_id: int
    step: int
    tool_called: str
    logprob_confidence: float | None
    prompted_confidence: float | None
    hybrid_confidence: float | None
    hybrid_method: str


@dataclass
class TaskRow:
    task_id: int
    reward: float
    total_steps: int


def load_data(results_dir: str) -> tuple[list[StepRow], list[TaskRow]]:
    results_path = Path(results_dir)

    steps = []
    with open(results_path / "step_uncertainty.csv") as f:
        for row in csv.DictReader(f):
            steps.append(StepRow(
                task_id=int(row["task_id"]),
                step=int(row["step"]),
                tool_called=row["tool_called"],
                logprob_confidence=_parse_float(row["logprob_confidence"]),
                prompted_confidence=_parse_float(row["prompted_confidence"]),
                hybrid_confidence=_parse_float(row["hybrid_confidence"]),
                hybrid_method=row["hybrid_method"],
            ))

    tasks = []
    with open(results_path / "task_outcomes.csv") as f:
        for row in csv.DictReader(f):
            tasks.append(TaskRow(
                task_id=int(row["task_id"]),
                reward=float(row["reward"]),
                total_steps=int(row["total_steps"]),
            ))

    return steps, tasks


def analyze(results_dir: str):
    results_path = Path(results_dir)

    # Auto-detect format: BFCL (single CSV) vs Tau-bench (step + task CSVs)
    if (results_path / "bfcl_uncertainty.csv").exists():
        analyze_bfcl(results_dir)
        return

    steps, tasks = load_data(results_dir)
    task_reward = {t.task_id: t.reward for t in tasks}

    console.print(f"\n[bold]Uncertainty Signal Analysis[/bold]")
    console.print(f"  Total steps: {len(steps)}")
    console.print(f"  Total tasks: {len(tasks)}")

    # 1. Coverage
    console.print(f"\n[bold]1. Estimator Coverage[/bold]")
    logprob_available = sum(1 for s in steps if s.logprob_confidence is not None)
    prompted_available = sum(1 for s in steps if s.prompted_confidence is not None)
    console.print(f"  Logprob:  {logprob_available}/{len(steps)} ({logprob_available/len(steps):.0%})")
    console.print(f"  Prompted: {prompted_available}/{len(steps)} ({prompted_available/len(steps):.0%})")

    # 2. Distribution shape
    console.print(f"\n[bold]2. Confidence Distributions[/bold]")
    for name, getter in [
        ("Logprob", lambda s: s.logprob_confidence),
        ("Prompted", lambda s: s.prompted_confidence),
        ("Hybrid", lambda s: s.hybrid_confidence),
    ]:
        scores = [getter(s) for s in steps if getter(s) is not None]
        if not scores:
            console.print(f"  {name}: no data")
            continue
        _print_distribution(name, scores)

    # 3. Correlation with task outcome
    console.print(f"\n[bold]3. Correlation: Mean Confidence vs Task Reward[/bold]")
    for name, getter in [
        ("Logprob", lambda s: s.logprob_confidence),
        ("Prompted", lambda s: s.prompted_confidence),
        ("Hybrid", lambda s: s.hybrid_confidence),
    ]:
        _analyze_correlation(name, steps, task_reward, getter)

    # 4. Per-step analysis: confidence on steps preceding failure
    console.print(f"\n[bold]4. Confidence Before Failure vs Success[/bold]")
    _analyze_failure_steps(steps, task_reward)

    # 5. Confidence by tool type
    console.print(f"\n[bold]5. Confidence by Tool Called[/bold]")
    _analyze_by_tool(steps)

    # 6. Correlation between estimators
    console.print(f"\n[bold]6. Logprob vs Prompted Correlation[/bold]")
    _analyze_estimator_correlation(steps)

    # 7. Exit criterion check
    console.print(f"\n[bold]7. Exit Criterion Check[/bold]")
    _check_exit_criteria(steps, task_reward)


def _print_distribution(name: str, scores: list[float]):
    mean = sum(scores) / len(scores)
    sorted_s = sorted(scores)
    median = sorted_s[len(sorted_s) // 2]
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5

    # Quartile bins
    bins = [0, 0.25, 0.5, 0.75, 1.01]
    labels = ["[0.00-0.25)", "[0.25-0.50)", "[0.50-0.75)", "[0.75-1.00]"]
    counts = [0] * 4
    for s in scores:
        for i in range(4):
            if bins[i] <= s < bins[i + 1]:
                counts[i] += 1
                break

    console.print(f"  {name}: mean={mean:.3f}, median={median:.3f}, std={std:.3f}, n={len(scores)}")
    for label, count in zip(labels, counts):
        bar = "#" * int(count / len(scores) * 40)
        console.print(f"    {label}: {count:4d} ({count/len(scores):5.1%}) {bar}")


def _analyze_correlation(
    name: str,
    steps: list[StepRow],
    task_reward: dict[int, float],
    getter,
):
    # Group steps by task, compute mean confidence per task
    task_confs: dict[int, list[float]] = {}
    for s in steps:
        val = getter(s)
        if val is not None:
            task_confs.setdefault(s.task_id, []).append(val)

    if len(task_confs) < 5:
        console.print(f"  {name}: insufficient data (need >= 5 tasks)")
        return

    xs = []  # mean confidence per task
    ys = []  # reward (0 or 1)
    for tid, confs in task_confs.items():
        if tid in task_reward:
            xs.append(sum(confs) / len(confs))
            ys.append(task_reward[tid])

    # Point-biserial correlation (reward is binary)
    r = _point_biserial(xs, ys)
    # AUC: can mean confidence distinguish pass/fail?
    auc = _auc_score(xs, ys)

    console.print(f"  {name}: r={r:.3f}, AUC={auc:.3f} (n={len(xs)} tasks)")

    # Mean confidence for passing vs failing tasks
    pass_confs = [x for x, y in zip(xs, ys) if y > 0]
    fail_confs = [x for x, y in zip(xs, ys) if y == 0]
    if pass_confs and fail_confs:
        console.print(
            f"    Pass tasks mean: {sum(pass_confs)/len(pass_confs):.3f} (n={len(pass_confs)})"
        )
        console.print(
            f"    Fail tasks mean: {sum(fail_confs)/len(fail_confs):.3f} (n={len(fail_confs)})"
        )


def _analyze_failure_steps(steps: list[StepRow], task_reward: dict[int, float]):
    """Compare confidence on the last 2 steps of passing vs failing tasks."""
    task_steps: dict[int, list[StepRow]] = {}
    for s in steps:
        task_steps.setdefault(s.task_id, []).append(s)

    for name, getter in [
        ("Prompted", lambda s: s.prompted_confidence),
        ("Hybrid", lambda s: s.hybrid_confidence),
    ]:
        pass_last = []
        fail_last = []
        for tid, ss in task_steps.items():
            if tid not in task_reward:
                continue
            ss_sorted = sorted(ss, key=lambda s: s.step)
            last_2 = ss_sorted[-2:] if len(ss_sorted) >= 2 else ss_sorted
            vals = [getter(s) for s in last_2 if getter(s) is not None]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            if task_reward[tid] > 0:
                pass_last.append(avg)
            else:
                fail_last.append(avg)

        if pass_last and fail_last:
            console.print(
                f"  {name} (last 2 steps): "
                f"pass={sum(pass_last)/len(pass_last):.3f}, "
                f"fail={sum(fail_last)/len(fail_last):.3f}, "
                f"delta={sum(pass_last)/len(pass_last) - sum(fail_last)/len(fail_last):.3f}"
            )


def _analyze_by_tool(steps: list[StepRow]):
    """Show mean confidence grouped by tool called."""
    tool_scores: dict[str, list[float]] = {}
    for s in steps:
        if s.hybrid_confidence is not None:
            tool_scores.setdefault(s.tool_called, []).append(s.hybrid_confidence)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Tool", min_width=20)
    table.add_column("Count", justify="right")
    table.add_column("Mean Conf", justify="right")
    table.add_column("Std", justify="right")

    for tool in sorted(tool_scores.keys(), key=lambda t: -len(tool_scores[t])):
        scores = tool_scores[tool]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        table.add_row(tool, str(len(scores)), f"{mean:.3f}", f"{std:.3f}")

    console.print(table)


def _analyze_estimator_correlation(steps: list[StepRow]):
    """Pearson correlation between logprob and prompted confidence."""
    pairs = [
        (s.logprob_confidence, s.prompted_confidence)
        for s in steps
        if s.logprob_confidence is not None and s.prompted_confidence is not None
    ]
    if len(pairs) < 10:
        console.print(f"  Insufficient paired data ({len(pairs)} pairs, need >= 10)")
        return

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    r = _pearson(xs, ys)
    console.print(f"  Pearson r = {r:.3f} (n={len(pairs)} paired steps)")
    console.print(f"  Interpretation: {'strong' if abs(r) > 0.5 else 'moderate' if abs(r) > 0.3 else 'weak'} correlation")


def _check_exit_criteria(steps: list[StepRow], task_reward: dict[int, float]):
    """Check if we meet the Phase 1 exit criterion: r > 0.3 or AUC > 0.6."""
    task_confs: dict[int, list[float]] = {}
    for s in steps:
        if s.hybrid_confidence is not None:
            task_confs.setdefault(s.task_id, []).append(s.hybrid_confidence)

    if len(task_confs) < 5:
        console.print("  [yellow]Insufficient data to evaluate exit criteria[/yellow]")
        return

    xs = []
    ys = []
    for tid, confs in task_confs.items():
        if tid in task_reward:
            xs.append(sum(confs) / len(confs))
            ys.append(task_reward[tid])

    r = abs(_point_biserial(xs, ys))
    auc = _auc_score(xs, ys)

    passed = r > 0.3 or auc > 0.6
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

    console.print(f"  Criterion: |r| > 0.3 OR AUC > 0.6")
    console.print(f"  Result: |r|={r:.3f}, AUC={auc:.3f} → {status}")

    if passed:
        console.print(f"  [green]Signal is predictive enough. Proceed to Phase 2.[/green]")
    else:
        console.print(f"  [yellow]Signal may be too noisy. Consider:[/yellow]")
        console.print(f"    - Trying different prompted assessment phrasings")
        console.print(f"    - Using ensemble disagreement instead")
        console.print(f"    - Falling back to simpler heuristics (step count + error rate)")


def analyze_bfcl(results_dir: str):
    """Analyze BFCL signal validation results (single-turn)."""
    results_path = Path(results_dir)
    records = []
    with open(results_path / "bfcl_uncertainty.csv") as f:
        for row in csv.DictReader(f):
            records.append(row)

    valid = [r for r in records if r.get("hybrid_method", "") != "error" and r.get("error", "") == ""]

    console.print(f"\n[bold]BFCL Uncertainty Signal Analysis[/bold]")
    console.print(f"  Total records: {len(records)} ({len(valid)} valid)")

    # 1. Coverage
    console.print(f"\n[bold]1. Estimator Coverage[/bold]")
    logprob_available = sum(1 for r in valid if _parse_float(r["logprob_confidence"]) is not None)
    prompted_available = sum(1 for r in valid if _parse_float(r["prompted_confidence"]) is not None)
    console.print(f"  Logprob:  {logprob_available}/{len(valid)} ({logprob_available/len(valid):.0%})")
    console.print(f"  Prompted: {prompted_available}/{len(valid)} ({prompted_available/len(valid):.0%})")

    # 2. Distribution
    console.print(f"\n[bold]2. Confidence Distributions[/bold]")
    for name, key in [("Logprob", "logprob_confidence"), ("Prompted", "prompted_confidence"), ("Hybrid", "hybrid_confidence")]:
        scores = [_parse_float(r[key]) for r in valid if _parse_float(r[key]) is not None]
        if scores:
            _print_distribution(name, scores)

    # 3. Confidence for correct vs incorrect
    console.print(f"\n[bold]3. Confidence: Correct vs Incorrect[/bold]")
    for name, key in [("Logprob", "logprob_confidence"), ("Prompted", "prompted_confidence"), ("Hybrid", "hybrid_confidence")]:
        correct_scores = [_parse_float(r[key]) for r in valid if int(r["correct"]) == 1 and _parse_float(r[key]) is not None]
        incorrect_scores = [_parse_float(r[key]) for r in valid if int(r["correct"]) == 0 and _parse_float(r[key]) is not None]
        if correct_scores and incorrect_scores:
            delta = sum(correct_scores)/len(correct_scores) - sum(incorrect_scores)/len(incorrect_scores)
            console.print(
                f"  {name}: correct={sum(correct_scores)/len(correct_scores):.3f} (n={len(correct_scores)}), "
                f"incorrect={sum(incorrect_scores)/len(incorrect_scores):.3f} (n={len(incorrect_scores)}), "
                f"delta={delta:+.3f}"
            )
        else:
            console.print(f"  {name}: insufficient data")

    # 4. Per-category breakdown
    console.print(f"\n[bold]4. Confidence by Category[/bold]")
    categories = sorted(set(r["category"] for r in valid))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Category", min_width=12)
    table.add_column("Count", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Hybrid Conf", justify="right")
    table.add_column("Correct Conf", justify="right")
    table.add_column("Incorrect Conf", justify="right")

    for cat in categories:
        cat_records = [r for r in valid if r["category"] == cat]
        cat_correct = sum(int(r["correct"]) for r in cat_records)
        hybrid_scores = [_parse_float(r["hybrid_confidence"]) for r in cat_records if _parse_float(r["hybrid_confidence"]) is not None]
        correct_h = [_parse_float(r["hybrid_confidence"]) for r in cat_records if int(r["correct"]) == 1 and _parse_float(r["hybrid_confidence"]) is not None]
        incorrect_h = [_parse_float(r["hybrid_confidence"]) for r in cat_records if int(r["correct"]) == 0 and _parse_float(r["hybrid_confidence"]) is not None]

        table.add_row(
            cat,
            str(len(cat_records)),
            f"{cat_correct/len(cat_records):.1%}",
            f"{sum(hybrid_scores)/len(hybrid_scores):.3f}" if hybrid_scores else "-",
            f"{sum(correct_h)/len(correct_h):.3f}" if correct_h else "-",
            f"{sum(incorrect_h)/len(incorrect_h):.3f}" if incorrect_h else "-",
        )
    console.print(table)

    # 5. Logprob vs Prompted correlation
    console.print(f"\n[bold]5. Logprob vs Prompted Correlation[/bold]")
    pairs = [
        (_parse_float(r["logprob_confidence"]), _parse_float(r["prompted_confidence"]))
        for r in valid
        if _parse_float(r["logprob_confidence"]) is not None and _parse_float(r["prompted_confidence"]) is not None
    ]
    if len(pairs) >= 10:
        r_val = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
        console.print(f"  Pearson r = {r_val:.3f} (n={len(pairs)})")
    else:
        console.print(f"  Insufficient paired data ({len(pairs)} pairs)")

    # 6. Exit criterion (point-biserial correlation + AUC)
    console.print(f"\n[bold]6. Exit Criterion Check[/bold]")
    for name, key in [("Logprob", "logprob_confidence"), ("Prompted", "prompted_confidence"), ("Hybrid", "hybrid_confidence")]:
        scores = []
        labels = []
        for r in valid:
            s = _parse_float(r[key])
            if s is not None:
                scores.append(s)
                labels.append(float(int(r["correct"])))
        if len(scores) < 10:
            console.print(f"  {name}: insufficient data")
            continue
        r_val = abs(_point_biserial(scores, labels))
        auc = _auc_score(scores, labels)
        passed = r_val > 0.3 or auc > 0.6
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {name}: |r|={r_val:.3f}, AUC={auc:.3f} → {status}")

    # Overall verdict
    hybrid_scores = []
    hybrid_labels = []
    for r in valid:
        s = _parse_float(r["hybrid_confidence"])
        if s is not None:
            hybrid_scores.append(s)
            hybrid_labels.append(float(int(r["correct"])))
    if len(hybrid_scores) >= 10:
        r_val = abs(_point_biserial(hybrid_scores, hybrid_labels))
        auc = _auc_score(hybrid_scores, hybrid_labels)
        passed = r_val > 0.3 or auc > 0.6
        if passed:
            console.print(f"\n  [green]Signal is predictive enough. Proceed to Phase 2.[/green]")
        else:
            console.print(f"\n  [yellow]Signal may be too noisy. Consider alternative estimators.[/yellow]")


# --- Stats helpers (no numpy dependency) ---


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _point_biserial(xs: list[float], ys: list[float]) -> float:
    """Point-biserial correlation between continuous xs and binary ys."""
    return _pearson(xs, ys)


def _auc_score(scores: list[float], labels: list[float]) -> float:
    """ROC AUC for binary classification. Positive label = 1.0."""
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda p: -p[0])

    tp = 0
    fp = 0
    total_pos = sum(1 for _, y in pairs if y > 0)
    total_neg = len(pairs) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for score, label in pairs:
        if label > 0:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def _parse_float(s: str) -> float | None:
    if not s or s.lower() in ("none", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m adaptive_compression.analyze_signal <results_dir>")
        sys.exit(1)
    analyze(sys.argv[1])

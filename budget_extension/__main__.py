"""CLI entry point for budget-constrained experiments.

Usage:
    python -m budget_extension experiment \
        --domain retail --num-tasks 50 --provider deepseek \
        --model deepseek-v4-flash \
        --baseline-dir results/adaptive_experiment_retail \
        --output results/budget_experiment_retail
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Budget-Constrained Adaptive Compression"
    )
    subparsers = parser.add_subparsers(dest="command")

    exp_parser = subparsers.add_parser(
        "experiment",
        help="Run budget-constrained comparison experiment",
    )
    exp_parser.add_argument("--domain", default="retail", choices=["retail", "airline"])
    exp_parser.add_argument("--num-tasks", type=int, default=50)
    exp_parser.add_argument("--task-ids", type=str, default=None,
                            help="Comma-separated task IDs")
    exp_parser.add_argument("--provider", default="deepseek",
                            choices=["openrouter", "deepinfra", "groq", "deepseek"])
    exp_parser.add_argument("--model", default="deepseek-v4-flash")
    exp_parser.add_argument("--temperature", type=float, default=0.0)
    exp_parser.add_argument("--concurrency", type=int, default=1)
    exp_parser.add_argument("--output", default="results/budget_experiment")
    exp_parser.add_argument("--baseline-dir", type=str, default=None,
                            help="Path to existing baseline results (for auto budget derivation)")
    exp_parser.add_argument("--budgets", type=str, default=None,
                            help="Comma-separated budget levels in tokens (e.g., '30000,60000,100000') or 'auto'")

    derive_parser = subparsers.add_parser(
        "derive-budgets",
        help="Show budget levels derived from baseline results",
    )
    derive_parser.add_argument("baseline_dir", help="Path to baseline results directory")

    args = parser.parse_args()

    if args.command == "experiment":
        _run_experiment(args)
    elif args.command == "derive-budgets":
        _derive_budgets(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_experiment(args):
    from token_efficient_agent.config import AgentConfig, ProviderType
    from budget_extension.budget_experiment import BudgetExperiment, derive_budgets

    provider = ProviderType(args.provider)
    config = AgentConfig(
        provider=provider,
        openrouter_model=args.model,
    )

    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    budgets = None
    if args.budgets and args.budgets != "auto":
        levels = [int(x.strip()) for x in args.budgets.split(",")]
        budgets = {f"b{v}": v for v in levels}
        budgets["unlimited"] = 0
    elif args.baseline_dir:
        budgets = derive_budgets(args.baseline_dir)
        print(f"Derived budgets: {budgets}")

    experiment = BudgetExperiment(
        config=config,
        domain=args.domain,
        task_ids=task_ids,
        num_tasks=args.num_tasks,
        temperature=args.temperature,
        concurrency=args.concurrency,
        output_dir=args.output,
        budgets=budgets,
        baseline_dir=args.baseline_dir if not budgets else None,
    )
    experiment.run()


def _derive_budgets(args):
    from budget_extension.budget_experiment import derive_budgets

    budgets = derive_budgets(args.baseline_dir)
    print("Budget levels derived from baseline token usage:")
    for level, tokens in sorted(budgets.items(), key=lambda x: x[1] if x[1] > 0 else float("inf")):
        label = f"{tokens:,}" if tokens > 0 else "unlimited"
        print(f"  {level}: {label} tokens")


if __name__ == "__main__":
    main()

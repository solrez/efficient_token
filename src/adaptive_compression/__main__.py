"""CLI entry point for the adaptive compression signal validation.

Usage:
    python -m adaptive_compression validate \
        --domain retail --num-tasks 50 --provider openrouter \
        --model deepseek-v4-flash --output results/signal_validation/

    python -m adaptive_compression analyze results/signal_validation/
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Uncertainty-Driven Adaptive Compression — Phase 1"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- validate (Tau-bench) ---
    val_parser = subparsers.add_parser(
        "validate",
        help="Run Tau-bench tasks with uncertainty estimation (no compression)",
    )
    val_parser.add_argument("--domain", default="retail", choices=["retail", "airline"])
    val_parser.add_argument("--num-tasks", type=int, default=50)
    val_parser.add_argument("--task-ids", type=str, default=None,
                            help="Comma-separated task IDs (e.g., 0,1,5,10)")
    val_parser.add_argument("--provider", default="deepinfra",
                            choices=["openrouter", "anthropic", "openai", "deepinfra", "groq", "deepseek"])
    val_parser.add_argument("--model", default="deepseek-v4-flash",
                            help="OpenRouter model preset name")
    val_parser.add_argument("--temperature", type=float, default=0.0)
    val_parser.add_argument("--concurrency", type=int, default=3)
    val_parser.add_argument("--output", default="results/signal_validation",
                            help="Output directory for results")
    val_parser.add_argument("--behavioral", action="store_true", default=False,
                            help="Enable behavioral signals (repetition, respond ratio, stall detection, composite)")

    # --- validate-bfcl ---
    bfcl_parser = subparsers.add_parser(
        "validate-bfcl",
        help="Run BFCL single-turn cases with uncertainty estimation (faster)",
    )
    bfcl_parser.add_argument("--provider", default="deepinfra",
                             choices=["openrouter", "anthropic", "openai", "deepinfra", "groq", "deepseek"])
    bfcl_parser.add_argument("--model", default="deepseek-v4-flash",
                             help="OpenRouter model preset name")
    bfcl_parser.add_argument("--num-cases", type=int, default=100,
                             help="Cases per category (4 categories)")
    bfcl_parser.add_argument("--categories", type=str, default=None,
                             help="Comma-separated categories (simple,multiple,parallel,irrelevance)")
    bfcl_parser.add_argument("--temperature", type=float, default=0.0)
    bfcl_parser.add_argument("--concurrency", type=int, default=5)
    bfcl_parser.add_argument("--output", default="results/signal_validation_bfcl",
                             help="Output directory for results")

    # --- experiment ---
    exp_parser = subparsers.add_parser(
        "experiment",
        help="Run adaptive vs static compression comparison experiment",
    )
    exp_parser.add_argument("--domain", default="retail", choices=["retail", "airline"])
    exp_parser.add_argument("--num-tasks", type=int, default=50)
    exp_parser.add_argument("--task-ids", type=str, default=None,
                            help="Comma-separated task IDs")
    exp_parser.add_argument("--provider", default="deepseek",
                            choices=["openrouter", "anthropic", "openai", "deepinfra", "groq", "deepseek"])
    exp_parser.add_argument("--model", default="deepseek-v4-flash")
    exp_parser.add_argument("--temperature", type=float, default=0.0)
    exp_parser.add_argument("--concurrency", type=int, default=1)
    exp_parser.add_argument("--output", default="results/adaptive_experiment")
    exp_parser.add_argument("--configs", type=str, default=None,
                            help="Comma-separated configs to run (e.g., 'adaptive' or 'baseline,adaptive'). Default: all")

    # --- analyze ---
    ana_parser = subparsers.add_parser(
        "analyze",
        help="Analyze uncertainty signal quality from validation results",
    )
    ana_parser.add_argument("results_dir", help="Path to validation results directory")

    args = parser.parse_args()

    if args.command == "validate":
        _run_validate(args)
    elif args.command == "validate-bfcl":
        _run_validate_bfcl(args)
    elif args.command == "experiment":
        _run_experiment(args)
    elif args.command == "analyze":
        from adaptive_compression.analyze_signal import analyze
        analyze(args.results_dir)
    else:
        parser.print_help()
        sys.exit(1)


def _run_validate(args):
    from token_efficient_agent.config import AgentConfig, ProviderType
    from adaptive_compression.validation_runner import SignalValidationRunner

    provider = ProviderType(args.provider)
    config = AgentConfig(
        provider=provider,
        openrouter_model=args.model,
    )

    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    runner = SignalValidationRunner(
        config=config,
        domain=args.domain,
        task_ids=task_ids,
        num_tasks=args.num_tasks,
        temperature=args.temperature,
        concurrency=args.concurrency,
        output_dir=args.output,
        behavioral=getattr(args, "behavioral", False),
    )
    runner.run()


def _run_validate_bfcl(args):
    from token_efficient_agent.config import AgentConfig, ProviderType
    from adaptive_compression.bfcl_validation import BFCLSignalValidation

    provider = ProviderType(args.provider)
    config = AgentConfig(
        provider=provider,
        openrouter_model=args.model if provider == ProviderType.OPENROUTER else "",
    )

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    runner = BFCLSignalValidation(
        config=config,
        categories=categories,
        num_per_category=args.num_cases,
        temperature=args.temperature,
        concurrency=args.concurrency,
        output_dir=args.output,
    )
    runner.run()


def _run_experiment(args):
    from token_efficient_agent.config import AgentConfig, ProviderType
    from adaptive_compression.adaptive_experiment import AdaptiveExperiment

    provider = ProviderType(args.provider)
    config = AgentConfig(
        provider=provider,
        openrouter_model=args.model,
    )

    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    only_configs = None
    if args.configs:
        only_configs = [c.strip() for c in args.configs.split(",")]

    experiment = AdaptiveExperiment(
        config=config,
        domain=args.domain,
        task_ids=task_ids,
        num_tasks=args.num_tasks,
        temperature=args.temperature,
        concurrency=args.concurrency,
        output_dir=args.output,
        only_configs=only_configs,
    )
    experiment.run()


if __name__ == "__main__":
    main()

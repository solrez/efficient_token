#!/usr/bin/env python3
"""CLI entry point for the token-efficient agent framework."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tau-bench"))

import asyncio

import click
from dotenv import load_dotenv
from rich.console import Console

from token_efficient_agent.config import AgentConfig, OptimizationFlags, ProviderType, TracingConfig

load_dotenv()
console = Console()

RESULTS_DIR = Path(__file__).parent / "results"


def _make_experiment_dir(provider: str, model: str = "", tag: str = "experiment") -> Path:
    """Create a timestamped results directory for this experiment run."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [tag, provider]
    if model:
        parts.append(model.replace("/", "_"))
    name = "_".join(parts) + f"_{ts}"
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@click.command()
@click.option("--provider", type=click.Choice(["anthropic", "openai", "openrouter"]), default="anthropic", help="LLM provider")
@click.option("--model", "openrouter_model", type=str, default="", help="OpenRouter model preset (deepseek-v4-flash, qwen3-plus, llama4-scout, mistral-small) or full model ID")
@click.option("--cache/--no-cache", default=False, help="Enable prompt caching layout optimization")
@click.option("--prune/--no-prune", default=False, help="Enable tool definition pruning")
@click.option("--progressive/--no-progressive", default=False, help="Enable progressive tool disclosure")
@click.option("--mask/--no-mask", default=False, help="Enable observation masking")
@click.option("--mask-window", default=3, type=int, help="Number of recent tool outputs to keep unmasked")
@click.option("--route/--no-route", default=False, help="Enable model routing (strong/weak)")
@click.option("--scenario", type=str, default="market_research", help="Scenario to run (market_research, data_pipeline_debug, customer_analysis)")
@click.option("--benchmark", is_flag=True, help="Run benchmark across all optimization combos")
@click.option("--accuracy", is_flag=True, help="Run BFCL accuracy benchmark")
@click.option("--experiment", is_flag=True, help="Run BFCL factorial experiment (8 configs × N runs)")
@click.option("--tau-bench", "tau_bench", is_flag=True, help="Run Tau-bench multi-turn experiment (20 configs)")
@click.option("--tau-domain", type=click.Choice(["retail", "airline"]), default="retail", help="Tau-bench domain")
@click.option("--tau-tasks", type=int, default=None, help="Number of Tau-bench tasks (default: all)")
@click.option("--tau-only", type=str, default=None, help="Only run these configs (comma-separated labels, e.g. 'prune,route,prune+route')")
@click.option("--tau-exclude", type=str, default=None, help="Exclude these configs (comma-separated labels, e.g. 'baseline,cache')")
@click.option("--accuracy-limit", type=int, default=None, help="Limit test cases per BFCL category")
@click.option("--num-runs", type=int, default=1, help="Number of runs per configuration")
@click.option("--temperature", type=float, default=0.0, help="LLM temperature (0.3 recommended for experiments)")
@click.option("--resume", type=str, default=None, help="Resume experiment from existing results dir")
@click.option("--analyze", type=str, default=None, help="Analyze a previous experiment CSV (path)")
@click.option("--export-csv", type=str, default=None, help="Export benchmark results to CSV")
@click.option("--concurrency", type=int, default=5, help="Max concurrent API requests")
@click.option("--max-steps", type=int, default=15, help="Maximum agent steps")
@click.option("--langfuse-public-key", envvar="LANGFUSE_PUBLIC_KEY", default="", help="Langfuse public key")
@click.option("--langfuse-secret-key", envvar="LANGFUSE_SECRET_KEY", default="", help="Langfuse secret key")
@click.option("--langfuse-host", envvar=["LANGFUSE_HOST", "LANGFUSE_BASE_URL"], default="https://cloud.langfuse.com", help="Langfuse host")
def main(
    provider, openrouter_model, cache, prune, progressive, mask, mask_window, route,
    scenario, benchmark, accuracy, experiment, tau_bench, tau_domain, tau_tasks,
    tau_only, tau_exclude,
    accuracy_limit, num_runs, temperature, resume, analyze, export_csv, concurrency, max_steps,
    langfuse_public_key, langfuse_secret_key, langfuse_host,
):
    """Run the token-efficient agent with specified optimizations."""
    import os
    if not langfuse_host or langfuse_host == "https://cloud.langfuse.com":
        langfuse_host = os.environ.get("LANGFUSE_BASE_URL", os.environ.get("LANGFUSE_HOST", langfuse_host))

    config = AgentConfig(
        provider=ProviderType(provider),
        openrouter_model=openrouter_model,
        optimizations=OptimizationFlags(
            prompt_caching=cache,
            tool_pruning=prune,
            progressive_disclosure=progressive,
            observation_masking=mask,
            observation_masking_window=mask_window,
            model_routing=route,
        ),
        tracing=TracingConfig(
            enabled=bool(langfuse_secret_key),
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            langfuse_host=langfuse_host,
        ),
        max_steps=max_steps,
    )

    if analyze:
        _run_analysis(analyze)
    elif tau_bench:
        _run_tau_bench(config, tau_domain, tau_tasks, num_runs, temperature, export_csv, resume, tau_only, tau_exclude)
    elif experiment:
        asyncio.run(_run_experiment(config, accuracy_limit, num_runs, temperature, concurrency, resume, export_csv))
    elif accuracy:
        asyncio.run(_run_accuracy(config, accuracy_limit, export_csv))
    elif benchmark:
        asyncio.run(_run_benchmark(config, scenario, export_csv))
    else:
        asyncio.run(_run_single(config, scenario))


async def _run_single(config: AgentConfig, scenario_name: str):
    from langchain_core.messages import HumanMessage
    from token_efficient_agent.graph import build_agent_graph
    from token_efficient_agent.tools.registry import create_default_registry
    from token_efficient_agent.tracing import TracingManager
    from token_efficient_agent.benchmark.scenarios import SCENARIOS

    scenario = SCENARIOS.get(scenario_name)
    if not scenario:
        console.print(f"[red]Unknown scenario: {scenario_name}[/red]")
        console.print(f"Available: {', '.join(SCENARIOS.keys())}")
        return

    opts = config.optimizations
    console.print(f"[bold]Provider:[/bold] {config.provider.value}")
    console.print(f"[bold]Scenario:[/bold] {scenario_name}")
    console.print(f"[bold]Optimizations:[/bold] {opts.label()}")
    console.print()

    import uuid
    tracing = TracingManager(config, session_id=str(uuid.uuid4())[:8])
    callback = tracing.create_callback()
    callbacks = [callback] if callback else []

    registry = create_default_registry()
    graph = build_agent_graph(config, registry=registry, callbacks=callbacks)

    initial_state = {
        "messages": [HumanMessage(content=scenario.initial_message)],
        "step_count": 0,
        "selected_tool": None,
        "current_model_tier": "strong",
    }

    console.print("[dim]Running agent...[/dim]")
    import time
    start = time.perf_counter()

    with tracing.trace_context(scenario_name, opts):
        result = await graph.ainvoke(
            initial_state,
            config={"callbacks": callbacks, "recursion_limit": config.max_steps * 3},
        )
    tracing.flush()

    elapsed_ms = (time.perf_counter() - start) * 1000

    console.print()
    console.print(f"[bold green]Completed in {elapsed_ms / 1000:.1f}s[/bold green]")
    console.print(f"[bold]Steps:[/bold] {result.get('step_count', 'N/A')}")

    last_msg = result["messages"][-1]
    console.print()
    console.print("[bold]Final response:[/bold]")
    console.print(last_msg.content if hasattr(last_msg, "content") else str(last_msg))

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    llm_calls = 0
    for msg in result["messages"]:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            llm_calls += 1
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            cache_info = usage.get("input_token_details", {}) or {}
            cr = cache_info.get("cache_read", 0) or cache_info.get("cached_tokens", 0) or 0
            cw = cache_info.get("cache_creation", 0) or 0
            total_cache_read += cr
            total_cache_write += cw

    console.print()
    console.print("[bold]Token usage:[/bold]")
    console.print(f"  Input tokens:  {total_input:,}")
    console.print(f"  Output tokens: {total_output:,}")
    if total_cache_read:
        console.print(f"  Cache reads:   {total_cache_read:,}")
    if total_cache_write:
        console.print(f"  Cache writes:  {total_cache_write:,}")
    console.print(f"  LLM calls:     {llm_calls}")
    console.print(f"  Latency:       {elapsed_ms / 1000:.1f}s")

    model_name = config.get_model()
    from token_efficient_agent.config import INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK, CACHE_READ_COST_PER_MTOK, CACHE_WRITE_COST_PER_MTOK
    input_rate = INPUT_COST_PER_MTOK.get(model_name, 3.0)
    output_rate = OUTPUT_COST_PER_MTOK.get(model_name, 15.0)
    cache_read_rate = CACHE_READ_COST_PER_MTOK.get(model_name, 0.0)
    cache_write_rate = CACHE_WRITE_COST_PER_MTOK.get(model_name, 0.0)
    non_cached = total_input - total_cache_read - total_cache_write
    cost = (non_cached / 1_000_000 * input_rate
            + total_output / 1_000_000 * output_rate
            + total_cache_read / 1_000_000 * cache_read_rate
            + total_cache_write / 1_000_000 * cache_write_rate)
    console.print(f"  Est. cost:     ${cost:.4f}")


async def _run_benchmark(config: AgentConfig, scenario_filter: str | None, export_csv: str | None):
    from token_efficient_agent.benchmark.runner import BenchmarkRunner
    from token_efficient_agent.benchmark.reporter import BenchmarkReporter
    from token_efficient_agent.benchmark.scenarios import SCENARIOS

    if scenario_filter and scenario_filter != "all":
        scenarios = [scenario_filter]
    else:
        scenarios = list(SCENARIOS.keys())

    console.print(f"[bold]Benchmark mode[/bold]")
    console.print(f"  Provider: {config.provider.value}")
    console.print(f"  Scenarios: {', '.join(scenarios)}")
    console.print(f"  Optimization profiles: 9 combos")
    console.print(f"  Total runs: {len(scenarios) * 9}")
    console.print()

    exp_dir = _make_experiment_dir(config.provider.value, config.openrouter_model, "benchmark")
    runner = BenchmarkRunner(base_config=config, scenarios=scenarios)
    results = await runner.run_all()

    reporter = BenchmarkReporter(results)
    reporter.print_comparison_table()
    reporter.print_per_optimization_impact()

    csv_out = export_csv or str(exp_dir / "results.csv")
    reporter.export_csv(csv_out)
    console.print(f"\n[bold]Results exported to:[/bold] {csv_out}")


async def _run_accuracy(config: AgentConfig, limit: int | None, export_csv: str | None):
    from token_efficient_agent.accuracy.runner import (
        AccuracyBenchmarkRunner,
        print_accuracy_report,
        export_accuracy_csv,
    )

    console.print("[bold]BFCL Accuracy Benchmark[/bold]")
    console.print(f"  Provider: {config.provider.value}")
    if limit:
        console.print(f"  Limit: {limit} cases per category")
    console.print()

    runner = AccuracyBenchmarkRunner(config=config, limit_per_category=limit)
    results = await runner.run_all()

    print_accuracy_report(results)

    if export_csv:
        export_accuracy_csv(results, export_csv)
        console.print(f"\n[bold]Results exported to:[/bold] {export_csv}")


async def _run_experiment(
    config: AgentConfig,
    limit: int | None,
    num_runs: int,
    temperature: float,
    concurrency: int,
    resume: str | None,
    export_csv: str | None,
):
    from token_efficient_agent.accuracy.experiment import BFCL_CONFIGS, TEMPERATURE, NUM_RUNS
    from token_efficient_agent.accuracy.runner import (
        AccuracyBenchmarkRunner,
        print_accuracy_report,
        export_accuracy_csv,
        export_per_case_csv,
    )
    from token_efficient_agent.accuracy.analysis import (
        analyze_experiment,
        print_analysis_report,
        export_interaction_csv,
    )

    actual_runs = num_runs if num_runs > 1 else NUM_RUNS
    actual_temp = temperature if temperature > 0 else TEMPERATURE

    if resume:
        exp_dir = Path(resume)
        console.print(f"  [bold]Resuming from:[/bold] {exp_dir}")
    else:
        exp_dir = _make_experiment_dir(config.provider.value, config.openrouter_model, "bfcl")
    csv_path = str(exp_dir / "results.csv")
    per_case_path = str(exp_dir / "per_case.csv")
    interaction_path = str(exp_dir / "interactions.csv")

    console.print("[bold]Factorial Experiment[/bold]")
    console.print(f"  Provider: {config.provider.value}")
    if config.openrouter_model:
        console.print(f"  Model: {config.openrouter_model} ({config.get_model()})")
    console.print(f"  Configurations: {len(BFCL_CONFIGS)}")
    console.print(f"  Runs per config: {actual_runs}")
    console.print(f"  Temperature: {actual_temp}")
    console.print(f"  Concurrency: {concurrency}")
    if limit:
        console.print(f"  Cases per category: {limit} (stratified sample)")
    else:
        console.print(f"  Cases: full BFCL (1,036)")

    model_name = config.get_model()
    from token_efficient_agent.config import INPUT_COST_PER_MTOK
    cost_per_call = INPUT_COST_PER_MTOK.get(model_name, 1.0) * 0.003
    total_calls = len(BFCL_CONFIGS) * (limit * 4 if limit else 1036) * actual_runs
    est_cost = total_calls * cost_per_call
    console.print(f"  Estimated API calls: {total_calls:,}")
    console.print(f"  Estimated cost: ${est_cost:.2f}")
    console.print(f"  Output dir: {exp_dir}")
    console.print(f"  Checkpoint: {csv_path} (auto-resume on restart)")
    console.print()

    params_file = exp_dir / "params.json"
    current_params = {
        "provider": config.provider.value,
        "model": config.openrouter_model or config.get_model(),
        "accuracy_limit": limit,
        "num_runs": actual_runs,
        "temperature": actual_temp,
        "concurrency": concurrency,
        "configs": len(BFCL_CONFIGS),
    }
    if params_file.exists():
        import json
        saved = json.loads(params_file.read_text())
        if saved.get("accuracy_limit") != current_params["accuracy_limit"]:
            console.print(f"[red]ERROR: --accuracy-limit mismatch! Saved: {saved['accuracy_limit']}, Current: {limit}[/red]")
            console.print(f"[red]Delete {exp_dir} and start fresh, or use the same --accuracy-limit[/red]")
            return
        if saved.get("num_runs") != current_params["num_runs"]:
            console.print(f"[red]ERROR: --num-runs mismatch! Saved: {saved['num_runs']}, Current: {actual_runs}[/red]")
            return
    else:
        import json
        params_file.write_text(json.dumps(current_params, indent=2))

    runner = AccuracyBenchmarkRunner(
        config=config,
        profiles=BFCL_CONFIGS,
        limit_per_category=limit,
        num_runs=actual_runs,
        temperature=actual_temp,
        concurrency=concurrency,
        checkpoint_path=csv_path,
    )
    results = await runner.run_all()

    print_accuracy_report(results)

    export_accuracy_csv(results, csv_path)
    console.print(f"\n[bold]Aggregate results:[/bold] {csv_path}")

    export_per_case_csv(results, per_case_path)
    console.print(f"[bold]Per-case results:[/bold] {per_case_path}")

    console.print("\n[bold]Running factorial analysis...[/bold]")
    analysis = analyze_experiment(csv_path)
    print_analysis_report(analysis)

    export_interaction_csv(analysis, interaction_path)
    console.print(f"[bold]Interaction effects:[/bold] {interaction_path}")

    if export_csv:
        import shutil
        shutil.copy2(csv_path, export_csv)
        console.print(f"[bold]Also copied to:[/bold] {export_csv}")


def _run_analysis(csv_path: str):
    from token_efficient_agent.accuracy.analysis import (
        analyze_experiment,
        print_analysis_report,
        export_interaction_csv,
    )

    console.print(f"[bold]Analyzing:[/bold] {csv_path}")

    analysis = analyze_experiment(csv_path)
    print_analysis_report(analysis)

    for category in ["simple", "multiple", "parallel", "irrelevance"]:
        console.print(f"\n[bold]Category: {category}[/bold]")
        cat_analysis = analyze_experiment(csv_path, category=category)
        print_analysis_report(cat_analysis)

    interaction_path = csv_path.replace(".csv", "_interactions.csv")
    export_interaction_csv(analysis, interaction_path)
    console.print(f"\n[bold]Interaction effects exported:[/bold] {interaction_path}")


def _run_tau_bench(
    config: AgentConfig,
    domain: str,
    num_tasks: int | None,
    num_runs: int,
    temperature: float,
    export_csv: str | None,
    resume: str | None = None,
    tau_only: str | None = None,
    tau_exclude: str | None = None,
):
    from token_efficient_agent.accuracy.experiment import TAU_CONFIGS, TEMPERATURE, NUM_RUNS
    from token_efficient_agent.accuracy.tau_runner import (
        TauBenchRunner,
        print_tau_report,
        export_tau_csv,
    )

    actual_runs = num_runs
    actual_temp = temperature if temperature > 0 else TEMPERATURE

    configs = TAU_CONFIGS
    if tau_only:
        only_labels = {s.strip() for s in tau_only.split(",")}
        configs = [c for c in configs if c.label() in only_labels]
        if not configs:
            console.print(f"[red]No configs matched --tau-only={tau_only}[/red]")
            console.print(f"Available: {', '.join(c.label() for c in TAU_CONFIGS)}")
            return
    elif tau_exclude:
        exclude_labels = {s.strip() for s in tau_exclude.split(",")}
        configs = [c for c in configs if c.label() not in exclude_labels]

    if resume:
        exp_dir = Path(resume)
        console.print(f"  [bold]Resuming from:[/bold] {exp_dir}")
    else:
        exp_dir = _make_experiment_dir(config.provider.value, config.openrouter_model, f"tau_{domain}")
    csv_out = export_csv or str(exp_dir / "results.csv")

    console.print(f"[bold]Tau-bench Experiment ({domain})[/bold]")
    console.print(f"  Provider: {config.provider.value}")
    if config.openrouter_model:
        console.print(f"  Model: {config.openrouter_model} ({config.get_model()})")
    console.print(f"  Configurations: {len(configs)}")
    console.print(f"  Tasks: {num_tasks or 'all'}")
    console.print(f"  Runs: {actual_runs}")
    console.print(f"  Output dir: {exp_dir}")
    if resume:
        console.print(f"  [bold yellow]Resume mode:[/bold yellow] will skip completed configs, retry failed ones")
    console.print()

    import json
    params_file = exp_dir / "params.json"
    if not resume:
        params_file.write_text(json.dumps({
            "type": "tau_bench",
            "provider": config.provider.value,
            "model": config.openrouter_model or config.get_model(),
            "domain": domain,
            "num_tasks": num_tasks,
            "num_runs": actual_runs,
            "temperature": actual_temp,
            "configs": len(configs),
        }, indent=2))

    log_file = str(exp_dir / "experiment.log")

    import sys, io

    import re as _re
    _ansi_re = _re.compile(r'\x1b\[[0-9;]*m')

    class TeeWriter:
        """Write to both terminal and log file, stripping ANSI codes from the log."""
        def __init__(self, original, log_path):
            self.original = original
            self.log = open(log_path, "a")
            self.encoding = getattr(original, 'encoding', 'utf-8')
        def write(self, text):
            self.original.write(text)
            self.log.write(_ansi_re.sub('', text))
            self.log.flush()
        def flush(self):
            self.original.flush()
            self.log.flush()
        def close(self):
            self.log.close()
        def isatty(self):
            return self.original.isatty()
        def fileno(self):
            return self.original.fileno()

    tee_stdout = TeeWriter(sys.stdout, log_file)
    tee_stderr = TeeWriter(sys.stderr, log_file)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    try:
        from rich.console import Console as RichConsole
        log_console = RichConsole(file=tee_stdout, force_terminal=True)

        log_console.print(f"  Log file: {log_file}")

        runner = TauBenchRunner(
            config=config,
            domain=domain,
            profiles=configs,
            num_tasks=num_tasks,
            num_runs=actual_runs,
            temperature=actual_temp,
            results_csv=csv_out,
            log_file=log_file,
            resume=bool(resume),
        )
        results = runner.run_all()

        print_tau_report(results)

        export_tau_csv(results, csv_out)
        log_console.print(f"\n[bold]Results exported to:[/bold] {csv_out}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee_stdout.close()
        tee_stderr.close()


if __name__ == "__main__":
    main()

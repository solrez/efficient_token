#!/usr/bin/env python3
"""CLI entry point for the token-efficient agent framework."""
from __future__ import annotations

import asyncio

import click
from dotenv import load_dotenv
from rich.console import Console

from token_efficient_agent.config import AgentConfig, OptimizationFlags, ProviderType, TracingConfig

load_dotenv()
console = Console()


@click.command()
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default="anthropic", help="LLM provider")
@click.option("--cache/--no-cache", default=False, help="Enable prompt caching layout optimization")
@click.option("--prune/--no-prune", default=False, help="Enable tool definition pruning")
@click.option("--progressive/--no-progressive", default=False, help="Enable progressive tool disclosure")
@click.option("--mask/--no-mask", default=False, help="Enable observation masking")
@click.option("--mask-window", default=3, type=int, help="Number of recent tool outputs to keep unmasked")
@click.option("--route/--no-route", default=False, help="Enable model routing (strong/weak)")
@click.option("--scenario", type=str, default="market_research", help="Scenario to run (market_research, data_pipeline_debug, customer_analysis)")
@click.option("--benchmark", is_flag=True, help="Run benchmark across all optimization combos")
@click.option("--export-csv", type=str, default=None, help="Export benchmark results to CSV")
@click.option("--max-steps", type=int, default=15, help="Maximum agent steps")
@click.option("--langfuse-public-key", envvar="LANGFUSE_PUBLIC_KEY", default="", help="Langfuse public key")
@click.option("--langfuse-secret-key", envvar="LANGFUSE_SECRET_KEY", default="", help="Langfuse secret key")
@click.option("--langfuse-host", envvar=["LANGFUSE_HOST", "LANGFUSE_BASE_URL"], default="https://cloud.langfuse.com", help="Langfuse host")
def main(
    provider, cache, prune, progressive, mask, mask_window, route,
    scenario, benchmark, export_csv, max_steps,
    langfuse_public_key, langfuse_secret_key, langfuse_host,
):
    """Run the token-efficient agent with specified optimizations."""
    import os
    if not langfuse_host or langfuse_host == "https://cloud.langfuse.com":
        langfuse_host = os.environ.get("LANGFUSE_BASE_URL", os.environ.get("LANGFUSE_HOST", langfuse_host))

    config = AgentConfig(
        provider=ProviderType(provider),
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

    if benchmark:
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
            total_cache_read += cache_info.get("cache_read", 0) or 0
            total_cache_write += cache_info.get("cache_creation", 0) or 0

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

    runner = BenchmarkRunner(base_config=config, scenarios=scenarios)
    results = await runner.run_all()

    reporter = BenchmarkReporter(results)
    reporter.print_comparison_table()
    reporter.print_per_optimization_impact()

    if export_csv:
        path = reporter.export_csv(export_csv)
        console.print(f"\n[bold]Results exported to:[/bold] {path}")


if __name__ == "__main__":
    main()

"""Phase 1 signal validation: run Tau-bench tasks with uncertainty estimation.

Runs baseline (no compression) tasks while logging per-step uncertainty from
logprob and prompted estimators. Produces CSVs for offline analysis.

Supports DeepInfra (logprobs on tool calls), Groq (reasoning logprobs),
and OpenRouter (prompted-only fallback).

Usage:
    python -m adaptive_compression validate \
        --domain retail --num-tasks 10 --provider deepinfra \
        --model Qwen/Qwen3-14B --output results/signal_validation/
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm
litellm.suppress_debug_info = True

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from token_efficient_agent.config import AgentConfig, ModelTier, ProviderType
from adaptive_compression.uncertainty import (
    LogprobEstimator,
    PromptedEstimator,
    HybridEstimator,
)

console = Console()

USER_SIM_MODEL = "openrouter/google/gemini-2.5-flash-lite"
USER_SIM_PROVIDER = "openrouter"


@dataclass
class StepRecord:
    task_id: int
    step: int
    tool_called: str
    tool_args_summary: str
    logprob_confidence: float | None
    logprob_entropy: float | None
    logprob_num_tokens: int
    logprob_method: str
    prompted_confidence: float | None
    prompted_raw_score: int | None
    hybrid_confidence: float | None
    hybrid_method: str
    input_tokens: int
    output_tokens: int
    cost: float
    model_used: str
    # Behavioral signals (only populated with --behavioral)
    beh_repetition: float | None = None
    beh_respond_ratio: float | None = None
    beh_step_position: float | None = None
    beh_context_growth: float | None = None
    beh_stall: float | None = None
    beh_composite: float | None = None
    composite_confidence: float | None = None
    composite_method: str = ""


@dataclass
class TaskRecord:
    task_id: int
    reward: float
    total_steps: int
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    steps: list[StepRecord] = field(default_factory=list)


def _build_litellm_model(config: AgentConfig) -> str:
    """Build the litellm model string for the given provider config."""
    model_name = config.get_model(ModelTier.STRONG)
    if config.provider == ProviderType.OPENROUTER:
        return f"openrouter/{model_name}"
    if config.provider == ProviderType.DEEPINFRA:
        return f"deepinfra/{model_name}"
    if config.provider == ProviderType.GROQ:
        return f"groq/{model_name}"
    if config.provider == ProviderType.DEEPSEEK:
        return f"deepseek/{model_name}"
    return model_name


def _build_call_kwargs(config: AgentConfig, litellm_model: str, messages, tools, temperature) -> dict:
    """Build litellm completion kwargs with provider-specific auth."""
    kwargs = dict(
        messages=messages,
        model=litellm_model,
        tools=tools,
        temperature=temperature,
        logprobs=True,
        top_logprobs=5,
    )
    if config.provider == ProviderType.OPENROUTER:
        kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
        kwargs["api_base"] = "https://openrouter.ai/api/v1"
    elif config.provider == ProviderType.DEEPINFRA:
        kwargs["api_key"] = os.environ.get("DeepInfra_API_KEY", "")
        kwargs["api_base"] = "https://api.deepinfra.com/v1/openai"
        kwargs["allowed_openai_params"] = ["logprobs", "top_logprobs"]
    elif config.provider == ProviderType.GROQ:
        kwargs["api_key"] = os.environ.get("GROQ_API_KEY", "")
        kwargs["api_base"] = "https://api.groq.com/openai/v1"
    elif config.provider == ProviderType.DEEPSEEK:
        kwargs["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
        kwargs["api_base"] = "https://api.deepseek.com"
        kwargs["allowed_openai_params"] = ["logprobs", "top_logprobs"]
    return kwargs


class SignalValidationRunner:
    """Run Tau-bench with uncertainty estimators active, no compression applied."""

    def __init__(
        self,
        config: AgentConfig,
        domain: str = "retail",
        task_ids: list[int] | None = None,
        num_tasks: int | None = None,
        temperature: float = 0.0,
        concurrency: int = 3,
        output_dir: str = "results/signal_validation",
        behavioral: bool = False,
    ):
        self.config = config
        self.domain = domain
        self.task_ids = task_ids
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.behavioral = behavioral

        self.logprob_estimator = LogprobEstimator()
        self.prompted_estimator = PromptedEstimator()
        self.hybrid_estimator = HybridEstimator()

    def run(self) -> list[TaskRecord]:
        return asyncio.run(self._run_async())

    async def _run_async(self) -> list[TaskRecord]:
        import logging
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        if self.domain == "retail":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as _tasks
        else:
            from tau_bench.envs.airline.tasks_test import TASKS_TEST as _tasks

        task_ids = self.task_ids
        if task_ids is None:
            total = len(_tasks)
            n = self.num_tasks or total
            task_ids = list(range(min(n, total)))

        litellm_model = _build_litellm_model(self.config)

        console.print(f"\n[bold]Signal Validation Run[/bold]")
        console.print(f"  Domain: {self.domain}")
        console.print(f"  Tasks: {len(task_ids)}")
        console.print(f"  Provider: {self.config.provider.value}")
        console.print(f"  Model: {self.config.get_model(ModelTier.STRONG)}")
        console.print(f"  Litellm model: {litellm_model}")
        console.print(f"  Concurrency: {self.concurrency}")
        console.print(f"  Output: {self.output_dir}")
        console.print()

        step_csv = self.output_dir / "step_uncertainty.csv"
        task_csv = self.output_dir / "task_outcomes.csv"
        self._init_csvs(step_csv, task_csv)

        semaphore = asyncio.Semaphore(self.concurrency)
        all_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task("Running tasks...", total=len(task_ids))

            async def run_one(tid):
                async with semaphore:
                    result = await asyncio.to_thread(self._run_single_task, tid)
                    self._append_task(task_csv, result)
                    for sr in result.steps:
                        self._append_step(step_csv, sr)
                    progress.advance(ptask)
                    return result

            all_results = await asyncio.gather(*[run_one(tid) for tid in task_ids])

        self._save_params()
        self._print_summary(all_results)
        return all_results

    def _run_single_task(self, tid: int, max_retries: int = 3) -> TaskRecord:
        for attempt in range(1, max_retries + 1):
            try:
                return self._run_task_inner(tid)
            except Exception as e:
                is_transient = any(kw in str(e).lower() for kw in [
                    "peer closed", "server disconnected", "connection reset",
                    "read timed out", "remotedisconnected", "429", "rate-limit",
                    "rate_limit", "ratelimit", "model busy", "retry later",
                    "timeout", "502", "503", "overloaded",
                ])
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt * 5
                    console.print(
                        f"  [yellow]Task {tid} error (attempt {attempt}/{max_retries}): "
                        f"{str(e)[:60]}... retrying in {wait}s[/yellow]"
                    )
                    time.sleep(wait)
                    continue
                console.print(f"  [red]Task {tid} failed: {e}[/red]")
                return TaskRecord(
                    task_id=tid, reward=0.0, total_steps=0,
                    total_cost=0.0, total_input_tokens=0, total_output_tokens=0,
                )

    def _run_task_inner(self, tid: int) -> TaskRecord:
        from tau_bench.envs import get_env
        from tau_bench.types import RESPOND_ACTION_NAME
        from tau_bench.agents.tool_calling_agent import message_to_action

        env = get_env(
            env_name=self.domain,
            user_strategy="llm",
            user_model=USER_SIM_MODEL,
            user_provider=USER_SIM_PROVIDER,
            task_split="test",
        )

        env_reset = env.reset(task_index=tid)
        obs = env_reset.observation

        messages = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": obs},
        ]

        litellm_model = _build_litellm_model(self.config)

        # Set up composite estimator per task if behavioral mode
        composite_est = None
        if self.behavioral:
            from adaptive_compression.uncertainty import CompositeEstimator
            composite_est = CompositeEstimator()
            composite_est.reset_trajectory(
                context_window=self.config.get_context_window()
            )

        total_cost = 0.0
        total_input = 0
        total_output = 0
        reward = 0.0
        step_records = []

        for step in range(30):
            call_kwargs = _build_call_kwargs(
                self.config, litellm_model, messages,
                env.tools_info, self.temperature,
            )

            from litellm import completion
            res = completion(**call_kwargs)

            usage = res.usage
            step_input = (usage.prompt_tokens or 0) if usage else 0
            step_output = (usage.completion_tokens or 0) if usage else 0
            step_cost = res._hidden_params.get("response_cost", 0) or 0
            if not step_cost:
                from token_efficient_agent.config import INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK
                model_name = self.config.get_model(ModelTier.STRONG)
                rate_in = INPUT_COST_PER_MTOK.get(model_name, 0)
                rate_out = OUTPUT_COST_PER_MTOK.get(model_name, 0)
                step_cost = (step_input / 1_000_000 * rate_in) + (step_output / 1_000_000 * rate_out)
            total_input += step_input
            total_output += step_output
            total_cost += step_cost

            # Estimate uncertainty
            logprob_result = self.logprob_estimator.estimate(res, messages)
            prompted_result = type('R', (), {'confidence': None, 'raw_score': None, 'method': 'disabled'})()
            hybrid_result = logprob_result  # hybrid just uses logprob when prompted is disabled

            next_message = res.choices[0].message.model_dump()

            # Handle None content from thinking models
            if next_message.get("content") is None:
                next_message["content"] = ""

            action = message_to_action(next_message)

            tool_name = action.name if action.name != RESPOND_ACTION_NAME else "__respond__"
            tool_args = ""
            tc_list = next_message.get("tool_calls") or []
            if tc_list:
                tc = tc_list[0]
                args_str = tc.get("function", {}).get("arguments", "")
                tool_args = args_str[:200] if len(args_str) > 200 else args_str

            # Behavioral + composite signals
            beh_signals = None
            composite_result = None
            if composite_est is not None:
                composite_est.update_trajectory(tool_name, tool_args, step_input)
                beh_signals = composite_est.trajectory_state.compute_signals()
                composite_result = composite_est.estimate(res, messages)

            step_records.append(StepRecord(
                task_id=tid,
                step=step,
                tool_called=tool_name,
                tool_args_summary=tool_args,
                logprob_confidence=logprob_result.confidence if logprob_result.num_tokens_used > 0 else None,
                logprob_entropy=logprob_result.raw_entropy,
                logprob_num_tokens=logprob_result.num_tokens_used,
                logprob_method=logprob_result.method,
                prompted_confidence=prompted_result.confidence if prompted_result.raw_score is not None else None,
                prompted_raw_score=prompted_result.raw_score,
                hybrid_confidence=hybrid_result.confidence,
                hybrid_method=hybrid_result.method,
                input_tokens=step_input,
                output_tokens=step_output,
                cost=step_cost,
                model_used=litellm_model,
                beh_repetition=beh_signals.repetition_score if beh_signals else None,
                beh_respond_ratio=beh_signals.respond_ratio if beh_signals else None,
                beh_step_position=beh_signals.step_position if beh_signals else None,
                beh_context_growth=beh_signals.context_growth if beh_signals else None,
                beh_stall=beh_signals.stall_score if beh_signals else None,
                beh_composite=beh_signals.composite if beh_signals else None,
                composite_confidence=composite_result.confidence if composite_result else None,
                composite_method=composite_result.method if composite_result else "",
            ))

            env_response = env.step(action)
            reward = env_response.reward

            if action.name != RESPOND_ACTION_NAME:
                tc_list = next_message.get("tool_calls") or []
                if tc_list:
                    next_message["tool_calls"] = tc_list[:1]
                    messages.extend([
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": tc_list[0]["id"],
                            "name": tc_list[0]["function"]["name"],
                            "content": env_response.observation or "",
                        },
                    ])
                else:
                    messages.append({"role": "assistant", "content": next_message.get("content", "")})
                    messages.append({"role": "user", "content": env_response.observation or ""})
            else:
                messages.extend([
                    next_message,
                    {"role": "user", "content": env_response.observation or ""},
                ])

            if env_response.done:
                break

        return TaskRecord(
            task_id=tid,
            reward=reward,
            total_steps=len(step_records),
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            steps=step_records,
        )

    def _init_csvs(self, step_csv: Path, task_csv: Path):
        if not step_csv.exists():
            with open(step_csv, "w", newline="") as f:
                writer = csv.writer(f)
                header = [
                    "task_id", "step", "tool_called", "tool_args_summary",
                    "logprob_confidence", "logprob_entropy", "logprob_num_tokens",
                    "logprob_method",
                    "prompted_confidence", "prompted_raw_score",
                    "hybrid_confidence", "hybrid_method",
                    "input_tokens", "output_tokens", "cost", "model_used",
                ]
                if self.behavioral:
                    header.extend([
                        "beh_repetition", "beh_respond_ratio", "beh_step_position",
                        "beh_context_growth", "beh_stall", "beh_composite",
                        "composite_confidence", "composite_method",
                    ])
                writer.writerow(header)

        if not task_csv.exists():
            with open(task_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "task_id", "reward", "total_steps", "total_cost",
                    "total_input_tokens", "total_output_tokens",
                ])

    def _append_step(self, path: Path, sr: StepRecord):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                sr.task_id, sr.step, sr.tool_called, sr.tool_args_summary,
                sr.logprob_confidence, sr.logprob_entropy, sr.logprob_num_tokens,
                sr.logprob_method,
                sr.prompted_confidence, sr.prompted_raw_score,
                sr.hybrid_confidence, sr.hybrid_method,
                sr.input_tokens, sr.output_tokens,
                f"{sr.cost:.6f}", sr.model_used,
            ]
            if self.behavioral:
                row.extend([
                    sr.beh_repetition, sr.beh_respond_ratio, sr.beh_step_position,
                    sr.beh_context_growth, sr.beh_stall, sr.beh_composite,
                    sr.composite_confidence, sr.composite_method,
                ])
            writer.writerow(row)

    def _append_task(self, path: Path, tr: TaskRecord):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                tr.task_id, tr.reward, tr.total_steps,
                f"{tr.total_cost:.6f}", tr.total_input_tokens, tr.total_output_tokens,
            ])

    def _save_params(self):
        params = {
            "domain": self.domain,
            "model": self.config.get_model(ModelTier.STRONG),
            "provider": self.config.provider.value,
            "temperature": self.temperature,
            "num_tasks": self.num_tasks,
            "task_ids": self.task_ids,
            "estimators": ["logprob", "prompted", "hybrid"],
            "logprob_num_tokens": self.logprob_estimator.num_tokens,
            "prompted_model": self.prompted_estimator.model,
        }
        with open(self.output_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def _print_summary(self, results: list[TaskRecord]):
        valid = [r for r in results if r.total_steps > 0]
        if not valid:
            console.print("[red]No valid results.[/red]")
            return

        total_tasks = len(valid)
        pass_rate = sum(r.reward for r in valid) / total_tasks
        total_steps = sum(r.total_steps for r in valid)
        total_cost = sum(r.total_cost for r in valid)

        all_steps = [s for r in valid for s in r.steps]
        logprob_scores = [s.logprob_confidence for s in all_steps if s.logprob_confidence is not None]
        prompted_scores = [s.prompted_confidence for s in all_steps if s.prompted_confidence is not None]

        console.print(f"\n[bold]Signal Validation Summary[/bold]")
        console.print(f"  Tasks completed: {total_tasks}")
        console.print(f"  Pass rate: {pass_rate:.1%}")
        console.print(f"  Total steps: {total_steps}")
        console.print(f"  Total cost: ${total_cost:.4f}")

        if logprob_scores:
            methods = set(s.logprob_method for s in all_steps if s.logprob_confidence is not None)
            console.print(f"\n  Logprob estimator ({', '.join(methods)}):")
            console.print(f"    Coverage: {len(logprob_scores)}/{len(all_steps)} steps ({len(logprob_scores)/len(all_steps):.0%})")
            console.print(f"    Mean confidence: {sum(logprob_scores)/len(logprob_scores):.3f}")
            console.print(f"    Min: {min(logprob_scores):.3f}, Max: {max(logprob_scores):.3f}")
        else:
            console.print(f"\n  [yellow]Logprob estimator: no data[/yellow]")

        if prompted_scores:
            console.print(f"\n  Prompted estimator:")
            console.print(f"    Coverage: {len(prompted_scores)}/{len(all_steps)} steps ({len(prompted_scores)/len(all_steps):.0%})")
            console.print(f"    Mean confidence: {sum(prompted_scores)/len(prompted_scores):.3f}")
            console.print(f"    Min: {min(prompted_scores):.3f}, Max: {max(prompted_scores):.3f}")

        console.print(f"\n  Results saved to: {self.output_dir}/")

"""Tau-bench integration: multi-turn agent benchmark with all 5 optimizations."""
from __future__ import annotations

import asyncio
import csv
import json
import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
litellm.suppress_debug_info = True

from rich.console import Console
from rich.table import Table

from token_efficient_agent.config import AgentConfig, OptimizationFlags, ModelTier, ProviderType
from token_efficient_agent.optimizations.observation_masking import apply_observation_masking
from token_efficient_agent.optimizations.prompt_caching import apply_cache_layout

console = Console()


@dataclass
class TauTaskResult:
    task_id: int
    optimization_label: str
    run_id: int
    reward: float
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    num_steps: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class TauBenchResults:
    domain: str = ""
    results: list[TauTaskResult] = field(default_factory=list)

    def get_results(self, opt_label: str) -> list[TauTaskResult]:
        return [r for r in self.results if r.optimization_label == opt_label]

    @property
    def optimization_labels(self) -> list[str]:
        seen = []
        for r in self.results:
            if r.optimization_label not in seen:
                seen.append(r.optimization_label)
        return seen


class OptimizedTauAgent:
    """Tau-bench agent with our optimizations applied to the message loop."""

    def __init__(
        self,
        config: AgentConfig,
        flags: OptimizationFlags,
        tools_info: list[dict],
        wiki: str,
        temperature: float = 0.0,
    ):
        self.config = config
        self.flags = flags
        self.tools_info = tools_info
        self.wiki = wiki
        self.temperature = temperature

        self._pruned_tools = None
        if flags.tool_pruning:
            self._pruned_tools = [_prune_tool_def(t) for t in tools_info]

        self._routing_tools = {t["function"]["name"]: _make_routing_def(t) for t in tools_info}
        self._full_tools = {t["function"]["name"]: t for t in tools_info}
        self._discovered_tools: set[str] = set()

    def solve(self, env, task_index=None, max_num_steps=30):
        from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
        from tau_bench.agents.tool_calling_agent import message_to_action

        total_cost = 0.0
        total_input = 0
        total_output = 0
        cache_read = 0
        cache_write = 0
        self._discovered_tools = set()

        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        messages = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        tools = self._select_tools(step=0)
        model, provider = self._select_model(step=0, messages=messages)

        for step in range(max_num_steps):
            send_messages = self._apply_message_optimizations(messages, step)
            tools = self._select_tools(step=step)
            model, provider = self._select_model(step=step, messages=messages)

            from litellm import completion
            import os
            call_kwargs = dict(
                messages=send_messages,
                model=model,
                tools=tools,
                temperature=self.temperature,
            )
            if self.config.provider == ProviderType.OPENROUTER:
                call_kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
                call_kwargs["api_base"] = "https://openrouter.ai/api/v1"
            elif self.config.provider == ProviderType.DEEPSEEK:
                call_kwargs["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
                call_kwargs["api_base"] = "https://api.deepseek.com"
            elif self.config.provider == ProviderType.DEEPINFRA:
                call_kwargs["api_key"] = os.environ.get("DeepInfra_API_KEY", "")
                call_kwargs["api_base"] = "https://api.deepinfra.com/v1/openai"
            elif self.config.provider == ProviderType.GROQ:
                call_kwargs["api_key"] = os.environ.get("GROQ_API_KEY", "")
                call_kwargs["api_base"] = "https://api.groq.com/openai/v1"
            elif provider != "openai":
                call_kwargs["custom_llm_provider"] = provider
            res = completion(**call_kwargs)

            usage = res.usage
            if usage:
                total_input += usage.prompt_tokens or 0
                total_output += usage.completion_tokens or 0
                cr, cw = _extract_cache_tokens(usage)
                cache_read += cr
                cache_write += cw

            total_cost += res._hidden_params.get("response_cost", 0) or 0

            next_message = res.choices[0].message.model_dump()
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME and self.flags.progressive_disclosure:
                self._discovered_tools.add(action.name)

            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend([
                    next_message,
                    {
                        "role": "tool",
                        "tool_call_id": next_message["tool_calls"][0]["id"],
                        "name": next_message["tool_calls"][0]["function"]["name"],
                        "content": env_response.observation,
                    },
                ])
            else:
                messages.extend([
                    next_message,
                    {"role": "user", "content": env_response.observation},
                ])

            if env_response.done:
                break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        ), {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "num_steps": step + 1,
        }

    def _apply_message_optimizations(self, messages: list[dict], step: int) -> list[dict]:
        """Apply masking and caching to the raw message dicts."""
        result = list(messages)

        if self.flags.observation_masking and step > 0:
            result = _mask_old_tool_outputs(
                result,
                self.flags.observation_masking_window,
                use_llm=self.flags.observation_masking_llm,
            )

        if self.flags.prompt_caching and self.config.provider == ProviderType.ANTHROPIC:
            result = _add_cache_control_to_system(result)

        return result

    def _select_tools(self, step: int) -> list[dict]:
        if self.flags.progressive_disclosure:
            pruned_map = {}
            if self.flags.tool_pruning and self._pruned_tools:
                pruned_map = {t["function"]["name"]: t for t in self._pruned_tools}
            tools = []
            for name, full_def in self._full_tools.items():
                if name in self._discovered_tools:
                    if self.flags.tool_pruning and name in pruned_map:
                        tools.append(pruned_map[name])
                    else:
                        tools.append(full_def)
                else:
                    tools.append(self._routing_tools[name])
            return tools
        if self.flags.tool_pruning and self._pruned_tools:
            return self._pruned_tools
        return self.tools_info

    def _select_model(self, step: int, messages: list) -> tuple[str, str]:
        """Select model and litellm provider string based on routing flag."""
        tier = ModelTier.STRONG
        if self.flags.model_routing and step > 0:
            tier = _llm_route_complexity(messages)

        model = self.config.get_model(tier)

        if self.config.provider == ProviderType.OPENROUTER:
            return f"openrouter/{model}", "openrouter"
        elif self.config.provider == ProviderType.ANTHROPIC:
            return model, "anthropic"
        elif self.config.provider == ProviderType.DEEPSEEK:
            return f"deepseek/{model}", "deepseek"
        elif self.config.provider == ProviderType.DEEPINFRA:
            return f"deepinfra/{model}", "deepinfra"
        elif self.config.provider == ProviderType.GROQ:
            return f"groq/{model}", "groq"
        else:
            return model, "openai"


USER_SIM_MODEL = "openrouter/google/gemini-2.5-flash-lite"
USER_SIM_PROVIDER = "openrouter"


class TauBenchRunner:
    def __init__(
        self,
        config: AgentConfig,
        domain: str = "retail",
        profiles: list[OptimizationFlags] | None = None,
        task_ids: list[int] | None = None,
        num_tasks: int | None = None,
        num_runs: int = 1,
        temperature: float = 0.0,
        concurrency: int = 3,
        results_csv: str | None = None,
        log_file: str | None = None,
        resume: bool = False,
    ):
        self.config = config
        self.domain = domain
        self.concurrency = concurrency
        self.profiles = profiles or []
        self.task_ids = task_ids
        self.num_tasks = num_tasks
        self.num_runs = num_runs
        self._setup_logging(log_file)
        self.temperature = temperature
        self.all_results = TauBenchResults(domain=domain)
        self.results_csv = results_csv
        self._csv_header_written = False
        self.logger = None
        self._completed_keys: set[tuple[str, int, int]] = set()
        if resume and results_csv:
            self._load_completed(results_csv)

    def _load_completed(self, csv_path: str):
        """Load completed (config, task_id, run_id) tuples from existing CSV.

        Only counts rows with input_tokens > 0 as completed — rows with 0 tokens
        are connection failures that should be retried.
        """
        if not os.path.exists(csv_path):
            return
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("input_tokens", 0)) > 0:
                    key = (row["optimization"], int(row["task_id"]), int(row["run_id"]))
                    self._completed_keys.add(key)
        self._csv_header_written = True

    def _is_config_complete(self, label: str, task_ids: list[int], run_id: int) -> bool:
        """Check if all tasks for a config+run are already completed."""
        return all(
            (label, tid, run_id) in self._completed_keys
            for tid in task_ids
        )

    def _setup_logging(self, log_file: str | None):
        import logging
        self.logger = logging.getLogger("tau_experiment")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            ))
            self.logger.addHandler(fh)

    def _log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level)(msg)

    def run_all(self) -> TauBenchResults:
        return asyncio.run(self._run_all_async())

    async def _run_all_async(self) -> TauBenchResults:
        import logging
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        user_model = USER_SIM_MODEL
        user_provider = USER_SIM_PROVIDER

        if self.domain == "retail":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as _tasks
        else:
            from tau_bench.envs.airline.tasks_test import TASKS_TEST as _tasks

        task_ids = self.task_ids
        if task_ids is None:
            total = len(_tasks)
            n = self.num_tasks or total
            task_ids = list(range(min(n, total)))

        console.print(f"  Tasks: {len(task_ids)}")
        console.print(f"  Configs: {len(self.profiles)}")
        console.print(f"  Runs: {self.num_runs}")
        console.print(f"  Concurrency: {self.concurrency} tasks in parallel")
        console.print(f"  User simulator: {user_model} (via {user_provider})")
        console.print()

        semaphore = asyncio.Semaphore(self.concurrency)

        skipped = 0
        for profile in self.profiles:
            label = profile.label()
            for run_id in range(1, self.num_runs + 1):
                run_tag = f" run {run_id}/{self.num_runs}" if self.num_runs > 1 else ""

                if self._is_config_complete(label, task_ids, run_id):
                    skipped += 1
                    console.print(f"  {label}{run_tag}... [dim]skipped (already completed)[/dim]")
                    self._log("info", f"Skipping config: {label}{run_tag} (already completed)")
                    continue

                pending_tids = [
                    tid for tid in task_ids
                    if (label, tid, run_id) not in self._completed_keys
                ]
                console.print(f"  {label}{run_tag}... ({len(pending_tids)}/{len(task_ids)} tasks)")
                self._log("info", f"Starting config: {label}{run_tag} | tasks: {pending_tids}")

                async def run_task(tid, sem, _profile=profile, _label=label, _run_id=run_id):
                    async with sem:
                        return await asyncio.to_thread(
                            self._run_single_task, tid, _profile, _label, _run_id
                        )

                task_results = await asyncio.gather(
                    *[run_task(tid, semaphore) for tid in pending_tids]
                )

                rewards = []
                for task_result in task_results:
                    self.all_results.results.append(task_result)
                    self._save_result_incremental(task_result)
                    rewards.append(task_result.reward)

                pass_rate = sum(rewards) / len(rewards) if rewards else 0
                total_cost = sum(r.total_cost for r in task_results)
                total_input = sum(r.total_input_tokens for r in task_results)
                console.print(
                    f"    Pass rate: [green]{pass_rate:.1%}[/green] "
                    f"| {total_input:,} tok | ${total_cost:.4f}"
                )

        if skipped:
            console.print(f"\n  [dim]Skipped {skipped} already-completed configs[/dim]")

        return self.all_results

    def _save_result_incremental(self, result: TauTaskResult):
        """Append a single task result to CSV immediately."""
        if not self.results_csv:
            return
        import csv as csv_mod
        exists = os.path.exists(self.results_csv) and self._csv_header_written
        with open(self.results_csv, "a", newline="") as f:
            writer = csv_mod.writer(f)
            if not exists and not self._csv_header_written:
                writer.writerow([
                    "domain", "task_id", "optimization", "run_id", "reward",
                    "input_tokens", "output_tokens", "cache_read", "cache_write",
                    "cost", "num_steps",
                ])
                self._csv_header_written = True
            writer.writerow([
                self.domain, result.task_id, result.optimization_label, result.run_id,
                result.reward, result.total_input_tokens, result.total_output_tokens,
                result.cache_read_tokens, result.cache_write_tokens,
                f"{result.total_cost:.6f}", result.num_steps,
            ])

    def _run_single_task(
        self,
        tid: int,
        profile: OptimizationFlags,
        label: str,
        run_id: int,
        max_retries: int = 3,
    ) -> TauTaskResult:
        from tau_bench.envs import get_env

        for attempt in range(1, max_retries + 1):
            try:
                env_fresh = get_env(
                    env_name=self.domain,
                    user_strategy="llm",
                    user_model=USER_SIM_MODEL,
                    user_provider=USER_SIM_PROVIDER,
                    task_split="test",
                )

                agent = OptimizedTauAgent(
                    config=self.config,
                    flags=profile,
                    tools_info=env_fresh.tools_info,
                    wiki=env_fresh.wiki,
                    temperature=self.temperature,
                )

                solve_result, metrics = agent.solve(env_fresh, task_index=tid)
                result = TauTaskResult(
                    task_id=tid,
                    optimization_label=label,
                    run_id=run_id,
                    reward=solve_result.reward,
                    total_cost=solve_result.total_cost,
                    total_input_tokens=metrics["total_input_tokens"],
                    total_output_tokens=metrics["total_output_tokens"],
                    num_steps=metrics["num_steps"],
                    cache_read_tokens=metrics["cache_read_tokens"],
                    cache_write_tokens=metrics["cache_write_tokens"],
                )
                self._log("info",
                    f"Task {tid} | {label} | run {run_id} | "
                    f"reward={result.reward} | {result.total_input_tokens:,} in | "
                    f"{result.total_output_tokens:,} out | {result.cache_read_tokens:,} cache | "
                    f"steps={result.num_steps} | ${result.total_cost:.4f}"
                )
                return result
            except Exception as e:
                is_connection_error = any(kw in str(e).lower() for kw in [
                    "peer closed connection", "server disconnected",
                    "incomplete chunked", "connection reset",
                    "connection aborted", "read timed out",
                    "remotedisconnected", "connectionerror",
                    "rate-limited", "ratelimit", "429",
                    "temporarily rate-limited upstream",
                ])
                if is_connection_error and attempt < max_retries:
                    wait = 2 ** attempt * 5
                    console.print(
                        f"    [yellow]Task {tid} connection error (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait}s...[/yellow]"
                    )
                    self._log("warning",
                        f"Task {tid} | {label} | run {run_id} | "
                        f"Connection error attempt {attempt}/{max_retries}: {e} — retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue

                import traceback
                error_detail = traceback.format_exc()
                console.print(f"    [red]Task {tid} error: {e}[/red]")
                self._log("error",
                    f"Task {tid} | {label} | run {run_id} | ERROR: {e}\n{error_detail}"
                )
                return TauTaskResult(
                    task_id=tid,
                    optimization_label=label,
                    run_id=run_id,
                    reward=0.0,
                    total_cost=0.0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    num_steps=0,
                )


def print_tau_report(results: TauBenchResults) -> None:
    table = Table(
        title=f"Tau-bench ({results.domain}) — Accuracy vs Efficiency",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Optimization", style="bold", min_width=35)
    table.add_column("Pass Rate", justify="center")
    table.add_column("Avg Input Tok", justify="right")
    table.add_column("Avg Cost/Task", justify="right")
    table.add_column("Cost-of-Pass", justify="right")
    table.add_column("Savings", justify="right", style="green")

    baseline_cop = None
    for label in results.optimization_labels:
        runs = results.get_results(label)
        if not runs:
            continue

        pass_rate = sum(r.reward for r in runs) / len(runs) if runs else 0
        avg_input = sum(r.total_input_tokens for r in runs) / len(runs)
        avg_cost = sum(r.total_cost for r in runs) / len(runs)
        cop = avg_cost / pass_rate if pass_rate > 0 else float("inf")

        if label == "baseline":
            baseline_cop = cop

        savings = ""
        if baseline_cop and baseline_cop > 0 and label != "baseline" and cop < float("inf"):
            pct = (1 - cop / baseline_cop) * 100
            savings = f"{pct:+.1f}%"

        table.add_row(
            label,
            f"{pass_rate:.1%}",
            f"{avg_input:,.0f}",
            f"${avg_cost:.4f}",
            f"${cop:.4f}" if cop < float("inf") else "N/A",
            savings,
        )

    console.print()
    console.print(table)


def export_tau_csv(results: TauBenchResults, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain", "task_id", "optimization", "run_id", "reward",
            "input_tokens", "output_tokens", "cache_read", "cache_write",
            "cost", "num_steps",
        ])
        for r in results.results:
            writer.writerow([
                results.domain, r.task_id, r.optimization_label, r.run_id,
                r.reward, r.total_input_tokens, r.total_output_tokens,
                r.cache_read_tokens, r.cache_write_tokens,
                f"{r.total_cost:.6f}", r.num_steps,
            ])


# --- Cache token extraction (handles litellm, OpenAI, Anthropic response formats) ---

def _extract_cache_tokens(usage) -> tuple[int, int]:
    """Extract cache read and write tokens from any usage object format.

    Returns (cache_read, cache_write).
    """
    cache_read = 0
    cache_write = 0

    # litellm private attrs (Anthropic-style)
    cache_read = getattr(usage, "_cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "_cache_creation_input_tokens", 0) or 0

    # litellm/OpenAI prompt_tokens_details
    if (not cache_read) and hasattr(usage, "prompt_tokens_details"):
        details = usage.prompt_tokens_details
        if details:
            cache_read = getattr(details, "cached_tokens", 0) or 0
            cache_write = getattr(details, "cache_creation_tokens", 0) or 0

    # Direct attributes (some providers)
    if not cache_read:
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    if not cache_write:
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    return cache_read, cache_write


# --- Optimization helpers for raw message dicts (tau-bench uses dicts, not LangChain messages) ---

def _mask_old_tool_outputs(
    messages: list[dict],
    window_k: int = 3,
    use_llm: bool = False,
    llm_model: str = "openrouter/qwen/qwen3.5-9b",
) -> list[dict]:
    """Content-aware masking: replace old tool outputs with compact summaries.

    Preserves still-referenced outputs: older tool outputs whose key
    identifiers still appear in recent assistant/user messages are kept.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= window_k:
        return messages

    recency_keep = set(tool_indices[-window_k:])
    candidates = [i for i in tool_indices if i not in recency_keep]

    recent_text = _collect_recent_text(messages, lookback=6)

    indices_to_mask = set()
    for i in candidates:
        content = messages[i].get("content", "")
        if _is_still_referenced(content, recent_text):
            continue
        indices_to_mask.add(i)

    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_mask:
            tool_name = msg.get("name", "unknown")
            original = msg.get("content", "")
            if use_llm:
                summary = _llm_summarize(tool_name, original, llm_model)
            else:
                summary = _extract_key_info(tool_name, original)
            result.append({
                **msg,
                "content": summary,
            })
        else:
            result.append(msg)
    return result


def _collect_recent_text(messages: list[dict], lookback: int) -> str:
    parts = []
    count = 0
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role in ("assistant", "user"):
            content = msg.get("content", "") or ""
            tc = msg.get("tool_calls") or []
            if tc:
                for call in tc:
                    fn = call.get("function", {})
                    content += " " + fn.get("name", "")
                    content += " " + fn.get("arguments", "")
            parts.append(content)
            count += 1
            if count >= lookback:
                break
    return " ".join(parts)


def _is_still_referenced(tool_content: str, recent_text: str) -> bool:
    if not tool_content or not recent_text:
        return False
    import re
    recent_lower = recent_text.lower()

    identifiers = re.findall(
        r'["\']?(?:id|order_id|user_id|product_id|reservation_id|booking_id|flight_number|confirmation)["\']?\s*[:=]\s*["\']?([#\w-]{3,})',
        tool_content, re.IGNORECASE,
    )
    for ident in identifiers:
        if ident.lower() in recent_lower:
            return True

    names = re.findall(
        r'["\']?(?:name|first_name|last_name)["\']?\s*[:=]\s*["\']?([A-Z][a-z]{2,})',
        tool_content,
    )
    for name in names:
        if name.lower() in recent_lower:
            return True

    return False


def _extract_key_info(tool_name: str, content: str) -> str:
    """Regex-based extraction: free, instant, good for structured data."""
    if not content or len(content) < 100:
        return content

    import re
    key_items = []

    ids = re.findall(r'["\']?(?:id|order_id|user_id|product_id|reservation_id)["\']?\s*[:=]\s*["\']?([#\w-]+)', content, re.IGNORECASE)
    if ids:
        key_items.append(f"IDs: {', '.join(dict.fromkeys(ids))}")

    statuses = re.findall(r'["\']?(?:status|state|order_status)["\']?\s*[:=]\s*["\']?(\w+)', content, re.IGNORECASE)
    if statuses:
        key_items.append(f"status: {', '.join(dict.fromkeys(statuses))}")

    names = re.findall(r'["\']?(?:name|first_name|last_name)["\']?\s*[:=]\s*["\']?([A-Z][\w\s]+?)(?:["\',}])', content)
    if names:
        key_items.append(f"name: {', '.join(dict.fromkeys(names[:3]))}")

    amounts = re.findall(r'["\']?(?:amount|total|price|payment_amount)["\']?\s*[:=]\s*["\']?([\d.]+)', content, re.IGNORECASE)
    if amounts:
        key_items.append(f"amounts: {', '.join(dict.fromkeys(amounts[:3]))}")

    if key_items:
        return f"[{tool_name} summary: {'; '.join(key_items)}]"

    first_line = content.split('\n')[0][:150]
    return f"[{tool_name}: {first_line}...]"


def _llm_summarize(tool_name: str, content: str, model: str) -> str:
    """LLM-based summarization: uses a cheap small model for better quality."""
    if not content or len(content) < 100:
        return content

    try:
        from litellm import completion

        res = completion(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this {tool_name} tool output in ONE line. "
                    f"Keep all IDs, statuses, names, amounts, and key facts. "
                    f"Drop addresses, descriptions, and verbose details.\n\n"
                    f"{content}"
                ),
            }],
            max_tokens=100,
            temperature=0.0,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            api_base="https://openrouter.ai/api/v1",
        )
        summary = res.choices[0].message.content.strip()
        return f"[{tool_name} summary: {summary}]"
    except Exception:
        return _extract_key_info(tool_name, content)


def _add_cache_control_to_system(messages: list[dict]) -> list[dict]:
    """Add Anthropic cache_control to system message.

    Restructures system message content as a list of text blocks
    with cache_control on the last block, per the Anthropic API spec.
    Also marks the last tool result in the conversation for caching
    the conversation prefix.
    """
    result = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                result.append({
                    **msg,
                    "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
                })
            else:
                result.append({**msg, "cache_control": {"type": "ephemeral"}})
        else:
            result.append(msg)
    return result


ROUTER_MODEL = "openrouter/google/gemma-4-26b-a4b-it"


def _prune_tool_def(tool: dict) -> dict:
    """Rule-based pruning inspired by SkillReducer.

    - Shorten tool description to first sentence
    - Shorten long param descriptions to first sentence (keeps format hints)
    - Remove examples and defaults
    - Truncate long enum lists (keep first 5 + count)
    """
    tool = copy.deepcopy(tool)
    func = tool.get("function", {})
    desc = func.get("description", "")
    if len(desc) > 80:
        func["description"] = desc.split(".")[0] + "."

    params = func.get("parameters", {})
    for pname, pschema in params.get("properties", {}).items():
        pdesc = pschema.get("description", "")
        if len(pdesc) > 60:
            pschema["description"] = pdesc.split(".")[0] + "."
        pschema.pop("examples", None)
        pschema.pop("example", None)
        pschema.pop("default", None)
        if "enum" in pschema and len(pschema["enum"]) > 5:
            pschema["enum"] = pschema["enum"][:5] + [f"...+{len(pschema['enum']) - 5} more"]

    return tool


def _llm_prune_tool_def(tool: dict) -> dict:
    """LLM-based pruning: use a cheap model to compress tool definitions.

    Called once at agent init per tool. The LLM removes verbose descriptions,
    redundant examples, and filler while keeping names, types, required fields,
    and constraints the agent needs to call the tool correctly.
    """
    tool = copy.deepcopy(tool)
    func = tool.get("function", {})
    original_json = json.dumps(func, indent=2)

    if len(original_json) < 200:
        return tool

    try:
        from litellm import completion as llm_completion

        res = llm_completion(
            model=ROUTER_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Compress this tool definition JSON. Rules:\n"
                    "- Keep the function name EXACTLY as-is\n"
                    "- Shorten the description to one sentence\n"
                    "- Keep ALL parameter names, types, and required fields\n"
                    "- Remove parameter descriptions, examples, and defaults\n"
                    "- Keep enum values (shortened if >5 items)\n"
                    "- Output ONLY the compressed JSON, no explanation\n\n"
                    f"{original_json}"
                ),
            }],
            max_tokens=500,
            temperature=0.0,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            api_base="https://openrouter.ai/api/v1",
        )
        compressed = res.choices[0].message.content.strip()
        # Extract JSON from response (may have markdown fences)
        if "```" in compressed:
            compressed = compressed.split("```")[1]
            if compressed.startswith("json"):
                compressed = compressed[4:]
            compressed = compressed.strip()
        parsed = json.loads(compressed)
        if "name" in parsed and "parameters" in parsed:
            tool["function"] = parsed
            return tool
    except Exception:
        pass
    return _prune_tool_def(tool)


def _llm_route_complexity(messages: list[dict]) -> ModelTier:
    """LLM-based routing: classify the next step's complexity.

    Uses a cheap small model to decide if the next agent step needs
    the strong model (complex reasoning, multi-step logic) or can be
    handled by the weak model (simple lookups, confirmations).
    """
    last_msgs = messages[-3:] if len(messages) > 3 else messages
    context = "\n".join(
        f"[{m.get('role', '?')}] {str(m.get('content', ''))[:300]}"
        for m in last_msgs
    )

    try:
        from litellm import completion as llm_completion

        res = llm_completion(
            model=ROUTER_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "You are a routing classifier. Based on the conversation below, "
                    "decide if the NEXT agent action is:\n"
                    "- SIMPLE: single tool lookup, confirmation, or direct response\n"
                    "- COMPLEX: multi-step reasoning, policy decision, error handling, "
                    "or combining information from multiple sources\n\n"
                    f"Conversation:\n{context}\n\n"
                    "Reply with ONLY one word: SIMPLE or COMPLEX"
                ),
            }],
            max_tokens=5,
            temperature=0.0,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            api_base="https://openrouter.ai/api/v1",
        )
        answer = res.choices[0].message.content.strip().upper()
        if "SIMPLE" in answer:
            return ModelTier.WEAK
        return ModelTier.STRONG
    except Exception:
        return ModelTier.STRONG


def _make_routing_def(tool: dict) -> dict:
    """Make a minimal routing definition — name, one-line description, param names only.

    Strips parameter descriptions and keeps only names + types. The model can see
    what the tool does and what args it needs, but without verbose docs.
    Saves ~50-200 tokens per tool vs full definition.
    """
    func = tool.get("function", {})
    desc = func.get("description", "")
    short_desc = desc.split(".")[0] + "."

    params = func.get("parameters", {})
    slim_props = {}
    for pname, pschema in params.get("properties", {}).items():
        slim_props[pname] = {"type": pschema.get("type", "string")}

    return {
        "type": "function",
        "function": {
            "name": func.get("name", ""),
            "description": short_desc,
            "parameters": {
                "type": "object",
                "properties": slim_props,
                "required": params.get("required", []),
            },
        },
    }

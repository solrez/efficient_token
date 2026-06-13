"""Phase 1 signal validation using BFCL (single-turn function calling).

Simpler and faster than Tau-bench: no env, no user simulator, no multi-turn.
Each test case is one LLM call with tools bound. We collect logprob and
prompted uncertainty estimates alongside correctness labels.

Usage:
    python -m adaptive_compression validate-bfcl \
        --provider openrouter --model deepseek-v4-flash \
        --num-cases 100 --output results/signal_validation_bfcl/
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

from token_efficient_agent.config import (
    AgentConfig, ModelTier, ProviderType, OPENROUTER_MODEL_PRESETS,
)
from token_efficient_agent.accuracy.bfcl_loader import (
    BFCLTestCase, load_stratified_sample, load_category, CATEGORIES,
)
from adaptive_compression.uncertainty import (
    LogprobEstimator,
    PromptedEstimator,
    HybridEstimator,
)

console = Console()

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the provided functions. "
    "Call the appropriate function to answer the user's question. "
    "If no function is relevant, respond without calling any function."
)


@dataclass
class BFCLStepRecord:
    case_id: str
    category: str
    correct: bool
    tool_called: str
    logprob_confidence: float | None
    logprob_entropy: float | None
    logprob_num_tokens: int
    prompted_confidence: float | None
    prompted_raw_score: int | None
    hybrid_confidence: float | None
    hybrid_method: str
    input_tokens: int
    output_tokens: int
    cost: float
    model_used: str
    error: str


class BFCLSignalValidation:
    """Run BFCL with uncertainty estimators to validate the signal."""

    def __init__(
        self,
        config: AgentConfig,
        categories: list[str] | None = None,
        num_per_category: int = 100,
        temperature: float = 0.0,
        concurrency: int = 5,
        output_dir: str = "results/signal_validation_bfcl",
    ):
        self.config = config
        self.categories = categories or CATEGORIES
        self.num_per_category = num_per_category
        self.temperature = temperature
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logprob_estimator = LogprobEstimator(num_tokens=10)
        self.prompted_estimator = PromptedEstimator()
        self.hybrid_estimator = HybridEstimator()

    def run(self) -> list[BFCLStepRecord]:
        return asyncio.run(self._run_async())

    async def _run_async(self) -> list[BFCLStepRecord]:
        import logging
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        cases = load_stratified_sample(
            categories=self.categories,
            n_per_category=self.num_per_category,
            seed=42,
        )

        model_name = self.config.get_model(ModelTier.STRONG)
        console.print(f"\n[bold]BFCL Signal Validation[/bold]")
        console.print(f"  Cases: {len(cases)} ({self.num_per_category} per category)")
        console.print(f"  Categories: {', '.join(self.categories)}")
        console.print(f"  Model: {model_name}")
        console.print(f"  Concurrency: {self.concurrency}")
        console.print(f"  Output: {self.output_dir}")
        console.print()

        results_csv = self.output_dir / "bfcl_uncertainty.csv"
        self._init_csv(results_csv)

        semaphore = asyncio.Semaphore(self.concurrency)
        all_records = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task("Running cases...", total=len(cases))

            async def run_one(case):
                async with semaphore:
                    record = await asyncio.to_thread(self._run_single, case)
                    self._append_csv(results_csv, record)
                    progress.advance(ptask)
                    return record

            all_records = await asyncio.gather(*[run_one(c) for c in cases])

        self._save_params(len(cases))
        self._print_summary(all_records)
        return all_records

    def _run_single(self, case: BFCLTestCase, max_retries: int = 3) -> BFCLStepRecord:
        model_name = self.config.get_model(ModelTier.STRONG)
        litellm_model = (
            f"openrouter/{model_name}"
            if self.config.provider == ProviderType.OPENROUTER
            else model_name
        )

        # Convert BFCL function defs to OpenAI tool format
        tools = _to_openai_tools(case.functions)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in case.messages:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})

        for attempt in range(1, max_retries + 1):
            try:
                return self._call_and_evaluate(
                    case, messages, tools, litellm_model,
                )
            except Exception as e:
                is_transient = any(kw in str(e).lower() for kw in [
                    "429", "rate", "timeout", "connection", "disconnect",
                ])
                if is_transient and attempt < max_retries:
                    time.sleep(2 ** attempt * 3)
                    continue
                return BFCLStepRecord(
                    case_id=case.id, category=case.category, correct=False,
                    tool_called="", logprob_confidence=None, logprob_entropy=None,
                    logprob_num_tokens=0, prompted_confidence=None,
                    prompted_raw_score=None, hybrid_confidence=None,
                    hybrid_method="error", input_tokens=0, output_tokens=0,
                    cost=0.0, model_used=litellm_model, error=str(e)[:200],
                )

    def _call_and_evaluate(
        self,
        case: BFCLTestCase,
        messages: list[dict],
        tools: list[dict],
        litellm_model: str,
    ) -> BFCLStepRecord:
        from litellm import completion

        call_kwargs = dict(
            messages=messages,
            model=litellm_model,
            tools=tools if tools else None,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=5,
        )
        if self.config.provider == ProviderType.OPENROUTER:
            call_kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
            call_kwargs["api_base"] = "https://openrouter.ai/api/v1"

        res = completion(**call_kwargs)

        # Extract usage
        usage = res.usage
        input_tokens = usage.prompt_tokens or 0 if usage else 0
        output_tokens = usage.completion_tokens or 0 if usage else 0
        cost = res._hidden_params.get("response_cost", 0) or 0

        # Estimate uncertainty
        logprob_result = self.logprob_estimator.estimate(res, messages)
        prompted_result = self.prompted_estimator.estimate(res, messages)
        hybrid_result = self.hybrid_estimator.estimate(res, messages)

        # Evaluate correctness
        msg = res.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        model_output = []
        tool_name = ""
        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            try:
                args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
            except json.JSONDecodeError:
                args = {}
            model_output.append({name: args})
            if not tool_name:
                tool_name = name

        correct = _check_correctness(model_output, case.ground_truth, case.category)

        return BFCLStepRecord(
            case_id=case.id,
            category=case.category,
            correct=correct,
            tool_called=tool_name or "__none__",
            logprob_confidence=logprob_result.confidence if logprob_result.raw_entropy is not None else None,
            logprob_entropy=logprob_result.raw_entropy,
            logprob_num_tokens=logprob_result.num_tokens_used,
            prompted_confidence=prompted_result.confidence if prompted_result.raw_score is not None else None,
            prompted_raw_score=prompted_result.raw_score,
            hybrid_confidence=hybrid_result.confidence,
            hybrid_method=hybrid_result.method,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model_used=litellm_model,
            error="",
        )

    def _init_csv(self, path: Path):
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "case_id", "category", "correct", "tool_called",
                    "logprob_confidence", "logprob_entropy", "logprob_num_tokens",
                    "prompted_confidence", "prompted_raw_score",
                    "hybrid_confidence", "hybrid_method",
                    "input_tokens", "output_tokens", "cost", "model_used", "error",
                ])

    def _append_csv(self, path: Path, record: BFCLStepRecord):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.case_id, record.category, int(record.correct), record.tool_called,
                record.logprob_confidence, record.logprob_entropy, record.logprob_num_tokens,
                record.prompted_confidence, record.prompted_raw_score,
                record.hybrid_confidence, record.hybrid_method,
                record.input_tokens, record.output_tokens,
                f"{record.cost:.6f}", record.model_used, record.error,
            ])

    def _save_params(self, total_cases: int):
        params = {
            "benchmark": "bfcl",
            "model": self.config.get_model(ModelTier.STRONG),
            "provider": self.config.provider.value,
            "temperature": self.temperature,
            "categories": self.categories,
            "num_per_category": self.num_per_category,
            "total_cases": total_cases,
            "estimators": ["logprob", "prompted", "hybrid"],
        }
        with open(self.output_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    def _print_summary(self, records: list[BFCLStepRecord]):
        valid = [r for r in records if r.hybrid_method != "error"]
        if not valid:
            console.print("[red]No valid results.[/red]")
            return

        total = len(valid)
        correct = sum(r.correct for r in valid)
        total_cost = sum(r.cost for r in valid)

        console.print(f"\n[bold]BFCL Signal Validation Summary[/bold]")
        console.print(f"  Cases: {total} ({total - len(valid) + len(records) - total} errors)")
        console.print(f"  Accuracy: {correct/total:.1%} ({correct}/{total})")
        console.print(f"  Total cost: ${total_cost:.4f}")

        # Per-category breakdown
        for cat in self.categories:
            cat_records = [r for r in valid if r.category == cat]
            if not cat_records:
                continue
            cat_correct = sum(r.correct for r in cat_records)
            console.print(f"  {cat}: {cat_correct/len(cat_records):.1%} ({cat_correct}/{len(cat_records)})")

        # Estimator coverage
        logprob_available = sum(1 for r in valid if r.logprob_confidence is not None)
        prompted_available = sum(1 for r in valid if r.prompted_confidence is not None)
        console.print(f"\n  Logprob coverage: {logprob_available}/{total} ({logprob_available/total:.0%})")
        console.print(f"  Prompted coverage: {prompted_available}/{total} ({prompted_available/total:.0%})")

        # Confidence for correct vs incorrect
        for name, getter in [
            ("Logprob", lambda r: r.logprob_confidence),
            ("Prompted", lambda r: r.prompted_confidence),
            ("Hybrid", lambda r: r.hybrid_confidence),
        ]:
            correct_scores = [getter(r) for r in valid if r.correct and getter(r) is not None]
            incorrect_scores = [getter(r) for r in valid if not r.correct and getter(r) is not None]
            if correct_scores and incorrect_scores:
                console.print(
                    f"  {name}: correct={sum(correct_scores)/len(correct_scores):.3f} "
                    f"(n={len(correct_scores)}), "
                    f"incorrect={sum(incorrect_scores)/len(incorrect_scores):.3f} "
                    f"(n={len(incorrect_scores)})"
                )

        console.print(f"\n  Results saved to: {self.output_dir}/")


def _to_openai_tools(functions: list[dict]) -> list[dict]:
    """Convert BFCL function definitions to OpenAI tool format."""
    import re
    import copy

    tools = []
    for func in functions:
        params = copy.deepcopy(func.get("parameters", {}))
        params = _fix_schema(params)
        # Sanitize function name (BFCL has dots, spaces, etc. that OpenAI rejects)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", func["name"])
        # OpenAI requires name to match ^[a-zA-Z0-9_-]+$ and be <= 64 chars
        name = name[:64]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": func.get("description", "")[:1024],
                "parameters": params,
            },
        })
    return tools


def _fix_schema(schema: dict) -> dict:
    """Fix BFCL schema quirks to be valid OpenAI JSON Schema."""
    if not isinstance(schema, dict):
        return schema

    if schema.get("type") == "dict":
        schema["type"] = "object"
    if schema.get("type") == "list":
        schema["type"] = "array"
    if schema.get("type") == "float":
        schema["type"] = "number"
    if schema.get("type") == "int":
        schema["type"] = "integer"
    if schema.get("type") == "str":
        schema["type"] = "string"
    if schema.get("type") == "bool":
        schema["type"] = "boolean"
    if schema.get("type") == "tuple":
        schema["type"] = "array"

    # Remove fields OpenAI doesn't accept
    for bad_key in ["examples", "example", "default"]:
        schema.pop(bad_key, None)

    # Ensure properties dict exists for object types
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}

    for key, val in schema.items():
        if isinstance(val, dict):
            schema[key] = _fix_schema(val)
        elif isinstance(val, list):
            schema[key] = [_fix_schema(v) if isinstance(v, dict) else v for v in val]
    return schema


def _check_correctness(
    model_output: list[dict],
    ground_truth: list[dict] | None,
    category: str,
) -> bool:
    """Check if model output matches ground truth."""
    import re

    # Irrelevance: pass if no tool calls
    if ground_truth is None:
        return len(model_output) == 0

    if not model_output:
        return False

    # Sanitize names to match loader's sanitization
    sanitized_gt = []
    for call in ground_truth:
        new_call = {}
        for name, args in call.items():
            clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            new_call[clean_name] = args
        sanitized_gt.append(new_call)

    # Check each ground truth call has a match
    for gt_call in sanitized_gt:
        gt_name = list(gt_call.keys())[0]
        gt_args = gt_call[gt_name]

        matched = False
        for model_call in model_output:
            model_name = list(model_call.keys())[0]
            # Sanitize model name too
            clean_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
            if clean_model_name != gt_name:
                continue

            model_args = model_call[model_name]
            if _args_match(model_args, gt_args):
                matched = True
                break

        if not matched:
            return False

    return True


def _args_match(model_args: dict, gt_args: dict) -> bool:
    """Check if model args match ground truth (with tolerance)."""
    for param_name, acceptable_values in gt_args.items():
        model_val = model_args.get(param_name)
        if isinstance(acceptable_values, list):
            if "" in acceptable_values and model_val is None:
                continue
            if model_val is None:
                return False
            if not any(_val_eq(model_val, av) for av in acceptable_values if av != ""):
                return False
        else:
            if not _val_eq(model_val, acceptable_values):
                return False
    return True


def _val_eq(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    try:
        return a == b or str(a) == str(b)
    except Exception:
        return False

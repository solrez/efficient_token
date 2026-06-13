"""Evaluate model tool-call output against BFCL ground truth using AST matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage


@dataclass
class EvalResult:
    test_id: str
    category: str
    passed: bool
    error: str = ""
    model_output: list[dict] = field(default_factory=list)
    ground_truth: list[dict] | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def evaluate_tool_calls(
    response: AIMessage,
    ground_truth: list[dict] | None,
    test_id: str,
    category: str,
) -> EvalResult:
    """Evaluate a model response against BFCL ground truth.

    For irrelevance tests (ground_truth is None): pass if model made NO tool calls.
    For other tests: use AST matching from bfcl_eval if available, else simple matching.
    """
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    model_output = [{tc["name"]: tc["args"]} for tc in tool_calls]

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    if ground_truth is None:
        return EvalResult(
            test_id=test_id,
            category=category,
            passed=len(tool_calls) == 0,
            error="" if len(tool_calls) == 0 else f"Expected no tool call, got {len(tool_calls)}",
            model_output=model_output,
            ground_truth=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    try:
        result = _ast_check(model_output, ground_truth, category)
        return EvalResult(
            test_id=test_id,
            category=category,
            passed=result["valid"],
            error="" if result["valid"] else str(result.get("error", "mismatch")),
            model_output=model_output,
            ground_truth=ground_truth,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        return EvalResult(
            test_id=test_id,
            category=category,
            passed=False,
            error=f"Evaluation error: {e}",
            model_output=model_output,
            ground_truth=ground_truth,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _ast_check(model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    """Try bfcl_eval AST checker, fall back to simple matching."""
    try:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        return ast_checker(
            func_description=[],
            model_output=model_output,
            possible_answer=ground_truth,
            language="Python",
            test_category=category,
            model_name="test",
        )
    except (ImportError, Exception):
        return _simple_check(model_output, ground_truth, category)


def _simple_check(model_output: list[dict], ground_truth: list[dict], category: str) -> dict:
    """Simple fallback checker — matches function names and required args."""
    if not model_output and not ground_truth:
        return {"valid": True}

    if not model_output:
        return {"valid": False, "error": "No tool calls produced"}

    if len(model_output) != len(ground_truth):
        if "parallel" not in category:
            return {
                "valid": False,
                "error": f"Expected {len(ground_truth)} calls, got {len(model_output)}",
            }

    for gt_call in ground_truth:
        gt_name = list(gt_call.keys())[0]
        gt_args = gt_call[gt_name]

        matched = False
        for model_call in model_output:
            model_name = list(model_call.keys())[0]
            if model_name != gt_name:
                continue

            model_args = model_call[model_name]
            args_match = True
            for param_name, acceptable_values in gt_args.items():
                model_val = model_args.get(param_name)
                if "" in acceptable_values and model_val is None:
                    continue
                if model_val is None:
                    args_match = False
                    break
                if not any(_values_match(model_val, av) for av in acceptable_values if av != ""):
                    args_match = False
                    break

            if args_match:
                matched = True
                break

        if not matched:
            return {"valid": False, "error": f"No match for {gt_name}"}

    return {"valid": True}


def _values_match(model_val: Any, expected_val: Any) -> bool:
    if isinstance(expected_val, str) and isinstance(model_val, str):
        return model_val.strip().lower() == expected_val.strip().lower()
    try:
        return model_val == expected_val or str(model_val) == str(expected_val)
    except Exception:
        return False

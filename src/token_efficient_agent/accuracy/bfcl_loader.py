"""Load BFCL test data and convert to LangChain-compatible format."""
from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "bfcl"

CATEGORIES = ["simple", "multiple", "parallel", "irrelevance"]


@dataclass
class BFCLTestCase:
    id: str
    category: str
    messages: list[dict]
    functions: list[dict]
    ground_truth: list[dict] | None
    langchain_tools: list[StructuredTool] = field(default_factory=list)


def load_category(category: str, limit: int | None = None) -> list[BFCLTestCase]:
    """Load BFCL test cases for a category."""
    prompt_file = DATA_DIR / "prompts" / f"{category}.json"
    answer_file = DATA_DIR / "answers" / f"{category}.json"

    prompts = _load_jsonl(prompt_file)
    answers = {}
    if answer_file.exists() and answer_file.stat().st_size > 0:
        try:
            for entry in _load_jsonl(answer_file):
                answers[entry["id"]] = entry.get("ground_truth")
        except (json.JSONDecodeError, KeyError):
            pass

    cases = []
    for prompt in prompts:
        if limit and len(cases) >= limit:
            break

        tc = BFCLTestCase(
            id=prompt["id"],
            category=category,
            messages=prompt["question"][0],
            functions=prompt["function"],
            ground_truth=answers.get(prompt["id"]),
        )
        tc.langchain_tools = [_bfcl_func_to_tool(f) for f in tc.functions]
        tc.ground_truth = _sanitize_ground_truth_names(tc.ground_truth)
        cases.append(tc)

    return cases


def load_all(categories: list[str] | None = None, limit_per_category: int | None = None) -> list[BFCLTestCase]:
    """Load test cases from multiple categories."""
    cats = categories or CATEGORIES
    cases = []
    for cat in cats:
        cases.extend(load_category(cat, limit=limit_per_category))
    return cases


def load_stratified_sample(
    categories: list[str] | None = None,
    n_per_category: int = 100,
    seed: int = 42,
) -> list[BFCLTestCase]:
    """Load a fixed stratified random sample for reproducible cross-validation.

    Uses the same seed across all calls so every config/run tests the same cases.
    """
    import random
    cats = categories or CATEGORIES
    cases = []
    for cat in cats:
        all_cases = load_category(cat)
        rng = random.Random(seed + hash(cat))
        n = min(n_per_category, len(all_cases))
        sampled = rng.sample(all_cases, n)
        cases.extend(sampled)
    return cases


def _sanitize_ground_truth_names(ground_truth: list[dict] | None) -> list[dict] | None:
    """Sanitize function names in ground truth to match tool name sanitization."""
    if ground_truth is None:
        return None
    import re
    sanitized = []
    for call in ground_truth:
        new_call = {}
        for name, args in call.items():
            clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            new_call[clean_name] = args
        sanitized.append(new_call)
    return sanitized


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _bfcl_func_to_tool(func_def: dict) -> StructuredTool:
    """Convert a BFCL function definition to a LangChain StructuredTool."""
    params = _fix_schema(copy.deepcopy(func_def.get("parameters", {})))
    raw_name = func_def["name"]
    import re
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
    description = func_def.get("description", raw_name)

    pydantic_fields = {}
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    for pname, pschema in properties.items():
        py_type = _schema_type_to_python(pschema.get("type", "string"))
        default = ... if pname in required else None
        field_desc = pschema.get("description", "")
        pydantic_fields[pname] = (py_type, default) if default is ... else (py_type | None, default)

    if not pydantic_fields:
        pydantic_fields["_placeholder"] = (str | None, None)

    model = create_model(f"{name}_input", **pydantic_fields)

    def _mock_func(**kwargs):
        return f"[mock] {name} called with {kwargs}"

    tool = StructuredTool.from_function(
        func=_mock_func,
        name=name,
        description=description,
        args_schema=model,
    )
    return tool


def _fix_schema(schema: dict) -> dict:
    """Fix BFCL schema quirks — 'dict' -> 'object', nested fixes."""
    if isinstance(schema, dict):
        if schema.get("type") == "dict":
            schema["type"] = "object"
        for key, val in schema.items():
            if isinstance(val, dict):
                schema[key] = _fix_schema(val)
            elif isinstance(val, list):
                schema[key] = [_fix_schema(v) if isinstance(v, dict) else v for v in val]
    return schema


def _schema_type_to_python(type_str: str) -> type:
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "dict": dict,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "list": list,
        "tuple": tuple,
    }
    return mapping.get(type_str, str)

"""Adaptive Tau-bench agent: uses uncertainty-driven compression controller.

Instead of static optimization flags, this agent dynamically adjusts
compression at each step based on the logprob uncertainty signal.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm
litellm.suppress_debug_info = True

from token_efficient_agent.config import (
    AgentConfig, OptimizationFlags, ModelTier, ProviderType,
    INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK,
)
from adaptive_compression.uncertainty import LogprobEstimator
from adaptive_compression.controller import (
    ThresholdController, CompressionProfile, ProfileName, PROFILES,
)


@dataclass
class AdaptiveStepLog:
    task_id: int
    step: int
    tool_called: str
    confidence: float | None
    profile_selected: str
    mask_window: int
    pruning_used: bool
    progressive_used: bool
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class AdaptiveTaskResult:
    task_id: int
    reward: float
    total_steps: int
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    step_logs: list[AdaptiveStepLog] = field(default_factory=list)
    controller_stats: dict = field(default_factory=dict)


class AdaptiveTauAgent:
    """Tau-bench agent with uncertainty-driven adaptive compression."""

    def __init__(
        self,
        config: AgentConfig,
        tools_info: list[dict],
        wiki: str,
        temperature: float = 0.0,
        controller: ThresholdController | None = None,
    ):
        self.config = config
        self.tools_info = tools_info
        self.wiki = wiki
        self.temperature = temperature
        self.controller = controller or ThresholdController()
        self.estimator = LogprobEstimator()

        self._full_tools = {t["function"]["name"]: t for t in tools_info}
        self._pruned_tools = {t["function"]["name"]: _prune_tool_def(t) for t in tools_info}
        self._routing_tools = {t["function"]["name"]: _make_routing_def(t) for t in tools_info}
        self._discovered_tools: set[str] = set()

    def solve(self, env, task_index: int, max_num_steps: int = 30):
        from tau_bench.types import SolveResult, RESPOND_ACTION_NAME
        from tau_bench.agents.tool_calling_agent import message_to_action

        self.controller.reset()
        self._discovered_tools = set()

        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        info = env_reset.info.model_dump()
        reward = 0.0

        messages = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        total_cost = 0.0
        total_input = 0
        total_output = 0
        step_logs = []

        # Start with conservative profile (first step needs full context)
        current_profile = PROFILES[ProfileName.PRESERVE]

        for step in range(max_num_steps):
            send_messages = self._apply_compression(messages, step, current_profile)
            tools = self._select_tools(current_profile)

            model, litellm_kwargs = self._build_call(send_messages, tools)
            res = litellm.completion(**litellm_kwargs)

            usage = res.usage
            step_input = (usage.prompt_tokens or 0) if usage else 0
            step_output = (usage.completion_tokens or 0) if usage else 0
            step_cost = res._hidden_params.get("response_cost", 0) or 0
            if not step_cost:
                model_name = self.config.get_model(ModelTier.STRONG)
                rate_in = INPUT_COST_PER_MTOK.get(model_name, 0)
                rate_out = OUTPUT_COST_PER_MTOK.get(model_name, 0)
                step_cost = (step_input / 1_000_000 * rate_in) + (step_output / 1_000_000 * rate_out)

            total_input += step_input
            total_output += step_output
            total_cost += step_cost

            # Estimate uncertainty and select next profile
            uncertainty = self.estimator.estimate(res, messages)
            confidence = uncertainty.confidence if uncertainty.num_tokens_used > 0 else 0.5
            next_profile = self.controller.select_profile(confidence)

            next_message = res.choices[0].message.model_dump()
            if next_message.get("content") is None:
                next_message["content"] = ""

            action = message_to_action(next_message)
            tool_name = action.name if action.name != RESPOND_ACTION_NAME else "__respond__"
            tool_args = ""
            tc_list = next_message.get("tool_calls") or []
            if tc_list:
                args_str = tc_list[0].get("function", {}).get("arguments", "")
                tool_args = args_str[:200] if len(args_str) > 200 else args_str

            step_logs.append(AdaptiveStepLog(
                task_id=task_index,
                step=step,
                tool_called=tool_name,
                confidence=confidence,
                profile_selected=current_profile.name.value,
                mask_window=current_profile.mask_window,
                pruning_used=current_profile.tool_pruning,
                progressive_used=current_profile.progressive,
                input_tokens=step_input,
                output_tokens=step_output,
                cost=step_cost,
            ))

            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Update controller with behavioral signals
            self.controller.update_step(tool_name, tool_args)

            if action.name != RESPOND_ACTION_NAME:
                self._discovered_tools.add(action.name)
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
                messages.extend([
                    next_message,
                    {"role": "user", "content": env_response.observation or ""},
                ])

            current_profile = next_profile

            if env_response.done:
                break

        return AdaptiveTaskResult(
            task_id=task_index,
            reward=reward,
            total_steps=len(step_logs),
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            step_logs=step_logs,
            controller_stats=self.controller.get_stats(),
        )

    def _apply_compression(
        self, messages: list[dict], step: int, profile: CompressionProfile
    ) -> list[dict]:
        """Apply masking based on the current profile's mask_window."""
        if step == 0 or profile.mask_window >= 999:
            return list(messages)

        return _mask_old_tool_outputs(messages, profile.mask_window)

    def _select_tools(self, profile: CompressionProfile) -> list[dict]:
        """Select tool definitions based on current profile."""
        if profile.progressive:
            tools = []
            for name, full_def in self._full_tools.items():
                if name in self._discovered_tools:
                    if profile.tool_pruning:
                        tools.append(self._pruned_tools[name])
                    else:
                        tools.append(full_def)
                else:
                    tools.append(self._routing_tools[name])
            return tools

        if profile.tool_pruning:
            return list(self._pruned_tools.values())

        return self.tools_info

    def _build_call(self, messages: list[dict], tools: list[dict]) -> tuple[str, dict]:
        """Build litellm completion kwargs."""
        model_name = self.config.get_model(ModelTier.STRONG)

        if self.config.provider == ProviderType.DEEPSEEK:
            litellm_model = f"deepseek/{model_name}"
        elif self.config.provider == ProviderType.DEEPINFRA:
            litellm_model = f"deepinfra/{model_name}"
        elif self.config.provider == ProviderType.GROQ:
            litellm_model = f"groq/{model_name}"
        elif self.config.provider == ProviderType.OPENROUTER:
            litellm_model = f"openrouter/{model_name}"
        else:
            litellm_model = model_name

        kwargs = dict(
            messages=messages,
            model=litellm_model,
            tools=tools,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=5,
        )

        if self.config.provider == ProviderType.DEEPSEEK:
            kwargs["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
            kwargs["api_base"] = "https://api.deepseek.com"
            kwargs["allowed_openai_params"] = ["logprobs", "top_logprobs"]
        elif self.config.provider == ProviderType.DEEPINFRA:
            kwargs["api_key"] = os.environ.get("DeepInfra_API_KEY", "")
            kwargs["api_base"] = "https://api.deepinfra.com/v1/openai"
            kwargs["allowed_openai_params"] = ["logprobs", "top_logprobs"]
        elif self.config.provider == ProviderType.GROQ:
            kwargs["api_key"] = os.environ.get("GROQ_API_KEY", "")
            kwargs["api_base"] = "https://api.groq.com/openai/v1"
        elif self.config.provider == ProviderType.OPENROUTER:
            kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
            kwargs["api_base"] = "https://openrouter.ai/api/v1"

        return litellm_model, kwargs


# --- Tool definition helpers (same as tau_runner.py) ---

def _prune_tool_def(tool: dict) -> dict:
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


def _make_routing_def(tool: dict) -> dict:
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


def _mask_old_tool_outputs(messages: list[dict], window_k: int) -> list[dict]:
    """Replace old tool outputs with compact summaries, preserving still-referenced ones.

    Two-pass approach:
    1. Recency: keep the last window_k tool outputs (always visible).
    2. Staleness: among older outputs, keep any whose key identifiers
       still appear in recent assistant/user messages.
    """
    import re

    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= window_k:
        return list(messages)

    recency_keep = set(tool_indices[-window_k:])
    candidates_to_mask = [i for i in tool_indices if i not in recency_keep]

    recent_text = _collect_recent_text(messages, lookback=6)

    indices_to_mask = set()
    for i in candidates_to_mask:
        content = messages[i].get("content", "")
        if _is_still_referenced(content, recent_text):
            continue
        indices_to_mask.add(i)

    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_mask:
            tool_name = msg.get("name", "unknown")
            original = msg.get("content", "")
            summary = _extract_key_info(tool_name, original)
            result.append({**msg, "content": summary})
        else:
            result.append(msg)
    return result


def _collect_recent_text(messages: list[dict], lookback: int) -> str:
    """Gather text from the last `lookback` assistant/user messages."""
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
    """Check if key identifiers from a tool output appear in recent messages."""
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

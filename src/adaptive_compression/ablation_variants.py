"""Ablation variant agents and controllers.

Creates modified versions of the adaptive agent for ablation studies.
Does NOT modify any existing classes — creates new subclasses that
override specific behaviors.
"""
from __future__ import annotations

import copy
import re

from adaptive_compression.adaptive_agent import (
    AdaptiveTauAgent,
    AdaptiveStepLog,
    AdaptiveTaskResult,
    _extract_key_info,
)
from adaptive_compression.controller import (
    ThresholdController,
    CompressionProfile,
    ProfileName,
    PROFILES,
)
from adaptive_compression.uncertainty import LogprobEstimator


# ---------------------------------------------------------------------------
# Variant 1: Adaptive WITHOUT staleness (recency-only masking)
# ---------------------------------------------------------------------------

def _mask_recency_only(messages: list[dict], window_k: int) -> list[dict]:
    """Original recency-only masking without staleness check."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= window_k:
        return list(messages)

    indices_to_mask = set(tool_indices[:-window_k])
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


class AdaptiveNoStalenessAgent(AdaptiveTauAgent):
    """Full adaptive controller but recency-only masking (no staleness check)."""

    def _apply_compression(self, messages, step, profile):
        if step == 0 or profile.mask_window >= 999:
            return list(messages)
        return _mask_recency_only(messages, profile.mask_window)


# ---------------------------------------------------------------------------
# Variant 2: Staleness-aware masking WITHOUT confidence (fixed profile)
# ---------------------------------------------------------------------------

class FixedProfileController(ThresholdController):
    """Always selects a fixed compression profile — no confidence signal."""

    def __init__(self, fixed_profile: ProfileName = ProfileName.LIGHT, **kwargs):
        super().__init__(**kwargs)
        self.fixed_profile = fixed_profile

    def select_profile(self, confidence: float) -> CompressionProfile:
        self.state.confidence_history.append(confidence)
        self.state.step_count += 1
        if self.state.step_count <= 2:
            self.state.profile_history.append(ProfileName.PRESERVE)
            return PROFILES[ProfileName.PRESERVE]
        self.state.profile_history.append(self.fixed_profile)
        return PROFILES[self.fixed_profile]


class StalenessOnlyAgent(AdaptiveTauAgent):
    """Staleness-aware masking with a fixed compression profile (no confidence)."""

    def __init__(self, config, tools_info, wiki, temperature=0.0,
                 fixed_profile=ProfileName.LIGHT):
        controller = FixedProfileController(fixed_profile=fixed_profile)
        super().__init__(
            config=config,
            tools_info=tools_info,
            wiki=wiki,
            temperature=temperature,
            controller=controller,
        )


# ---------------------------------------------------------------------------
# Variant 3: Adaptive WITHOUT behavioral penalties
# ---------------------------------------------------------------------------

class NoBehavioralController(ThresholdController):
    """Threshold controller with all behavioral penalties disabled."""

    def __init__(self, **kwargs):
        super().__init__(
            stall_penalty=0.0,
            respond_loop_penalty=0.0,
            declining_trend_penalty=0.0,
            **kwargs,
        )


class AdaptiveNoBehavioralAgent(AdaptiveTauAgent):
    """Full adaptive with staleness masking but no behavioral penalties."""

    def __init__(self, config, tools_info, wiki, temperature=0.0):
        controller = NoBehavioralController()
        super().__init__(
            config=config,
            tools_info=tools_info,
            wiki=wiki,
            temperature=temperature,
            controller=controller,
        )


# ---------------------------------------------------------------------------
# Variant 4: Adaptive WITHOUT progressive disclosure
# ---------------------------------------------------------------------------

class NoProgressiveController(ThresholdController):
    """Threshold controller that strips progressive from all profiles."""

    def select_profile(self, confidence: float) -> CompressionProfile:
        profile = super().select_profile(confidence)
        return CompressionProfile(
            name=profile.name,
            mask_window=profile.mask_window,
            tool_pruning=profile.tool_pruning,
            progressive=False,
        )


class AdaptiveNoProgressiveAgent(AdaptiveTauAgent):
    """Full adaptive with staleness masking but progressive disclosure disabled."""

    def __init__(self, config, tools_info, wiki, temperature=0.0):
        controller = NoProgressiveController()
        super().__init__(
            config=config,
            tools_info=tools_info,
            wiki=wiki,
            temperature=temperature,
            controller=controller,
        )


# ---------------------------------------------------------------------------
# Variant 5: Trace-context — strip assistant verbosity, keep all tool outputs
# ---------------------------------------------------------------------------
#
# Hypothesis: the agent's own reasoning prose (the `content` of past assistant
# messages) has near-zero value on replay once the step is taken. The thing
# that prevents loops / repeated work is knowing *what was done*, not *why it
# was justified*. So we replace each older assistant message's verbose content
# with a compact structured action line parsed from its tool_calls, while
# keeping the tool_calls skeleton (required for tool_call_id linkage) and
# every tool output verbatim.

def _summarize_assistant_action(msg: dict) -> str:
    """Build a terse action line from an assistant message's tool_calls.

    Example: "[called get_order(order_id=#W123, item_id=456)]".
    For a respond turn (no tool_calls), preserve the actual response text the
    user saw — the agent's own prior turns are part of an ongoing dialogue and
    dropping them breaks multi-turn continuity. Truncate long responses.
    """
    import json

    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        # Respond turn: keep what was actually said (truncated), not a marker.
        said = (msg.get("content") or "").strip()
        if not said:
            return "[responded to user]"
        if len(said) > 200:
            said = said[:200] + "..."
        return f"[responded: {said}]"

    parts = []
    for call in tool_calls:
        fn = call.get("function", {}) or {}
        name = fn.get("name", "unknown")
        raw_args = fn.get("arguments", "") or ""
        arg_str = ""
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            if isinstance(parsed, dict):
                arg_str = ", ".join(f"{k}={v}" for k, v in parsed.items())
            else:
                # Non-dict JSON args (array/primitive): keep them, don't drop.
                arg_str = json.dumps(parsed)
        except (ValueError, TypeError):
            arg_str = raw_args[:120]
        if len(arg_str) > 160:
            arg_str = arg_str[:160] + "..."
        parts.append(f"{name}({arg_str})")
    return f"[called {'; '.join(parts)}]"


def _strip_assistant_verbosity(messages: list[dict], keep_last_k: int = 1) -> list[dict]:
    """Replace verbose content of older assistant messages with action lines.

    The most recent `keep_last_k` assistant messages keep their full content
    (the model may still be mid-reasoning on the current thread); all older
    assistant messages have their prose `content` replaced by a structured
    action line. tool_calls are preserved untouched so tool_call_id linkage
    stays valid. Tool outputs are never modified here.
    """
    assistant_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "assistant"
    ]
    if len(assistant_indices) <= keep_last_k:
        return list(messages)

    indices_to_strip = set(assistant_indices[:-keep_last_k]) if keep_last_k > 0 \
        else set(assistant_indices)

    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_strip:
            result.append({**msg, "content": _summarize_assistant_action(msg)})
        else:
            result.append(msg)
    return result


class TraceContextAgent(AdaptiveTauAgent):
    """Strip assistant reasoning verbosity, keep all tool outputs verbatim.

    Tests the hypothesis that assistant prose is free savings: the structured
    trace of *what was done* (action lines + full tool outputs) is enough
    context, and the model's own past chain-of-thought can be discarded.
    Uses the full adaptive controller but disables tool-output masking so the
    effect of verbosity stripping is isolated.
    """

    def __init__(self, config, tools_info, wiki, temperature=0.0, keep_last_k=1):
        super().__init__(
            config=config,
            tools_info=tools_info,
            wiki=wiki,
            temperature=temperature,
        )
        self.keep_last_k = keep_last_k

    def _apply_compression(self, messages, step, profile):
        # Isolate the variable under test: never mask tool outputs, only strip
        # assistant verbosity. The controller still drives tool selection
        # (progressive/pruning) via _select_tools.
        if step == 0:
            return list(messages)
        return _strip_assistant_verbosity(messages, keep_last_k=self.keep_last_k)


# ---------------------------------------------------------------------------
# Variant 6: Trace-compact — trace stripping STACKED with masking + progressive
# ---------------------------------------------------------------------------
#
# The trace-context variant isolated verbosity stripping and left tool outputs
# and tool definitions fully intact, so its absolute token count stayed high
# (fixed per-step overhead is dominated by tool defs + accumulating outputs,
# not assistant prose). This variant stacks all three safe techniques to get
# the absolute count down while keeping the trace's "all outputs preserved"
# guarantee replaced by staleness-aware masking (still-referenced outputs are
# kept; only truly stale ones are summarized):
#   - trace:        old assistant prose -> structured action lines
#   - masking:      staleness-aware tool-output masking (fixed small window)
#   - progressive:  routing-only tool defs until a tool is first used
#
# It uses a FIXED compression profile (not the confidence controller) so the
# contribution of stacking is attributable without confounding by adaptivity.

class FixedCompactController(ThresholdController):
    """Always returns a fixed mask+progressive profile (no confidence signal).

    Mirrors FixedProfileController but defaults to a profile that also enables
    progressive disclosure, so the base agent's _select_tools serves routing
    defs and _apply_compression masks stale outputs.
    """

    def __init__(self, fixed_profile: ProfileName = ProfileName.LIGHT_PROG, **kwargs):
        super().__init__(**kwargs)
        self.fixed_profile = fixed_profile

    def select_profile(self, confidence: float) -> CompressionProfile:
        self.state.confidence_history.append(confidence)
        self.state.step_count += 1
        # First 2 steps preserve full context to understand the task.
        if self.state.step_count <= 2:
            self.state.profile_history.append(ProfileName.PRESERVE)
            return PROFILES[ProfileName.PRESERVE]
        self.state.profile_history.append(self.fixed_profile)
        return PROFILES[self.fixed_profile]


class TraceCompactAgent(AdaptiveTauAgent):
    """Trace verbosity stripping stacked with staleness masking + progressive.

    Layers the trace action-line rewrite on top of the base agent's existing
    staleness-aware tool masking and progressive tool-definition disclosure,
    driven by a fixed compression profile. This is the "max safe compression"
    candidate: it cuts the three big token layers (assistant prose, stale
    outputs, upfront tool defs) without dropping still-referenced data.
    """

    def __init__(self, config, tools_info, wiki, temperature=0.0,
                 keep_last_k=1, fixed_profile=ProfileName.MODERATE):
        controller = FixedCompactController(fixed_profile=fixed_profile)
        super().__init__(
            config=config,
            tools_info=tools_info,
            wiki=wiki,
            temperature=temperature,
            controller=controller,
        )
        self.keep_last_k = keep_last_k

    def _apply_compression(self, messages, step, profile):
        # First run the base staleness-aware tool masking for this profile,
        # then strip assistant verbosity on top. Order matters: masking reads
        # assistant/user text to decide staleness, so do it before we rewrite
        # the assistant content into terse action lines.
        masked = super()._apply_compression(messages, step, profile)
        if step == 0:
            return masked
        return _strip_assistant_verbosity(masked, keep_last_k=self.keep_last_k)


# ---------------------------------------------------------------------------
# Registry: map ablation names to agent factory functions
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "ablation-no-staleness": {
        "description": "Adaptive controller, recency-only masking (no staleness)",
        "agent_class": AdaptiveNoStalenessAgent,
        "agent_kwargs": {},
    },
    "ablation-staleness-only": {
        "description": "Staleness-aware masking with fixed light profile (no confidence)",
        "agent_class": StalenessOnlyAgent,
        "agent_kwargs": {"fixed_profile": ProfileName.LIGHT},
    },
    "ablation-no-behavioral": {
        "description": "Full adaptive, staleness masking, no behavioral penalties",
        "agent_class": AdaptiveNoBehavioralAgent,
        "agent_kwargs": {},
    },
    "ablation-no-progressive": {
        "description": "Full adaptive, staleness masking, progressive disclosure disabled",
        "agent_class": AdaptiveNoProgressiveAgent,
        "agent_kwargs": {},
    },
    "ablation-trace-context": {
        "description": "Strip assistant verbosity to action lines, keep all tool outputs verbatim",
        "agent_class": TraceContextAgent,
        "agent_kwargs": {"keep_last_k": 1},
    },
    "ablation-trace-compact": {
        "description": "Trace stripping stacked with staleness masking + progressive disclosure",
        "agent_class": TraceCompactAgent,
        "agent_kwargs": {"keep_last_k": 1, "fixed_profile": ProfileName.MODERATE},
    },
}

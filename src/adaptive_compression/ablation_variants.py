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
}

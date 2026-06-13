"""Adaptive compression controller.

Maps per-step uncertainty scores to compression profiles. Uses logprob
confidence as the primary signal, with behavioral signals as adjustments.

Profiles use all three compression techniques:
  - Observation masking (variable window)
  - Progressive disclosure
  - Tool pruning

The key design principle: behavioral signals ADJUST the confidence score
rather than hard-overriding the profile selection. This keeps the controller
responsive without being too conservative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProfileName(str, Enum):
    MAX_COMPRESS = "max_compress"
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    LIGHT_PROG = "light_prog"
    LIGHT = "light"
    PRESERVE = "preserve"


@dataclass
class CompressionProfile:
    name: ProfileName
    mask_window: int
    tool_pruning: bool
    progressive: bool

    def label(self) -> str:
        parts = []
        if self.tool_pruning:
            parts.append("prune")
        if self.progressive:
            parts.append("progressive")
        parts.append(f"mask(k={self.mask_window})")
        return f"adaptive:{self.name.value}[{'+'.join(parts)}]"


PROFILES = {
    ProfileName.MAX_COMPRESS: CompressionProfile(
        name=ProfileName.MAX_COMPRESS,
        mask_window=1,
        tool_pruning=True,
        progressive=True,
    ),
    ProfileName.AGGRESSIVE: CompressionProfile(
        name=ProfileName.AGGRESSIVE,
        mask_window=1,
        tool_pruning=False,
        progressive=True,
    ),
    ProfileName.MODERATE: CompressionProfile(
        name=ProfileName.MODERATE,
        mask_window=3,
        tool_pruning=False,
        progressive=True,
    ),
    ProfileName.LIGHT_PROG: CompressionProfile(
        name=ProfileName.LIGHT_PROG,
        mask_window=5,
        tool_pruning=False,
        progressive=True,
    ),
    ProfileName.LIGHT: CompressionProfile(
        name=ProfileName.LIGHT,
        mask_window=5,
        tool_pruning=False,
        progressive=False,
    ),
    ProfileName.PRESERVE: CompressionProfile(
        name=ProfileName.PRESERVE,
        mask_window=999,
        tool_pruning=False,
        progressive=False,
    ),
}


@dataclass
class ControllerState:
    """Tracks controller decisions across a trajectory."""
    confidence_history: list[float] = field(default_factory=list)
    profile_history: list[ProfileName] = field(default_factory=list)
    tool_history: list[str] = field(default_factory=list)
    args_history: list[str] = field(default_factory=list)
    step_count: int = 0


class ThresholdController:
    """Threshold-based adaptive controller.

    Primary signal: logprob confidence [0, 1]
    Behavioral adjustments: nudge confidence down when problems detected.
    """

    def __init__(
        self,
        high_threshold: float = 0.90,
        mid_high_threshold: float = 0.85,
        mid_threshold: float = 0.80,
        low_threshold: float = 0.65,
        trend_window: int = 3,
        max_steps: int = 30,
        stall_penalty: float = 0.15,
        respond_loop_penalty: float = 0.10,
        declining_trend_penalty: float = 0.08,
    ):
        self.high_threshold = high_threshold
        self.mid_high_threshold = mid_high_threshold
        self.mid_threshold = mid_threshold
        self.low_threshold = low_threshold
        self.trend_window = trend_window
        self.max_steps = max_steps
        self.stall_penalty = stall_penalty
        self.respond_loop_penalty = respond_loop_penalty
        self.declining_trend_penalty = declining_trend_penalty
        self.state = ControllerState()

    def reset(self):
        self.state = ControllerState()

    def update_step(self, tool_name: str, tool_args: str):
        """Call after each step to update behavioral state."""
        self.state.tool_history.append(tool_name)
        self.state.args_history.append(tool_args)

    def select_profile(self, confidence: float) -> CompressionProfile:
        """Select compression profile for the next step."""
        self.state.confidence_history.append(confidence)
        self.state.step_count += 1

        # First 2 steps: always preserve (need to understand the task)
        if self.state.step_count <= 2:
            self.state.profile_history.append(ProfileName.PRESERVE)
            return PROFILES[ProfileName.PRESERVE]

        # Apply behavioral adjustments to confidence
        adjusted = self._apply_adjustments(confidence)

        # Map adjusted confidence to profile
        position = self.state.step_count / self.max_steps
        profile_name = self._select(adjusted, position)
        self.state.profile_history.append(profile_name)

        return PROFILES[profile_name]

    def _apply_adjustments(self, confidence: float) -> float:
        """Nudge confidence based on behavioral signals."""
        adjusted = confidence

        # Stall: same tool+args back-to-back
        if self._detect_stall():
            adjusted -= self.stall_penalty

        # Respond loop: too many __respond__ recently
        if self._respond_ratio() > 0.6:
            adjusted -= self.respond_loop_penalty

        # Declining trend
        trend = self._compute_trend()
        if trend < -0.05:
            adjusted -= self.declining_trend_penalty

        return max(0.0, min(1.0, adjusted))

    def _select(self, confidence: float, position: float) -> ProfileName:
        # Very high + late trajectory: max compression (all techniques)
        if confidence > self.high_threshold and position > 0.4:
            return ProfileName.MAX_COMPRESS

        # Very high + past early phase: aggressive (mask=1 + progressive)
        if confidence > self.high_threshold and position > 0.2:
            return ProfileName.AGGRESSIVE

        # High: moderate (mask=3 + progressive)
        if confidence > self.mid_high_threshold:
            return ProfileName.MODERATE

        # Medium-high: light with progressive
        if confidence > self.mid_threshold:
            return ProfileName.LIGHT_PROG

        # Medium: light (mask=5 only)
        if confidence > self.low_threshold:
            return ProfileName.LIGHT

        # Low: preserve
        return ProfileName.PRESERVE

    def _detect_stall(self) -> bool:
        """Same tool+args back-to-back, or same tool 3x in a row."""
        if len(self.state.tool_history) < 2:
            return False
        last = self.state.tool_history[-1]
        if last == "__respond__":
            return False
        if (last == self.state.tool_history[-2]
                and self.state.args_history[-1] == self.state.args_history[-2]):
            return True
        if len(self.state.tool_history) >= 3:
            recent = self.state.tool_history[-3:]
            if all(t == recent[0] for t in recent) and recent[0] != "__respond__":
                return True
        return False

    def _respond_ratio(self) -> float:
        """Fraction of __respond__ in last 6 steps."""
        if len(self.state.tool_history) < 3:
            return 0.0
        window = self.state.tool_history[-6:]
        return sum(1 for t in window if t == "__respond__") / len(window)

    def _compute_trend(self) -> float:
        """Confidence trend. Negative = declining."""
        history = self.state.confidence_history
        if len(history) < 2:
            return 0.0
        window = history[-self.trend_window:]
        if len(window) < 2:
            return 0.0
        return (window[-1] - window[0]) / len(window)

    def get_stats(self) -> dict:
        """Summary statistics for logging."""
        history = self.state.confidence_history
        profiles = self.state.profile_history
        if not history:
            return {}

        from collections import Counter
        profile_counts = Counter(p.value for p in profiles)

        return {
            "steps": self.state.step_count,
            "mean_confidence": sum(history) / len(history),
            "min_confidence": min(history),
            "max_confidence": max(history),
            "profile_distribution": dict(profile_counts),
            "final_profile": profiles[-1].value if profiles else None,
        }

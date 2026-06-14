"""Budget-aware adaptive compression controller.

Extends ThresholdController with budget awareness: as remaining tokens
shrink, confidence thresholds shift down so compression kicks in earlier.
"""
from __future__ import annotations

from adaptive_compression.controller import (
    ThresholdController,
    CompressionProfile,
    ProfileName,
    PROFILES,
)


class BudgetAwareController(ThresholdController):
    """Threshold controller that adjusts compression based on remaining budget.

    When budget is plentiful (>50% remaining), behaves identically to the
    base controller. As budget depletes, thresholds shift down so the
    controller compresses more aggressively at the same confidence level.
    """

    def __init__(self, token_budget: int, **kwargs):
        super().__init__(**kwargs)
        self.token_budget = token_budget
        self.tokens_spent = 0

    def update_budget(self, step_tokens: int):
        self.tokens_spent += step_tokens

    @property
    def remaining_ratio(self) -> float:
        if self.token_budget <= 0:
            return 1.0
        return max(0.0, 1.0 - self.tokens_spent / self.token_budget)

    @property
    def budget_exhausted(self) -> bool:
        return self.tokens_spent >= self.token_budget

    def select_profile(self, confidence: float) -> CompressionProfile:
        self.state.confidence_history.append(confidence)
        self.state.step_count += 1

        if self.state.step_count <= 2:
            self.state.profile_history.append(ProfileName.PRESERVE)
            return PROFILES[ProfileName.PRESERVE]

        adjusted = self._apply_adjustments(confidence)
        adjusted = self._apply_budget_pressure(adjusted)

        position = self.state.step_count / self.max_steps
        profile_name = self._select(adjusted, position)
        self.state.profile_history.append(profile_name)

        return PROFILES[profile_name]

    def _apply_budget_pressure(self, confidence: float) -> float:
        """Shift confidence up when budget is running low.

        Higher effective confidence → more aggressive compression profile.
        The mapping:
          >50% remaining: no adjustment
          20-50% remaining: +0.05 (nudge toward more compression)
          10-20% remaining: +0.10
          <10% remaining: +0.15
        """
        r = self.remaining_ratio

        if r > 0.5:
            return confidence
        elif r > 0.2:
            return min(1.0, confidence + 0.05)
        elif r > 0.1:
            return min(1.0, confidence + 0.10)
        else:
            return min(1.0, confidence + 0.15)

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["token_budget"] = self.token_budget
        stats["tokens_spent"] = self.tokens_spent
        stats["remaining_ratio"] = self.remaining_ratio
        return stats

"""Behavioral uncertainty signals extracted from trajectory state.

These signals require NO logprobs and work with any model/provider.
They detect failure patterns observable from the agent's actions alone:

1. Repetition score — same tool called consecutively
2. Respond ratio — fraction of recent steps that are __respond__ (no tool use)
3. Step position score — normalized position in trajectory (later = more uncertain)
4. Context growth rate — how fast input tokens are growing
5. Stall detection — agent hasn't made progress (same tools, same args)

All return values in [0, 1] where 1 = fully confident, 0 = very uncertain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BehavioralSignals:
    """Per-step behavioral features computed from trajectory history."""
    repetition_score: float  # 1.0 = no repetition, 0.0 = stuck in loop
    respond_ratio: float  # 1.0 = all tool calls, 0.0 = all __respond__
    step_position: float  # 1.0 = early in trajectory, 0.0 = near step limit
    context_growth: float  # 1.0 = normal growth, 0.0 = exploding context
    stall_score: float  # 1.0 = making progress, 0.0 = repeating exact actions
    composite: float  # weighted combination of all signals


@dataclass
class TrajectoryState:
    """Accumulates trajectory history for behavioral signal extraction."""
    tool_history: list[str] = field(default_factory=list)
    args_history: list[str] = field(default_factory=list)
    token_history: list[int] = field(default_factory=list)
    max_steps: int = 30
    context_window: int = 131072

    def update(self, tool_name: str, tool_args: str, input_tokens: int):
        self.tool_history.append(tool_name)
        self.args_history.append(tool_args)
        self.token_history.append(input_tokens)

    def compute_signals(self) -> BehavioralSignals:
        step = len(self.tool_history)
        if step == 0:
            return BehavioralSignals(
                repetition_score=1.0, respond_ratio=1.0,
                step_position=1.0, context_growth=1.0,
                stall_score=1.0, composite=1.0,
            )

        rep = self._repetition_score()
        resp = self._respond_ratio()
        pos = self._step_position()
        ctx = self._context_growth()
        stall = self._stall_score()

        composite = (
            0.30 * rep
            + 0.25 * resp
            + 0.15 * pos
            + 0.10 * ctx
            + 0.20 * stall
        )

        return BehavioralSignals(
            repetition_score=rep,
            respond_ratio=resp,
            step_position=pos,
            context_growth=ctx,
            stall_score=stall,
            composite=composite,
        )

    def _repetition_score(self) -> float:
        """1.0 = no consecutive repetitions, decays toward 0 with more repeats."""
        if len(self.tool_history) < 2:
            return 1.0

        window = self.tool_history[-5:]
        consecutive = 0
        for i in range(len(window) - 1, 0, -1):
            if window[i] == window[i - 1] and window[i] != "__respond__":
                consecutive += 1
            else:
                break

        if consecutive == 0:
            return 1.0
        return max(0.0, 1.0 - consecutive * 0.3)

    def _respond_ratio(self) -> float:
        """1.0 = all tool calls, 0.0 = all __respond__ in recent window."""
        window = self.tool_history[-6:]
        if not window:
            return 1.0
        responds = sum(1 for t in window if t == "__respond__")
        ratio = responds / len(window)
        return max(0.0, 1.0 - ratio)

    def _step_position(self) -> float:
        """1.0 at step 0, decays toward 0 at max_steps."""
        step = len(self.tool_history)
        return max(0.0, 1.0 - (step / self.max_steps))

    def _context_growth(self) -> float:
        """1.0 = context mostly empty, decays toward 0 as context fills up.

        Measures what fraction of the context window is used. An agent
        consuming 80%+ of its context is running out of room and likely
        struggling with a complex task.
        """
        if not self.token_history:
            return 1.0

        current_tokens = self.token_history[-1]
        utilization = current_tokens / max(self.context_window, 1)
        return max(0.0, 1.0 - utilization)

    def _stall_score(self) -> float:
        """1.0 = all unique recent actions, 0.0 = repeating exact (tool, args) pairs."""
        if len(self.tool_history) < 3:
            return 1.0

        window = min(6, len(self.tool_history))
        recent_actions = [
            (self.tool_history[-i], self.args_history[-i])
            for i in range(1, window + 1)
        ]

        unique = len(set(recent_actions))
        total = len(recent_actions)
        return unique / total

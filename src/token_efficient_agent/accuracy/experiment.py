"""Experiment configurations for BFCL and Tau-bench benchmarks.

BFCL (single-turn): Only prune, progressive, route matter.
  Full 2^3 factorial = 8 configs.

Tau-bench (multi-turn): All 5 factors matter.
  Resolution IV 2^(5-1) fractional factorial + 4 supplements = 20 configs.
  Generator: E = ABCD (two-factor interactions aliased with three-factor only).
"""
from __future__ import annotations

from token_efficient_agent.config import OptimizationFlags

TEMPERATURE = 0.3
NUM_RUNS = 3
BFCL_SUBSAMPLE_SIZE = 100
STRATIFIED_SEED = 42


def _f(cache=False, prune=False, progressive=False, mask=False, route=False) -> OptimizationFlags:
    return OptimizationFlags(
        prompt_caching=cache,
        tool_pruning=prune,
        progressive_disclosure=progressive,
        observation_masking=mask,
        model_routing=route,
    )


# --- BFCL configs: 3 factors (prune × progressive × route), full 2^3 = 8 ---
BFCL_CONFIGS: list[OptimizationFlags] = [
    _f(),                                              # baseline
    _f(prune=True),                                    # prune
    _f(progressive=True),                              # progressive
    _f(route=True),                                    # route
    _f(prune=True, progressive=True),                  # prune+progressive
    _f(prune=True, route=True),                        # prune+route
    _f(progressive=True, route=True),                  # progressive+route
    _f(prune=True, progressive=True, route=True),      # all three
]

BFCL_FACTOR_NAMES = ["prune", "progressive", "route"]

BFCL_FACTOR_MATRIX = [
    [0, 0, 0],  # baseline
    [1, 0, 0],  # prune
    [0, 1, 0],  # progressive
    [0, 0, 1],  # route
    [1, 1, 0],  # prune+progressive
    [1, 0, 1],  # prune+route
    [0, 1, 1],  # progressive+route
    [1, 1, 1],  # all three
]


# --- Tau-bench configs: 4 factors (prune, progressive, mask, route), full 2^4 = 16 ---
# Caching removed: OpenRouter auto-caches at provider level, our cache flag is a no-op.
TAU_CONFIGS: list[OptimizationFlags] = [
    _f(),                                                          # 1: baseline
    _f(prune=True),                                                # 2: prune
    _f(progressive=True),                                          # 3: progressive
    _f(mask=True),                                                 # 4: mask
    _f(route=True),                                                # 5: route
    _f(prune=True, progressive=True),                              # 6: prune+progressive
    _f(prune=True, mask=True),                                     # 7: prune+mask
    _f(prune=True, route=True),                                    # 8: prune+route
    _f(progressive=True, mask=True),                               # 9: progressive+mask
    _f(progressive=True, route=True),                              # 10: progressive+route
    _f(mask=True, route=True),                                     # 11: mask+route
    _f(prune=True, progressive=True, mask=True),                   # 12: prune+progressive+mask
    _f(prune=True, progressive=True, route=True),                  # 13: prune+progressive+route
    _f(prune=True, mask=True, route=True),                         # 14: prune+mask+route
    _f(progressive=True, mask=True, route=True),                   # 15: progressive+mask+route
    _f(prune=True, progressive=True, mask=True, route=True),       # 16: all four
]

TAU_FACTOR_NAMES = ["prune", "progressive", "mask", "route"]

TAU_FACTOR_MATRIX = [
    [0, 0, 0, 0],  # 1: baseline
    [1, 0, 0, 0],  # 2: prune
    [0, 1, 0, 0],  # 3: progressive
    [0, 0, 1, 0],  # 4: mask
    [0, 0, 0, 1],  # 5: route
    [1, 1, 0, 0],  # 6: prune+progressive
    [1, 0, 1, 0],  # 7: prune+mask
    [1, 0, 0, 1],  # 8: prune+route
    [0, 1, 1, 0],  # 9: progressive+mask
    [0, 1, 0, 1],  # 10: progressive+route
    [0, 0, 1, 1],  # 11: mask+route
    [1, 1, 1, 0],  # 12: prune+progressive+mask
    [1, 1, 0, 1],  # 13: prune+progressive+route
    [1, 0, 1, 1],  # 14: prune+mask+route
    [0, 1, 1, 1],  # 15: progressive+mask+route
    [1, 1, 1, 1],  # 16: all four
]

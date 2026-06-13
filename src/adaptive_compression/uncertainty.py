"""Uncertainty estimation at the agent-step level.

Strategies:
  A) LogprobEstimator    — Token logprob confidence (entropy or mean logprob).
  B) PromptedEstimator   — Ask a cheap model to rate the agent's confidence.
  C) HybridEstimator     — Use logprobs when available, fall back to prompted.
  D) CompositeEstimator  — Blend logprobs + prompted + behavioral signals.
                           Works with any model/provider. Activated via --behavioral flag.

All return a confidence score in [0, 1] where 1 = fully confident.
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class UncertaintyResult:
    confidence: float  # [0, 1], 1 = fully confident
    method: str  # "logprob" | "prompted" | "hybrid_logprob" | "hybrid_prompted"
    raw_entropy: float | None = None  # unnormalized entropy (logprob only)
    raw_score: int | None = None  # 0-100 integer (prompted only)
    num_tokens_used: int = 0  # how many tokens contributed to the estimate


class UncertaintyEstimator(ABC):
    @abstractmethod
    def estimate(self, response: dict, messages: list[dict]) -> UncertaintyResult:
        """Estimate confidence for the agent's response at this step."""
        ...


class LogprobEstimator(UncertaintyEstimator):
    """Compute confidence from the first N output tokens' log probabilities.

    Two modes depending on what the provider returns:

    1. If top_logprobs are available (e.g. Groq): compute Shannon entropy over
       the distribution. Lower entropy → higher confidence.
    2. If only the chosen token's logprob is available (e.g. DeepInfra): use
       mean logprob. More negative → less confident.

    Both are normalized to [0, 1] where 1 = fully confident.
    """

    def __init__(self, num_tokens: int = 0):
        # 0 = use all available tokens (recommended for thinking models)
        self.num_tokens = num_tokens

    def estimate(self, response: dict, messages: list[dict]) -> UncertaintyResult:
        logprobs_data = _extract_logprobs(response)
        if not logprobs_data:
            return UncertaintyResult(
                confidence=0.5,
                method="logprob",
                raw_entropy=None,
                num_tokens_used=0,
            )

        tokens_to_use = logprobs_data
        if self.num_tokens > 0:
            tokens_to_use = logprobs_data[: self.num_tokens]

        # Check if we have top_logprobs (entropy mode) or just logprob (mean mode)
        has_top = any(
            token_info.get("top_logprobs") for token_info in tokens_to_use
        )

        if has_top:
            return self._entropy_mode(tokens_to_use)
        else:
            return self._mean_logprob_mode(tokens_to_use)

    def _entropy_mode(self, tokens: list[dict]) -> UncertaintyResult:
        """Shannon entropy over top-k logprobs distribution."""
        entropies = []
        for token_info in tokens:
            top_logprobs = token_info.get("top_logprobs", [])
            if not top_logprobs:
                continue
            entropy = _shannon_entropy(top_logprobs)
            entropies.append(entropy)

        if not entropies:
            return UncertaintyResult(
                confidence=0.5, method="logprob_entropy",
                raw_entropy=None, num_tokens_used=0,
            )

        avg_entropy = sum(entropies) / len(entropies)
        k = len(tokens[0].get("top_logprobs", []))
        max_entropy = math.log(k) if k > 1 else 1.0
        normalized = avg_entropy / max_entropy
        confidence = max(0.0, min(1.0, 1.0 - normalized))

        return UncertaintyResult(
            confidence=confidence,
            method="logprob_entropy",
            raw_entropy=avg_entropy,
            num_tokens_used=len(entropies),
        )

    def _mean_logprob_mode(self, tokens: list[dict]) -> UncertaintyResult:
        """Confidence from chosen tokens' logprobs. More negative = less confident.

        Uses two signals:
        1. worst_k: mean of the K most negative logprobs (decision points where
           the model hesitated — these carry the uncertainty signal)
        2. mean_all: mean of all token logprobs (overall fluency)

        The final score blends both, weighted toward worst_k since that's
        where uncertainty concentrates in thinking models.
        """
        logprobs = [t.get("logprob", 0.0) for t in tokens if t.get("logprob") is not None]
        if not logprobs:
            return UncertaintyResult(
                confidence=0.5, method="logprob_mean",
                raw_entropy=None, num_tokens_used=0,
            )

        mean_all = sum(logprobs) / len(logprobs)

        # Worst-K: the most uncertain tokens (most negative logprobs)
        k = min(5, len(logprobs))
        sorted_lps = sorted(logprobs)
        worst_k = sum(sorted_lps[:k]) / k

        # Blend: 70% worst-K (captures decision uncertainty), 30% mean (overall)
        blended = 0.7 * worst_k + 0.3 * mean_all
        confidence = max(0.0, min(1.0, math.exp(blended)))

        return UncertaintyResult(
            confidence=confidence,
            method="logprob_mean",
            raw_entropy=blended,
            num_tokens_used=len(logprobs),
        )


class PromptedEstimator(UncertaintyEstimator):
    """Ask a cheap model to rate the agent's confidence in its last action.

    Uses a separate, minimal LLM call (~50 output tokens) to the weakest
    available model. Returns the self-assessed score normalized to [0, 1].
    """

    def __init__(
        self,
        model: str = "openrouter/google/gemma-4-26b-a4b-it",
        api_base: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.api_base = api_base

    def estimate(self, response: dict, messages: list[dict]) -> UncertaintyResult:
        agent_action = _summarize_action(response)
        recent_context = _summarize_recent(messages, n=4)

        prompt = (
            "You are a strict evaluator scoring an AI assistant's tool call. "
            "Be critical — most calls should score 40-80, not 90-100.\n\n"
            f"User request:\n{recent_context}\n\n"
            f"Assistant's action:\n{agent_action}\n\n"
            "Score 0-100 based on these criteria:\n"
            "- 90-100: Trivial/obvious correct call (e.g., get_weather('Paris') for 'weather in Paris')\n"
            "- 70-89: Likely correct but requires some judgment (right tool, reasonable args)\n"
            "- 40-69: Uncertain — tool seems right but args could be wrong, or multiple tools could work\n"
            "- 20-39: Probably wrong — questionable tool choice or clearly bad arguments\n"
            "- 0-19: Almost certainly wrong — irrelevant tool or nonsensical arguments\n\n"
            "Respond with ONLY a number."
        )

        try:
            from litellm import completion

            res = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                api_base=self.api_base,
            )
            content = res.choices[0].message.content
            # Some "thinking" models return None content (answer consumed by reasoning)
            if content is None:
                reasoning = getattr(res.choices[0].message, "reasoning_content", None)
                if reasoning:
                    content = reasoning
                else:
                    return UncertaintyResult(
                        confidence=0.5,
                        method="prompted",
                        raw_score=None,
                    )
            score = _parse_score(content.strip())
            return UncertaintyResult(
                confidence=score / 100.0,
                method="prompted",
                raw_score=score,
            )
        except Exception:
            return UncertaintyResult(
                confidence=0.5,
                method="prompted",
                raw_score=None,
            )


class HybridEstimator(UncertaintyEstimator):
    """Use logprobs when available, fall back to prompted assessment."""

    def __init__(
        self,
        num_tokens: int = 10,
        prompted_model: str = "openrouter/google/gemma-4-26b-a4b-it",
    ):
        self.logprob_estimator = LogprobEstimator(num_tokens=num_tokens)
        self.prompted_estimator = PromptedEstimator(model=prompted_model)

    def estimate(self, response: dict, messages: list[dict]) -> UncertaintyResult:
        logprobs_data = _extract_logprobs(response)
        if logprobs_data:
            result = self.logprob_estimator.estimate(response, messages)
            result.method = "hybrid_logprob"
            return result

        result = self.prompted_estimator.estimate(response, messages)
        result.method = "hybrid_prompted"
        return result


class CompositeEstimator(UncertaintyEstimator):
    """Blend logprobs + prompted + behavioral signals into a single score.

    Uses whatever signals are available:
    - Logprobs (when provider supports it): strongest signal
    - Behavioral (always available): repetition, respond ratio, stall detection
    - Prompted (always available, costs one cheap API call): semantic assessment

    The behavioral signals require a TrajectoryState object to be maintained
    across steps by the caller.
    """

    def __init__(
        self,
        prompted_model: str = "openrouter/google/gemma-4-26b-a4b-it",
        logprob_weight: float = 0.55,
        behavioral_weight: float = 0.45,
        prompted_weight: float = 0.0,
        enable_prompted: bool = False,
    ):
        from adaptive_compression.behavioral import TrajectoryState
        self.logprob_estimator = LogprobEstimator()
        self.prompted_estimator = PromptedEstimator(model=prompted_model) if enable_prompted else None
        self.trajectory_state = TrajectoryState()
        self.w_logprob = logprob_weight
        self.w_behavioral = behavioral_weight
        self.w_prompted = prompted_weight

    def update_trajectory(self, tool_name: str, tool_args: str, input_tokens: int):
        """Call this after each step to update behavioral state."""
        self.trajectory_state.update(tool_name, tool_args, input_tokens)

    def reset_trajectory(self, context_window: int = 131072):
        """Call at the start of each new task."""
        from adaptive_compression.behavioral import TrajectoryState
        self.trajectory_state = TrajectoryState(context_window=context_window)

    def estimate(self, response: dict, messages: list[dict]) -> UncertaintyResult:
        logprob_result = self.logprob_estimator.estimate(response, messages)
        has_logprobs = logprob_result.num_tokens_used > 0

        prompted_result = None
        has_prompted = False
        if self.prompted_estimator is not None:
            prompted_result = self.prompted_estimator.estimate(response, messages)
            has_prompted = prompted_result.raw_score is not None

        behavioral = self.trajectory_state.compute_signals()

        if has_logprobs and has_prompted:
            confidence = (
                self.w_logprob * logprob_result.confidence
                + self.w_behavioral * behavioral.composite
                + self.w_prompted * prompted_result.confidence
            )
            method = "composite_all"
        elif has_logprobs:
            total = self.w_logprob + self.w_behavioral
            confidence = (
                (self.w_logprob / total) * logprob_result.confidence
                + (self.w_behavioral / total) * behavioral.composite
            )
            method = "composite_logprob+behavioral"
        elif has_prompted:
            total = self.w_prompted + self.w_behavioral
            confidence = (
                (self.w_prompted / total) * prompted_result.confidence
                + (self.w_behavioral / total) * behavioral.composite
            )
            method = "composite_prompted+behavioral"
        else:
            confidence = behavioral.composite
            method = "composite_behavioral_only"

        confidence = max(0.0, min(1.0, confidence))

        return UncertaintyResult(
            confidence=confidence,
            method=method,
            raw_entropy=logprob_result.raw_entropy,
            raw_score=prompted_result.raw_score if prompted_result else None,
            num_tokens_used=logprob_result.num_tokens_used,
        )


# --- Helpers ---


def _extract_logprobs(response) -> list[dict] | None:
    """Extract per-token logprobs from a response.

    Checks both `content` and `reasoning` fields — thinking models (e.g. Qwen3
    on Groq) return logprobs on reasoning tokens even for tool calls, which is
    more useful for uncertainty estimation since it captures the model's
    deliberation about which action to take.

    Handles dict, litellm ModelResponse, Groq response objects, and pydantic objects.
    Returns None if logprobs not present.
    """
    # Convert to dict if it's a pydantic/litellm/groq object
    if not isinstance(response, dict):
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        else:
            return None

    choices = response.get("choices", [])
    if not choices:
        return None
    logprobs = choices[0].get("logprobs")
    if not logprobs or not isinstance(logprobs, dict):
        return None

    # Prefer reasoning logprobs (thinking models) — these capture the
    # model's deliberation about tool selection, which is exactly the
    # uncertainty signal we want.
    # Providers use different keys: "reasoning" (Groq), "reasoning_content" (DeepSeek)
    for reasoning_key in ("reasoning", "reasoning_content"):
        reasoning = logprobs.get(reasoning_key)
        if reasoning and isinstance(reasoning, list) and len(reasoning) > 0:
            return reasoning

    content = logprobs.get("content")
    if content and isinstance(content, list) and len(content) > 0:
        return content

    return None


def _shannon_entropy(top_logprobs: list[dict]) -> float:
    """Compute Shannon entropy from a list of {token, logprob} dicts.

    Logprobs are in natural log. Convert to probabilities, then compute
    H = -sum(p * ln(p)).
    """
    probs = []
    for item in top_logprobs:
        lp = item.get("logprob", -100.0)
        probs.append(math.exp(lp))

    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def _summarize_action(response: dict) -> str:
    """Extract the agent's action from its response for the prompted estimator."""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if not choices:
            return "(no response)"
        msg = choices[0].get("message", {})
    else:
        msg = getattr(getattr(response, "choices", [None])[0], "message", None)
        if msg is None:
            return "(no response)"
        msg = msg.model_dump() if hasattr(msg, "model_dump") else {"content": str(msg)}

    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        parts = []
        for tc in tool_calls[:2]:
            func = tc.get("function", {})
            name = func.get("name", "?")
            args = func.get("arguments", "")
            if len(args) > 200:
                args = args[:200] + "..."
            parts.append(f"Call: {name}({args})")
        return "\n".join(parts)

    content = msg.get("content", "")
    if content:
        return content[:300]
    return "(empty response)"


def _summarize_recent(messages: list[dict], n: int = 4) -> str:
    """Summarize the last N messages for context."""
    recent = messages[-n:] if len(messages) > n else messages
    parts = []
    for m in recent:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        truncated = content[:200] + "..." if len(content) > 200 else content
        parts.append(f"[{role}] {truncated}")
    return "\n".join(parts)


def _parse_score(text: str) -> int:
    """Extract an integer 0-100 from the model's response."""
    import re

    numbers = re.findall(r"\d+", text)
    if not numbers:
        return 50
    score = int(numbers[0])
    return max(0, min(100, score))

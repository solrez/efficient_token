"""Budget-constrained Tau-bench agent.

Extends AdaptiveTauAgent with a hard token budget. The agent stops
when cumulative input tokens exceed the budget. When paired with
BudgetAwareController, compression adapts to remaining budget.
"""
from __future__ import annotations

from adaptive_compression.adaptive_agent import (
    AdaptiveTauAgent,
    AdaptiveStepLog,
    AdaptiveTaskResult,
)
from adaptive_compression.controller import ProfileName, PROFILES
from adaptive_compression.uncertainty import LogprobEstimator
from budget_extension.budget_controller import BudgetAwareController
from token_efficient_agent.config import (
    AgentConfig, ModelTier,
    INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK,
)

import litellm
litellm.suppress_debug_info = True


class BudgetTauAgent(AdaptiveTauAgent):
    """Tau-bench agent with a hard token budget.

    When token_budget is set, the agent stops early if cumulative
    input tokens would exceed the budget. The BudgetAwareController
    adjusts compression thresholds based on remaining budget.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools_info: list[dict],
        wiki: str,
        token_budget: int,
        temperature: float = 0.0,
    ):
        controller = BudgetAwareController(token_budget=token_budget)
        super().__init__(
            config=config,
            tools_info=tools_info,
            wiki=wiki,
            temperature=temperature,
            controller=controller,
        )
        self.token_budget = token_budget

    def solve(self, env, task_index: int, max_num_steps: int = 30):
        from tau_bench.types import RESPOND_ACTION_NAME
        from tau_bench.agents.tool_calling_agent import message_to_action

        self.controller.reset()
        self.controller.token_budget = self.token_budget
        self.controller.tokens_spent = 0
        self._discovered_tools = set()
        self._notes_injected = 0

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
        budget_exceeded = False

        current_profile = PROFILES[ProfileName.PRESERVE]

        for step in range(max_num_steps):
            if self.controller.budget_exhausted:
                budget_exceeded = True
                break

            send_messages = self._apply_compression(messages, step, current_profile)
            send_messages = self._inject_budget_note(send_messages)
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

            self.controller.update_budget(step_input)

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

        stats = self.controller.get_stats()
        stats["budget_exceeded"] = budget_exceeded
        stats["budget_notes_injected"] = self._notes_injected

        return AdaptiveTaskResult(
            task_id=task_index,
            reward=reward,
            total_steps=len(step_logs),
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            step_logs=step_logs,
            controller_stats=stats,
        )

    def _inject_budget_note(self, messages: list[dict]) -> list[dict]:
        """Append a budget status note to the system message when budget is running low."""
        r = self.controller.remaining_ratio

        if r > 0.5:
            return messages

        remaining = self.token_budget - self.controller.tokens_spent

        if r > 0.2:
            note = (
                f"\n\n[Budget notice: ~{remaining:,} tokens remaining out of {self.token_budget:,}. "
                f"Be efficient — avoid unnecessary tool calls and keep responses concise.]"
            )
        elif r > 0.1:
            note = (
                f"\n\n[Budget warning: only ~{remaining:,} tokens remaining. "
                f"Prioritize completing the task. Use minimal tool calls and short responses.]"
            )
        else:
            note = (
                f"\n\n[Budget critical: ~{remaining:,} tokens left. "
                f"You must finish the task now or it will be terminated.]"
            )

        self._notes_injected += 1

        result = list(messages)
        sys_msg = result[0]
        result[0] = {**sys_msg, "content": sys_msg.get("content", "") + note}
        return result

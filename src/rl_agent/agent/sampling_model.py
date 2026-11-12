"""``SamplingModel`` — the actor as an ai-runtime ``BatchModel``: prompts in, sampled responses out."""

from ai_runtime import BatchModel

from src.rl_agent.token_policy_model import TokenPolicyModel


class SamplingModel(BatchModel):
    """Feeds the actor GPU worker's batches to ``TokenPolicyModel.sample`` at the sampling temperature.

    Every prompt weighs 1, so the worker's weight budget is the number of prompts generated for at once.
    """

    def __init__(self, policy: TokenPolicyModel, max_new_tokens: int, temperature: float) -> None:
        self._policy = policy
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def load(self) -> None:
        """Nothing to do: the policy is loaded at construction."""

    def unload(self) -> None:
        """Nothing to do: the policy stays on its device for the other threads that use it."""

    @staticmethod
    def weight(item: str) -> int:
        """One per prompt."""
        return 1

    def process(self, items: list[str], budgets: list) -> list[tuple[list[int], str]]:
        """One ``(response token ids, response text)`` per prompt, in the order given."""
        return self._policy.sample(items, self._max_new_tokens, self._temperature)

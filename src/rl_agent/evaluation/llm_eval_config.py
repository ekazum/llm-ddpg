"""``LLMEvalConfig`` — the evaluation settings of this project (config/eval.yaml)."""

from typing import Any

from pydantic import Field
from rl_core.eval_config import EvalConfig


class LLMEvalConfig(EvalConfig):
    """Models, generation settings, reward model batching and locations of an evaluation job."""

    # Policy and reward model, as Hugging Face ids or local directories.
    policy_model: str
    reward_model: str
    # Devices the two models are loaded on.
    policy_device: str
    reward_device: str
    # Chat template arguments of the policy; the same as in config/prompts.yaml.
    chat_template_kwargs: dict[str, Any]
    # Longest response, in tokens.
    max_new_tokens: int = Field(ge=1)
    # Sampling temperature; 0 generates greedily.
    temperature: float = Field(ge=0.0)
    # Longest (prompt, response) input the reward model reads, in tokens.
    reward_max_tokens: int = Field(ge=1)
    # Reward model batching: pairs per sort window, idle seconds before a flush, characters per batch.
    reward_window_size: int = Field(ge=1)
    reward_idle_timeout_s: float = Field(gt=0.0)
    reward_batch_chars: int = Field(ge=1)
    # Pairs scored one by one and as one batch at start, and the largest score difference accepted.
    reward_batch_check_pairs: int = Field(ge=0)
    reward_batch_tolerance: float = Field(ge=0.0)
    # Directory of the three prompts files, and the directory run records are written under.
    prompts_dir: str
    records_dir: str

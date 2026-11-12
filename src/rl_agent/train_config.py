"""``TrainConfig`` — the training job's settings (config/train.yaml)."""

from typing import Any

from ai_runtime.yaml_config import YamlConfig
from pydantic import Field


class TrainConfig(YamlConfig):
    """Models, sampling, reward model batching, replay buffer, run length and locations of the training job."""

    # Policy (the actor) and reward model, as Hugging Face ids or local directories.
    policy_model: str
    reward_model: str
    # Devices the two models are loaded on.
    policy_device: str
    reward_device: str
    # Chat template arguments of the policy; the same as in config/prompts.yaml.
    chat_template_kwargs: dict[str, Any]
    # Longest response, in tokens; also the width of the replay buffer's response column.
    max_new_tokens: int = Field(ge=1)
    # Sampling temperature of the actor; above 0.
    temperature: float = Field(gt=0.0)
    # Prompts per actor batch, and the most prompts the actor GPU worker generates for at once.
    actor_batch_size: int = Field(ge=1)
    generate_batch_size: int = Field(ge=1)
    # Seconds the actor GPU worker waits for more prompts before it generates for what it has.
    generate_idle_timeout_s: float = Field(gt=0.0)
    # Longest (prompt, response) input the reward model reads, in tokens.
    reward_max_tokens: int = Field(ge=1)
    # Reward model batching: pairs per sort window, idle seconds before a flush, characters per batch.
    reward_window_size: int = Field(ge=1)
    reward_idle_timeout_s: float = Field(gt=0.0)
    reward_batch_chars: int = Field(ge=1)
    # Replay buffer rows, and its prioritized replay settings: priority exponent, first importance
    # weight exponent, draws over which it rises to 1, and the floor added to each TD error.
    replay_capacity: int = Field(ge=1)
    per_alpha: float = Field(ge=0.0, le=1.0)
    per_beta_start: float = Field(ge=0.0, le=1.0)
    per_n_anneal: int = Field(ge=1)
    per_priority_floor: float = Field(gt=0.0)
    # Device the learner trains on.
    learner_device: str
    # Rows per learner batch, and rows stored before the first draw.
    learner_batch_size: int = Field(ge=1)
    min_buffer_size: int = Field(ge=1)
    # Batches drawn per hold of the buffer lock, drawn batches queued ahead, and pinned slots in the pool.
    block_batches: int = Field(ge=1)
    prefetch_depth: int = Field(ge=1)
    pinned_slots: int = Field(ge=1)
    # Learner updates between two buffer stats rows.
    stats_every_updates: int = Field(ge=1)
    # Transitions the actor writes before the job stops.
    max_actor_transitions: int = Field(ge=1)
    # Transitions written per learner update at most; 0 turns the limit off.
    push_per_update_ratio: float = Field(ge=0.0)
    # Seed of the prompt order and of the replay draws.
    seed: int
    # Directory of the three prompts files, and the directory run records are written under.
    prompts_dir: str
    records_dir: str

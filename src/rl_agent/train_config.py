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
    # Devices the critic trains on, one copy each; with two or more, each takes an equal share of every batch and
    # the copies average their gradients.
    learner_devices: list[str] = Field(min_length=1)
    # Rows per learner batch, and rows stored before the first draw.
    learner_batch_size: int = Field(ge=1)
    min_buffer_size: int = Field(ge=1)
    # Batches drawn per hold of the buffer lock, drawn batches queued ahead, and pinned slots in the pool.
    block_batches: int = Field(ge=1)
    prefetch_depth: int = Field(ge=1)
    pinned_slots: int = Field(ge=1)
    # Learner updates between two buffer stats and learner_diag rows.
    stats_every_updates: int = Field(ge=1)
    # Critic learning rate, falling linearly to the minimum over max_updates.
    critic_lr: float = Field(gt=0.0)
    critic_lr_min: float = Field(ge=0.0)
    # Step of the target critic toward the critic after each update, in (0, 1].
    critic_tau: float = Field(gt=0.0, le=1.0)
    # Rows per critic forward and backward pass; a batch is processed in such pieces.
    critic_micro_batch: int = Field(ge=1)
    # Actor learning rate, falling linearly to the minimum over max_updates.
    actor_lr: float = Field(gt=0.0)
    actor_lr_min: float = Field(ge=0.0)
    # Critic updates before the first actor step, and critic updates per actor step after it.
    actor_start_update: int = Field(ge=0)
    actor_every: int = Field(ge=1)
    # Rows per actor forward and backward pass; bounds the (rows, tokens, vocab) probabilities in memory.
    actor_micro_batch: int = Field(ge=1)
    # Critic updates before the job stops.
    max_updates: int = Field(ge=1)
    # Updates between two snapshots (critic and actor); 0 turns snapshots off.
    checkpoint_every: int = Field(ge=0)
    # Keep every snapshot (true), or only the best dev one and the latest (false: the others are deleted).
    keep_snapshots: bool
    # Device of the eval policy, which scores each actor snapshot on dev and the best one on test.
    eval_device: str
    # Prompts generated per eval batch, and the sampling temperature of the eval passes (0 = greedy).
    eval_batch_size: int = Field(ge=1)
    eval_temperature: float = Field(ge=0.0)
    # Dev prompts scored per snapshot, drawn once per run from the seed; test prompts scored (0 = every row).
    dev_limit: int = Field(ge=0)
    test_limit: int = Field(ge=0)
    # Most transitions the actor writes.
    max_actor_transitions: int = Field(ge=1)
    # Transitions stored per learner update, held by both sides; 0 turns the limit off.
    push_per_update_ratio: float = Field(ge=0.0)
    # Seed of the prompt order, the replay draws and the dev and test samples.
    seed: int
    # Directory of the three prompts files, and the directory run records are written under.
    prompts_dir: str
    records_dir: str

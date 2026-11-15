"""``LearnerBatch`` — one drawn batch as ``LearnerProcessor`` hands it to ``LearnerGPUWorker``."""

from dataclasses import dataclass

import numpy as np
import torch

from src.rl_agent.agent.pinned_slot import PinnedSlot


@dataclass(frozen=True)
class LearnerBatch:
    """Host tensors of one batch, all views into ``slot``; ``indices`` are the buffer rows, for the priority update."""

    # (B,) float32: reward of each row, and (B,) bool: whether s' ends the episode.
    reward: torch.Tensor
    done: torch.Tensor
    # (B,) float32: importance weights of the draw.
    is_weights: torch.Tensor
    # (B, T) int64: the critic input, padded on the right to the longest of the batch.
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    # (B,) int64: index of the last token before the response, and the number of critic values (response tokens + 1).
    prompt_end: torch.Tensor
    value_len: torch.Tensor
    # (B,) int64: response tokens the policy chose, from the input index prompt_end + 1 on (its end of turn included).
    action_len: torch.Tensor
    # (B,) int64 buffer rows; host only.
    indices: np.ndarray
    # The slot behind the tensors; the learner returns it to the pool.
    slot: PinnedSlot

"""``QBatch`` — one learner batch on the learner device, as the critic step reads it."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class QBatch:
    """Device tensors of one batch; ``indices`` stay on the host for the priority update."""

    # (B,) float32: reward of each row, and its importance weight.
    reward: torch.Tensor
    is_weights: torch.Tensor
    # (B, T) int64: the critic input, padded on the right.
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    # (B,) int64: index of the last token before the response, and the number of critic values.
    prompt_end: torch.Tensor
    value_len: torch.Tensor
    # (B,) int64 buffer rows.
    indices: np.ndarray

    @property
    def num_rows(self) -> int:
        """Rows in the batch."""
        return int(self.reward.shape[0])

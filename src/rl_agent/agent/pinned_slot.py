"""``PinnedSlot`` — one reusable set of host tensors for a learner batch, pinned when the learner is on a GPU.

Slots live in a pool (a bounded queue): ``LearnerProcessor`` takes one and fills it, ``LearnerGPUWorker`` puts it back
once its copies to the GPU have finished. An empty pool stops the processors, which keeps them behind the learner.
"""

import numpy as np
import torch

from src.rl_agent.agent.trajectory_codec import TrajectoryCodec


class PinnedSlot:
    """The replay columns of a batch, its importance weights and its critic input, each as a torch tensor.

    The replay columns have numpy views over the same memory, so the buffer's ``gather_into`` writes straight in.
    """

    def __init__(self, batch_size: int, max_new_tokens: int, max_input_tokens: int, pin_memory: bool) -> None:
        """Allocate every tensor once; ``max_input_tokens`` is the widest critic input."""
        self._pin_memory = pin_memory
        self.prompt_row = self._tensor((batch_size,), torch.int32)
        self.response_ids = self._tensor((batch_size, max_new_tokens), torch.int32)
        self.response_len = self._tensor((batch_size,), torch.int16)
        self.reward = self._tensor((batch_size,), torch.float32)
        self.done = self._tensor((batch_size,), torch.bool)
        self.is_weights = self._tensor((batch_size,), torch.float32)
        self.input_ids = self._tensor((batch_size, max_input_tokens), torch.int64)
        self.attention_mask = self._tensor((batch_size, max_input_tokens), torch.int64)
        self.prompt_end = self._tensor((batch_size,), torch.int64)
        self.value_len = self._tensor((batch_size,), torch.int64)

    def replay_views(self) -> dict[str, np.ndarray]:
        """Numpy views of the replay columns, named as in ``TrajectoryCodec``."""
        return {
            TrajectoryCodec.PROMPT_ROW: self.prompt_row.numpy(),
            TrajectoryCodec.RESPONSE_IDS: self.response_ids.numpy(),
            TrajectoryCodec.RESPONSE_LEN: self.response_len.numpy(),
            TrajectoryCodec.REWARD: self.reward.numpy(),
            TrajectoryCodec.DONE: self.done.numpy(),
        }

    def _tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """An uninitialized host tensor, pinned when the slot is."""
        tensor = torch.empty(shape, dtype=dtype)
        return tensor.pin_memory() if self._pin_memory else tensor

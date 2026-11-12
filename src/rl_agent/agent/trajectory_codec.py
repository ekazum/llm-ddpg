"""``TrajectoryCodec`` — the replay row layout of this project."""

import numpy as np
from rl_core.dense_codec import DenseCodec


class TrajectoryCodec(DenseCodec):
    """rl-core's ``DenseCodec`` over the fields of ``TrajectoryBatch``: one column per field."""

    PROMPT_ROW = "prompt_row"
    RESPONSE_IDS = "response_ids"
    RESPONSE_LEN = "response_len"
    REWARD = "reward"
    DONE = "done"

    def __init__(self, max_new_tokens: int) -> None:
        """The response column is ``max_new_tokens`` wide."""
        super().__init__({
            self.PROMPT_ROW: ((), np.int32),
            self.RESPONSE_IDS: ((max_new_tokens,), np.int32),
            self.RESPONSE_LEN: ((), np.int16),
            self.REWARD: ((), np.float32),
            self.DONE: ((), np.bool_),
        })

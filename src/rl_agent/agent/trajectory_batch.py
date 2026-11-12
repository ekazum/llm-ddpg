"""``TrajectoryBatch`` — one actor batch of transitions, as the replay writer pushes it."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryBatch:
    """s = the prompt (its train row), a = the response, r = the frozen reward model score, s' terminal.

    s' is the prompt followed by the response, so it is not stored apart.
    """

    # (n,) int32: row of the prompt in the train prompts file.
    prompt_row: np.ndarray
    # (n, max_new_tokens) int32: response token ids, padded with the padding id.
    response_ids: np.ndarray
    # (n,) int16: tokens in each response.
    response_len: np.ndarray
    # (n,) float32: reward of each (prompt, response).
    reward: np.ndarray
    # (n,) bool: whether s' ends the episode; always true while an action is the whole response.
    done: np.ndarray

    @property
    def num_rows(self) -> int:
        """Transitions in the batch."""
        return int(self.prompt_row.shape[0])

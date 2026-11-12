"""``PromptSampler`` — the train prompts in a fresh random order per pass."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rl_core.row_sampler import RowSampler

from pipeline.prompt_schema import PromptSchema

logger = logging.getLogger(__name__)


class PromptSampler(RowSampler):
    """rl-core's ``RowSampler`` over the rows of the train prompts file, with the prompt text of each row."""

    def __init__(self, prompts_dir: str, seed_seq: np.random.SeedSequence) -> None:
        """Read the prompt column of the train file; a row is its position in that file."""
        path = Path(prompts_dir) / PromptSchema.file_name("train")
        self._prompts: list[str] = pd.read_parquet(path, columns=[PromptSchema.PROMPT])[PromptSchema.PROMPT].tolist()
        super().__init__(len(self._prompts), seed_seq)
        logger.info("PromptSampler: %d train prompts from %s", len(self._prompts), path)

    def prompts(self, rows: list[int]) -> list[str]:
        """The prompt text of each row, in the order given."""
        return [self._prompts[row] for row in rows]

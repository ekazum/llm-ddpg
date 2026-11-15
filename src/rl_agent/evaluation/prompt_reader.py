"""``PromptReader`` — reads the prompts file of one split for an evaluation pass."""

from pathlib import Path

import pandas as pd

from pipeline.prompt_schema import PromptSchema


class PromptReader:
    """The three prompts files under one directory, cut to the row limit of the job.

    Without a seed the cut keeps the first ``limit`` rows; with one, a random sample of ``limit`` rows drawn from
    the seed, the same rows on every read.
    """

    def __init__(self, prompts_dir: str, limit: int, seed: int | None = None) -> None:
        self._dir = Path(prompts_dir)
        self._limit = limit
        self._seed = seed

    def check(self) -> None:
        """Raise unless every split's prompts file exists."""
        missing = [str(self._path(split)) for split in PromptSchema.SPLITS if not self._path(split).is_file()]
        if missing:
            raise FileNotFoundError(f"PromptReader: missing prompts files {missing}; run scripts/run_prompts.sh")

    def read(self, split: str) -> pd.DataFrame:
        """The split's rows, cut to ``limit`` when a limit is set, longest prompt first."""
        frame = pd.read_parquet(self._path(split))
        if 0 < self._limit < len(frame):
            frame = frame.head(self._limit) if self._seed is None else frame.sample(
                n=self._limit, random_state=self._seed)
        return frame.sort_values(PromptSchema.PROMPT_TOKENS, ascending=False).reset_index(drop=True)

    def _path(self, split: str) -> Path:
        return self._dir / PromptSchema.file_name(split)

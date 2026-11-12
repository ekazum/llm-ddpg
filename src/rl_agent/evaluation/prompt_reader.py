"""``PromptReader`` — reads the prompts file of one split for an evaluation pass."""

from pathlib import Path

import pandas as pd

from pipeline.prompt_schema import PromptSchema


class PromptReader:
    """The three prompts files under one directory, cut to the row limit of the job."""

    def __init__(self, prompts_dir: str, limit: int) -> None:
        self._dir = Path(prompts_dir)
        self._limit = limit

    def check(self) -> None:
        """Raise unless every split's prompts file exists."""
        missing = [str(self._path(split)) for split in PromptSchema.SPLITS if not self._path(split).is_file()]
        if missing:
            raise FileNotFoundError(f"PromptReader: missing prompts files {missing}; run scripts/run_prompts.sh")

    def read(self, split: str) -> pd.DataFrame:
        """The split's rows, the first ``limit`` when a limit is set, longest prompt first."""
        frame = pd.read_parquet(self._path(split))
        if self._limit > 0:
            frame = frame.head(self._limit)
        return frame.sort_values(PromptSchema.PROMPT_TOKENS, ascending=False).reset_index(drop=True)

    def _path(self, split: str) -> Path:
        return self._dir / PromptSchema.file_name(split)

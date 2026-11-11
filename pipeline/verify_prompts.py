"""``VerifyPrompts`` — checks on the three prompts files; the stats JSON exists only when every check passed.

    python pipeline/verify_prompts.py --config config/prompts.yaml --input PROMPTS_TRAIN PROMPTS_DEV PROMPTS_TEST \\
        --output STATS/verify_prompts.json

Per file: the schema columns and no others, no nulls, no empty prompts, unique keys, ``split`` equal to the
file's split, ``prompt_tokens`` between 1 and ``max_prompt_tokens``, at least ``min_rows[split]`` rows.
Across files: no key in two files, no dev or test prompt text in train.
"""

import argparse
import logging
import os

import pandas as pd
from ai_pipeline import VerifyStage
from ai_runtime.logging import setup_logging

from prompt_pipeline_config import PromptPipelineConfig
from prompt_schema import PromptSchema

logger = logging.getLogger(__name__)


class VerifyPrompts(VerifyStage):
    """``VerifyStage`` over the train, dev and test prompts files."""

    STATS_FILE = "verify_prompts.json"

    def __init__(
        self,
        config: PromptPipelineConfig,
        input_paths: list[str],
        output_path: str,
    ) -> None:
        if len(input_paths) != len(PromptSchema.SPLITS):
            raise ValueError(f"VerifyPrompts: expected {len(PromptSchema.SPLITS)} inputs {PromptSchema.SPLITS}, "
                             f"got {len(input_paths)}")
        if os.path.basename(output_path) != self.STATS_FILE:
            raise ValueError(f"VerifyPrompts: output must be <stats_dir>/{self.STATS_FILE}, got {output_path}")
        super().__init__(input_paths, output_path, os.path.dirname(os.path.abspath(output_path)))
        self._config = config
        self.module_name = "verify_prompts"
        self.phase = "verify"
        self._stats: dict = {}

    @staticmethod
    def normalized(prompts: pd.Series) -> set[str]:
        """Prompt texts under ``PromptSchema.normalized_prompt``, for the leak check."""
        return {PromptSchema.normalized_prompt(text) for text in prompts.tolist()}

    def check_file(self, split: str, df: pd.DataFrame) -> list[str]:
        """Errors of one prompts file."""
        errors: list[str] = []
        missing = [c for c in PromptSchema.COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in PromptSchema.COLUMNS]
        if missing:
            errors.append(f"{split}: missing columns {missing}")
        if extra:
            errors.append(f"{split}: unexpected columns {extra}")
        if missing:
            return errors
        for column in PromptSchema.COLUMNS:
            if nulls := self.null_count(df, column):
                errors.append(f"{split}: {nulls} null values in '{column}'")
        for column in (*PromptSchema.TEXT_COLUMNS, PromptSchema.SPLIT):
            if not pd.api.types.is_string_dtype(df[column]):
                errors.append(f"{split}: '{column}' is {df[column].dtype}, not a string column")
        if empty := int((df[PromptSchema.PROMPT].str.strip().str.len() == 0).sum()):
            errors.append(f"{split}: {empty} empty prompts")
        if dupes := self.duplicate_count(df, PromptSchema.KEY):
            errors.append(f"{split}: {dupes} duplicate keys")
        if wrong_split := int((df[PromptSchema.SPLIT] != split).sum()):
            errors.append(f"{split}: {wrong_split} rows whose split column is not '{split}'")
        tokens = df[PromptSchema.PROMPT_TOKENS]
        if not pd.api.types.is_integer_dtype(tokens):
            errors.append(f"{split}: '{PromptSchema.PROMPT_TOKENS}' is {tokens.dtype}, not an integer column")
        else:
            if below := int((tokens < 1).sum()):
                errors.append(f"{split}: {below} rows with prompt_tokens below 1")
            if above := int((tokens > self._config.max_prompt_tokens).sum()):
                errors.append(f"{split}: {above} rows with prompt_tokens above {self._config.max_prompt_tokens}")
        minimum = self._config.min_rows.get(split, 0)
        if len(df) < minimum:
            errors.append(f"{split}: {len(df)} rows, below min_rows {minimum}")
        return errors

    def check_across(self, frames: dict[str, pd.DataFrame]) -> list[str]:
        """Errors between files: shared keys, dev or test prompts present in train."""
        errors: list[str] = []
        splits = list(frames)
        for i, first in enumerate(splits):
            first_keys = set(frames[first][PromptSchema.KEY].tolist())
            for second in splits[i + 1:]:
                shared = first_keys & set(frames[second][PromptSchema.KEY].tolist())
                if shared:
                    errors.append(f"{first}/{second}: {len(shared)} keys in both files")
        train_prompts = self.normalized(frames["train"][PromptSchema.PROMPT])
        for split in ("dev", "test"):
            leaked = self.normalized(frames[split][PromptSchema.PROMPT]) & train_prompts
            if leaked:
                errors.append(f"{split}: {len(leaked)} prompts also in train")
        return errors

    def run(self) -> None:
        frames = {split: pd.read_parquet(path) for split, path in zip(PromptSchema.SPLITS, self.input_paths)}
        errors: list[str] = []
        for split, df in frames.items():
            errors.extend(self.check_file(split, df))
        if not errors:
            errors.extend(self.check_across(frames))
        self.raise_on_errors(errors)
        rows = {split: len(df) for split, df in frames.items()}
        self._stats = {"rows": rows, "rows_total": sum(rows.values()), "validation_passed": True}
        logger.info("%s: passed, rows %s", self.module_name, rows)
        self.write_stats()

    def collect_stats(self) -> dict:
        return dict(self._stats)


def main() -> None:
    """Command line: config, three inputs in split order, the stats JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PromptPipelineConfig YAML.")
    parser.add_argument("--input", nargs=3, required=True, metavar=("TRAIN", "DEV", "TEST"))
    parser.add_argument("--output", required=True, help=f"<stats_dir>/{VerifyPrompts.STATS_FILE}")
    args = parser.parse_args()
    setup_logging()
    VerifyPrompts(PromptPipelineConfig.from_yaml(args.config), args.input, args.output).run()


if __name__ == "__main__":
    main()

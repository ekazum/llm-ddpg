"""``SplitDev`` — split the train rows into train and dev by a seeded hash of the key; the test rows pass through.

    python pipeline/split_dev.py --config config/prompts.yaml --input FILTERED_TRAIN FILTERED_TEST \\
        --output PROMPTS_TRAIN PROMPTS_DEV PROMPTS_TEST --stats-dir STATS

The ``dev_size`` rows whose ``sha256("<hash_seed>:<key>")`` ranks lowest become the dev split, so the
split is the same on every run and every machine. Train rows whose prompt text, under
``PromptSchema.normalized_prompt``, appears in dev or test are dropped, so no evaluation prompt is trained on.
Each output gains the ``split`` column.
"""

import argparse
import hashlib
import logging

import numpy as np
import pandas as pd
from ai_pipeline import Stage
from ai_runtime.logging import setup_logging

from prompt_pipeline_config import PromptPipelineConfig
from prompt_schema import PromptSchema

logger = logging.getLogger(__name__)


class SplitDev(Stage):
    """Two filtered inputs (train source, test source) to three prompts files (train, dev, test)."""

    def __init__(
        self,
        config: PromptPipelineConfig,
        input_paths: list[str],
        output_paths: list[str],
        stats_dir: str,
    ) -> None:
        if len(input_paths) != 2:
            raise ValueError(f"SplitDev: expected 2 inputs (train source, test source), got {len(input_paths)}")
        if len(output_paths) != len(PromptSchema.SPLITS):
            raise ValueError(f"SplitDev: expected {len(PromptSchema.SPLITS)} outputs {PromptSchema.SPLITS}, "
                             f"got {len(output_paths)}")
        super().__init__(input_paths, output_paths, stats_dir)
        self._config = config
        self._output_paths = list(output_paths)
        self.module_name = "split_dev"
        self.phase = "split"
        self._stats: dict = {}

    def dev_mask(self, keys: pd.Series) -> np.ndarray:
        """True for the ``dev_size`` keys with the lowest seeded hash."""
        dev_size = self._config.dev_size
        if dev_size >= len(keys):
            raise ValueError(f"{self.module_name}: dev_size {dev_size} must be below the train row count {len(keys)}")
        seed = self._config.hash_seed
        ranks = np.array([hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest() for key in keys])
        order = np.argsort(ranks, kind="stable")
        mask = np.zeros(len(keys), dtype=bool)
        mask[order[:dev_size]] = True
        return mask

    def run(self) -> None:
        train_source = pd.read_parquet(self.input_paths[0])
        test_source = pd.read_parquet(self.input_paths[1])
        is_dev = self.dev_mask(train_source[PromptSchema.KEY])
        dev = train_source[is_dev]
        train = train_source[~is_dev]
        held_out = {
            PromptSchema.normalized_prompt(text)
            for text in pd.concat([dev[PromptSchema.PROMPT], test_source[PromptSchema.PROMPT]]).tolist()
        }
        in_held_out = train[PromptSchema.PROMPT].map(PromptSchema.normalized_prompt).isin(held_out)
        rows_dropped = int(in_held_out.sum())
        train = train[~in_held_out]
        frames = {"train": train, "dev": dev, "test": test_source}
        counts: dict[str, int] = {}
        for split, path in zip(PromptSchema.SPLITS, self._output_paths):
            df = (frames[split].assign(**{PromptSchema.SPLIT: split})
                  .astype({PromptSchema.SPLIT: "string"}).reset_index(drop=True))
            self._write_parquet(df, path)
            counts[split] = len(df)
            logger.info("%s: %s %d rows -> %s", self.module_name, split, len(df), path)
        self._stats = {
            "rows": counts,
            "rows_dropped_train_prompt_in_dev_or_test": rows_dropped,
            "dev_size": self._config.dev_size,
            "hash_seed": self._config.hash_seed,
        }
        logger.info("%s: dropped %d train rows whose prompt is in dev or test", self.module_name, rows_dropped)
        self.write_stats()

    def collect_stats(self) -> dict:
        return dict(self._stats)


def main() -> None:
    """Command line: config, two inputs, three outputs in split order, the stats directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PromptPipelineConfig YAML.")
    parser.add_argument("--input", nargs=2, required=True, metavar=("TRAIN_SOURCE", "TEST_SOURCE"))
    parser.add_argument("--output", nargs=3, required=True, metavar=("TRAIN", "DEV", "TEST"))
    parser.add_argument("--stats-dir", required=True)
    args = parser.parse_args()
    setup_logging()
    SplitDev(PromptPipelineConfig.from_yaml(args.config), args.input, args.output, args.stats_dir).run()


if __name__ == "__main__":
    main()

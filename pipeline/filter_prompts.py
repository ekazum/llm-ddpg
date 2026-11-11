"""``FilterPrompts`` — keep the prompts a policy can take, and add their templated token count.

    python pipeline/filter_prompts.py --config config/prompts.yaml --input RAW --output FILTERED --stats-dir STATS

Drops rows with a null or empty text column, rows whose key repeats an earlier row's, and prompts
whose chat-templated token count exceeds ``max_prompt_tokens``. Adds ``prompt_tokens``.
Stats go to ``<stats-dir>/filter_<output basename>.json``.
"""

import argparse
import logging
import os
from typing import Any, Optional, cast

import pandas as pd
from ai_pipeline import FilterStage
from ai_runtime.logging import setup_logging
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from prompt_pipeline_config import PromptPipelineConfig
from prompt_schema import PromptSchema

logger = logging.getLogger(__name__)


class FilterPrompts(FilterStage):
    """``FilterStage`` over the extract columns: text present, keys unique, templated length within the limit."""

    def __init__(
        self,
        config: PromptPipelineConfig,
        input_path: str,
        output_path: str,
        stats_dir: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__([input_path], output_path, stats_dir)
        self._config = config
        # The auto class's stub has no return type; it builds a PreTrainedTokenizerBase.
        self._tokenizer: PreTrainedTokenizerBase = (
            tokenizer if tokenizer is not None
            else cast(Any, AutoTokenizer).from_pretrained(config.tokenizer)
        )
        basename = os.path.splitext(os.path.basename(output_path))[0]
        self.module_name = f"filter_{basename}"
        self.phase = "filter"
        self._stats: dict = {}

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop empty, duplicate and over-long prompts; add ``prompt_tokens``."""
        missing = [c for c in PromptSchema.EXTRACT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{self.module_name}: input lacks columns {missing}")
        rows_in = len(df)
        text_columns = list(PromptSchema.TEXT_COLUMNS)
        df = df.dropna(subset=text_columns)
        non_empty = (df[text_columns].apply(lambda s: s.str.strip().str.len() > 0)).all(axis=1)
        df = df[non_empty]
        rows_text = len(df)
        df = df.drop_duplicates(subset=[PromptSchema.KEY], keep="first")
        rows_unique = len(df)
        df = df.assign(**{PromptSchema.PROMPT_TOKENS: self.prompt_tokens(df[PromptSchema.PROMPT].tolist())})
        within = df[PromptSchema.PROMPT_TOKENS] <= self._config.max_prompt_tokens
        df = df[within].reset_index(drop=True)
        self._stats = {
            "rows_in": rows_in,
            "rows_dropped_empty": rows_in - rows_text,
            "rows_dropped_duplicate_key": rows_text - rows_unique,
            "rows_dropped_over_length": rows_unique - len(df),
            "rows_out": len(df),
            "max_prompt_tokens": self._config.max_prompt_tokens,
            "prompt_tokens_max": int(df[PromptSchema.PROMPT_TOKENS].max()) if len(df) else 0,
            "prompt_tokens_mean": float(df[PromptSchema.PROMPT_TOKENS].mean()) if len(df) else 0.0,
        }
        logger.info("%s: %d rows in, %d out", self.module_name, rows_in, len(df))
        return df

    def prompt_tokens(self, prompts: list[str]) -> list[int]:
        """Token count of each prompt as one user turn under the chat template, generation prompt included."""
        texts: list[str] = [
            str(self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **self._config.chat_template_kwargs,
            ))
            for prompt in prompts
        ]
        encoded = self._tokenizer(texts, add_special_tokens=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def collect_stats(self) -> dict:
        return dict(self._stats)


def main() -> None:
    """Command line: config, one input parquet, one output parquet, the stats directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PromptPipelineConfig YAML.")
    parser.add_argument("--input", required=True, help="Parquet with the extract columns.")
    parser.add_argument("--output", required=True, help="Output parquet.")
    parser.add_argument("--stats-dir", required=True)
    args = parser.parse_args()
    setup_logging()
    FilterPrompts(PromptPipelineConfig.from_yaml(args.config), args.input, args.output, args.stats_dir).run()


if __name__ == "__main__":
    main()

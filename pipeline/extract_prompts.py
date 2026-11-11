"""Extract one UltraFeedback preference source into the prompts extract columns.

    python pipeline/extract_prompts.py --config config/pipeline.yaml --source NAME --input RAW --output OUT

ultrafeedback_binarized stores ``chosen`` and ``rejected`` as message lists (user turn, assistant turn);
the text kept is the last message's content when its role is ``assistant``, else null, which the filter
stage drops. ``prompt`` and ``prompt_id`` are copied as they are.
"""

import logging
from typing import Optional, Sequence

import pandas as pd
from ai_pipeline import ExtractStage

from cli import parse
from pipeline_config import PipelineConfig
from prompt_schema import PromptSchema

logger = logging.getLogger(__name__)


class ExtractPrompts(ExtractStage):
    """One raw ultrafeedback_binarized parquet to ``PromptSchema.EXTRACT_COLUMNS``."""

    SOURCE_COLUMNS = ("prompt_id", "prompt", "chosen", "rejected")

    def __init__(self, config: PipelineConfig, source: str,
                 input_paths: list[str], output_path: str) -> None:
        super().__init__(input_paths, output_path, config.stats_dir)
        self.module_name = f"extract_{source}"
        self.phase = "extract"
        self._stats: dict = {}

    @staticmethod
    def last_assistant_text(messages: Optional[Sequence[dict]]) -> Optional[str]:
        """Content of the last message when it is the assistant's, else None."""
        if messages is None or len(messages) == 0:
            return None
        last = messages[-1]
        if last.get("role") != "assistant":
            return None
        return last.get("content")

    def extract(self) -> pd.DataFrame:
        raw = pd.read_parquet(self.input_paths[0], columns=list(self.SOURCE_COLUMNS))
        logger.info("Loaded %s: %d rows", self.input_paths[0], len(raw))
        df = pd.DataFrame({
            PromptSchema.KEY: raw["prompt_id"].astype("string"),
            PromptSchema.PROMPT: raw["prompt"].astype("string"),
            PromptSchema.CHOSEN: raw["chosen"].map(self.last_assistant_text).astype("string"),
            PromptSchema.REJECTED: raw["rejected"].map(self.last_assistant_text).astype("string"),
        })
        self._stats = {
            "rows": len(df),
            "rows_without_chosen_reply": int(df[PromptSchema.CHOSEN].isna().sum()),
            "rows_without_rejected_reply": int(df[PromptSchema.REJECTED].isna().sum()),
        }
        return df

    def collect_stats(self) -> dict:
        return dict(self._stats)


def main() -> None:
    """Command line entry: one source name, one raw parquet, one output parquet."""
    args, config = parse(__doc__, "source")
    ExtractPrompts(config, args.source, args.input, args.output[0]).run()


if __name__ == "__main__":
    main()

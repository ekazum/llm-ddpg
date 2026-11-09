"""Merge every stage's stats JSON into <DATA_DIR>/stats/pipeline_stats.json.

    python pipeline/merge_stats.py --config config/pipeline.yaml --input STATS_JSON... --output PIPELINE_STATS

The summary's row count is ``rows_total`` from verify_prompts.json.
"""

from ai_pipeline import MergeStatsStage

from cli import parse
from pipeline_config import PipelineConfig


class MergeStats(MergeStatsStage):
    """``MergeStatsStage`` with the project's names."""

    def __init__(self, config: PipelineConfig, input_paths: list[str],
                 output_path: str) -> None:
        super().__init__(input_paths, output_path, config.stats_dir)
        self.module_name = "merge_stats"
        self.phase = "merge"
        self.count_key = "rows_total"


def main() -> None:
    """Command line entry: the stats JSON files that gate the merge, one output path."""
    args, config = parse(__doc__)
    MergeStats(config, args.input, args.output[0]).run()


if __name__ == "__main__":
    main()

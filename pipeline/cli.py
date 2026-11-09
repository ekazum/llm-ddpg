"""The command line the project's stage modules share: --config, --input, --output and named extras."""

import argparse
from typing import Optional

from ai_runtime.logging import setup_logging

from pipeline_config import PipelineConfig


def parse(description: Optional[str], *extra: str) -> tuple[argparse.Namespace, PipelineConfig]:
    """Parse --config, --input, --output and each ``--<extra>``; set up logging; load the config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="config/pipeline.yaml")
    parser.add_argument("--input", nargs="*", default=[])
    parser.add_argument("--output", nargs="+", required=True)
    for name in extra:
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    setup_logging()
    return args, PipelineConfig.from_yaml(args.config)

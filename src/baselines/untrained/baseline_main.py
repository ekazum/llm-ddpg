"""Entry point of the baseline job: the untrained policy scored on dev and on test.

    python -m src.baselines.untrained.baseline_main --config config/eval.yaml
"""

import argparse

from ai_runtime.logging import setup_logging

from src.baselines.untrained.baseline_manager import BaselineManager
from src.rl_agent.evaluation.llm_eval_config import LLMEvalConfig


def main() -> None:
    """Load the config and run the baseline job."""
    parser = argparse.ArgumentParser(description="Score the untrained policy on dev and on test.")
    parser.add_argument("--config", required=True, help="config/eval.yaml")
    args = parser.parse_args()
    setup_logging()
    BaselineManager(LLMEvalConfig.from_yaml(args.config)).run()


if __name__ == "__main__":
    main()

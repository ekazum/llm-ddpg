"""Entry point of the training job.

    python -m src.rl_agent.rl_agent_main --config config/train.yaml
"""

import argparse

from ai_runtime.logging import setup_logging

from src.rl_agent.rl_manager import RLManager
from src.rl_agent.train_config import TrainConfig


def main() -> None:
    """Load the config and run the training job."""
    parser = argparse.ArgumentParser(description="Run the training job.")
    parser.add_argument("--config", required=True, help="config/train.yaml")
    args = parser.parse_args()
    setup_logging()
    RLManager(TrainConfig.from_yaml(args.config)).run()


if __name__ == "__main__":
    main()

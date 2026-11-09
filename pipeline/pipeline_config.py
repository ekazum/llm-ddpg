"""``PipelineConfig`` — the project's prompt pipeline workflow settings (config/pipeline.yaml)."""

import os
from typing import Any

from ai_runtime.yaml_config import YamlConfig


class PipelineConfig(YamlConfig):
    """Data directory, raw sources, which source is train and which is test, and the stage config paths."""

    # Every pipeline output goes under this directory.
    DATA_DIR: str
    # Raw parquet files by source name.
    SOURCES: dict[str, str]
    # The source split into train and dev, and the source that is the test split.
    TRAIN_SOURCE: str
    TEST_SOURCE: str
    # PromptPipelineConfig YAML for rl_llm's filter, split and verify stages.
    PROMPTS_CONFIG: str
    # Hugging Face cache holding the tokenizer; jobs run offline against it.
    HF_HOME: str

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        sources = raw.get("SOURCES", {})
        for key in ("TRAIN_SOURCE", "TEST_SOURCE"):
            if raw.get(key) not in sources:
                raise ValueError(f"PipelineConfig: {key} must name one of SOURCES")
        if raw.get("TRAIN_SOURCE") == raw.get("TEST_SOURCE"):
            raise ValueError("PipelineConfig: TRAIN_SOURCE and TEST_SOURCE must differ")

    def path(self, *parts: str) -> str:
        """A path under DATA_DIR."""
        return os.path.join(self.DATA_DIR, *parts)

    @property
    def stats_dir(self) -> str:
        """Directory of every stage's stats JSON."""
        return self.path("stats")

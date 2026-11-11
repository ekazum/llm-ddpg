"""``PromptPipelineConfig`` — settings of the prompt pipeline stages (``FilterPrompts``, ``SplitDev``, ``VerifyPrompts``).

Loads from YAML through ``from_yaml``. The file is ``config/prompts.yaml``, named in ``config/pipeline.yaml``.
"""

from typing import Any

from ai_runtime.yaml_config import YamlConfig

from prompt_schema import PromptSchema


class PromptPipelineConfig(YamlConfig):
    """Tokenizer, prompt length limit, dev split size and the row minimums the verify stage accepts."""

    # Hugging Face tokenizer id or local path; its chat template measures a prompt's length.
    tokenizer: str
    # Keyword arguments passed to apply_chat_template, e.g. {enable_thinking: false}.
    chat_template_kwargs: dict[str, Any] = {}
    # Prompts whose templated token count exceeds this are dropped; positive.
    max_prompt_tokens: int
    # Rows moved from the train source into the dev split; positive and below the train row count.
    dev_size: int
    # Seed mixed into the key hash that ranks the rows for the dev split.
    hash_seed: int = 0
    # Minimum rows per split that VerifyPrompts accepts, e.g. {train: 50000, dev: 2000, test: 1000}.
    min_rows: dict[str, int] = {}

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        for key in ("tokenizer", "max_prompt_tokens", "dev_size"):
            if key not in raw:
                raise ValueError(f"PromptPipelineConfig: {key} is required")
        if int(raw["max_prompt_tokens"]) <= 0:
            raise ValueError("PromptPipelineConfig: max_prompt_tokens must be positive")
        if int(raw["dev_size"]) <= 0:
            raise ValueError("PromptPipelineConfig: dev_size must be positive")
        unknown = set(raw.get("min_rows", {})) - set(PromptSchema.SPLITS)
        if unknown:
            raise ValueError(f"PromptPipelineConfig: min_rows names unknown splits {sorted(unknown)}")

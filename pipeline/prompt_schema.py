"""``PromptSchema`` — the columns of a prompts parquet: the contract between the prompt pipeline and its readers.

An extract stage writes ``EXTRACT_COLUMNS``; ``FilterPrompts`` adds ``PROMPT_TOKENS``;
``SplitDev`` adds ``SPLIT`` and writes one file per split, named by ``file_name``.
"""


class PromptSchema:
    """Column names, split names and file names of the prompts parquet files."""

    # Unique string key of a prompt; the same value never appears in two splits.
    KEY = "prompt_id"
    # The user turn the policy answers, as plain text without a chat template.
    PROMPT = "prompt"
    # Reference replies from the source preference pair, as plain text.
    CHOSEN = "chosen"
    REJECTED = "rejected"
    # Token count of the prompt after the tokenizer's chat template.
    PROMPT_TOKENS = "prompt_tokens"
    # One of SPLITS; equal to the file's split.
    SPLIT = "split"

    SPLITS = ("train", "dev", "test")
    TEXT_COLUMNS = (KEY, PROMPT, CHOSEN, REJECTED)
    EXTRACT_COLUMNS = (KEY, PROMPT, CHOSEN, REJECTED)
    COLUMNS = (KEY, PROMPT, CHOSEN, REJECTED, PROMPT_TOKENS, SPLIT)

    @staticmethod
    def normalized_prompt(text: str) -> str:
        """The prompt text as compared across splits: whitespace runs collapsed, case folded."""
        return " ".join(text.split()).casefold()

    @staticmethod
    def file_name(split: str) -> str:
        """The parquet file name of one split: ``prompts_<split>.parquet``."""
        if split not in PromptSchema.SPLITS:
            raise ValueError(f"PromptSchema: unknown split {split!r}; expected one of {PromptSchema.SPLITS}")
        return f"prompts_{split}.parquet"

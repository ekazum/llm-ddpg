"""``RewardModelTokenizer`` — the reward model's input: (prompt, response) as two turns of its chat template."""

from typing import Any, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class RewardModelTokenizer:
    """Formats and tokenizes pairs exactly as the reward model reads them; the frozen model and the critic share it.

    A user turn and an assistant turn, no system prompt, the template's own start token removed, right padding.
    """

    def __init__(self, model_name: str, max_tokens: int) -> None:
        """Load the reward model's tokenizer; inputs are cut at ``max_tokens``."""
        self._max_tokens = max_tokens
        # The auto class's stub has no return type; it builds a PreTrainedTokenizerBase.
        self._tokenizer: PreTrainedTokenizerBase = cast(Any, AutoTokenizer).from_pretrained(model_name)
        self._tokenizer.padding_side = "right"

    @property
    def max_tokens(self) -> int:
        """Longest input, in tokens."""
        return self._max_tokens

    @property
    def pad_token_id(self) -> int:
        """The id that pads an input after its end."""
        return int(self._tokenizer.pad_token_id)

    def encode(self, prompts: list[str], responses: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(input_ids, attention_mask)``, both ``(n, longest input)`` int64, one row per pair."""
        texts = [self._format(prompt, response) for prompt, response in zip(prompts, responses)]
        encoded = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self._max_tokens)
        return encoded["input_ids"], encoded["attention_mask"]

    def _format(self, prompt: str, response: str) -> str:
        """The pair as two turns under the chat template, without the template's own start token."""
        text = str(self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False,
        ))
        bos = self._tokenizer.bos_token
        # The tokenizer adds its start token itself; the model card strips the template's copy.
        if isinstance(bos, str) and text.startswith(bos):
            text = text[len(bos):]
        return text

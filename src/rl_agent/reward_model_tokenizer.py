"""``RewardModelTokenizer`` — the reward model's input: (prompt, response) as two turns of its chat template."""

from typing import Any, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class RewardModelTokenizer:
    """Formats and tokenizes pairs exactly as the reward model reads them; the frozen model and the critic share it.

    A user turn and an assistant turn, no system prompt, the template's own start token removed, right padding.
    """

    # Stands in for the response when the template is split into the text before and after it.
    _RESPONSE_MARK = "@@RESPONSE@@"

    def __init__(self, model_name: str, max_tokens: int) -> None:
        """Load the reward model's tokenizer; inputs are cut at ``max_tokens``."""
        self._max_tokens = max_tokens
        # The auto class's stub has no return type; it builds a PreTrainedTokenizerBase.
        self._tokenizer: PreTrainedTokenizerBase = cast(Any, AutoTokenizer).from_pretrained(model_name)
        self._tokenizer.padding_side = "right"
        # The template's text after the response, the same for every pair.
        self._tail_ids: list[int] = self._token_ids([self._split("")[1]])[0]

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

    def encode_ids(
        self, prompts: list[str], response_ids: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The critic input from the policy's own response tokens: the template's text, then the ids, then its end.

        Returns ``(input_ids, attention_mask, prompt_end, value_len)``: ``(n, longest)`` int64 padded on the right,
        the index of the last token before the response, and the number of critic values (response tokens + 1).
        A trailing end-of-turn id is dropped, since the template's end adds it back. Needs the policy's vocabulary.
        """
        prefixes = self._token_ids([self._split(prompt)[0] for prompt in prompts])
        eos = self._tokenizer.eos_token_id
        rows: list[list[int]] = []
        prompt_end: list[int] = []
        value_len: list[int] = []
        for prefix, ids in zip(prefixes, response_ids):
            body = ids[:-1] if ids and ids[-1] == eos else ids
            rows.append(prefix + body + self._tail_ids)
            prompt_end.append(len(prefix) - 1)
            value_len.append(len(body) + 1)
        longest = max(len(row) for row in rows)
        if longest > self._max_tokens:
            raise ValueError(f"RewardModelTokenizer: a critic input has {longest} tokens; the limit is "
                             f"{self._max_tokens}")
        input_ids = torch.full((len(rows), longest), self.pad_token_id, dtype=torch.int64)
        attention_mask = torch.zeros((len(rows), longest), dtype=torch.int64)
        for i, row in enumerate(rows):
            input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.int64)
            attention_mask[i, :len(row)] = 1
        return (input_ids, attention_mask, torch.tensor(prompt_end, dtype=torch.int64),
                torch.tensor(value_len, dtype=torch.int64))

    def _split(self, prompt: str) -> tuple[str, str]:
        """The pair's template text before and after the response."""
        before, after = self._format(prompt, self._RESPONSE_MARK).split(self._RESPONSE_MARK)
        return before, after

    def _token_ids(self, texts: list[str]) -> list[list[int]]:
        """The ids of each text, tokenized as ``encode`` does, without padding."""
        encoded: list[list[int]] = self._tokenizer(texts)["input_ids"]
        return encoded

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

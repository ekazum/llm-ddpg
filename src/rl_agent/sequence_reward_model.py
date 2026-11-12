"""``SequenceRewardModel`` — a sequence-classification reward model: (prompt, response) in, one score out."""

import logging
from typing import Any, cast

import torch
from ai_runtime import BatchModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class SequenceRewardModel(BatchModel):
    """The frozen reward model as an ai-runtime ``BatchModel``: the worker thread feeds it pairs.

    The input is a user turn and an assistant turn under the model's own chat template, with no system prompt.
    """

    # Characters a token is taken to span when a pair's weight is capped at the truncation length.
    _CHARS_PER_TOKEN = 4

    def __init__(self, model_name: str, device: str, max_tokens: int) -> None:
        """Load the tokenizer and the weights (bfloat16, one output) in eval mode."""
        self._device = torch.device(device)
        self._max_tokens = max_tokens
        # The auto classes' stubs have no return type; they build these two types.
        self._tokenizer: PreTrainedTokenizerBase = cast(Any, AutoTokenizer).from_pretrained(model_name)
        self._tokenizer.padding_side = "right"
        self._model: PreTrainedModel = cast(Any, AutoModelForSequenceClassification).from_pretrained(
            model_name, dtype=torch.bfloat16, num_labels=1)
        if self._model.config.pad_token_id is None:
            # The score is read at the last token that is not padding.
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.to(self._device)
        self._model.eval()
        logger.info("SequenceRewardModel: %s on %s", model_name, self._device)

    def load(self) -> None:
        """Nothing to do: the weights are loaded at construction."""

    def unload(self) -> None:
        """Move the weights off the device and free its memory."""
        self._model.to(torch.device("cpu"))
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def weight(self, item: tuple[str, str]) -> int:
        """Characters of the pair, capped at what truncation keeps; the batch budget is in characters."""
        return min(len(item[0]) + len(item[1]), self._max_tokens * self._CHARS_PER_TOKEN)

    def process(self, items: list[tuple[str, str]], budgets: list) -> list[float]:
        """One score per ``(prompt, response)`` pair, in the order given."""
        texts = [self._format(prompt, response) for prompt, response in items]
        encoded = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self._max_tokens)
        encoded = encoded.to(self._device)
        with torch.no_grad():
            logits: torch.Tensor = self._model(**encoded).logits
        return logits[:, 0].float().cpu().tolist()

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

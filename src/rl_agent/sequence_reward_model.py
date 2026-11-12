"""``SequenceRewardModel`` — a sequence-classification reward model: (prompt, response) in, one score out."""

import logging
from typing import Any, Optional, cast

import torch
from ai_runtime import BatchModel
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from src.rl_agent.reward_model_tokenizer import RewardModelTokenizer

logger = logging.getLogger(__name__)


class SequenceRewardModel(BatchModel):
    """The frozen reward model as an ai-runtime ``BatchModel``: the worker thread feeds it pairs.

    The input comes from ``RewardModelTokenizer``: a user turn and an assistant turn, no system prompt.
    """

    # Characters a token is taken to span when a pair's weight is capped at the truncation length.
    _CHARS_PER_TOKEN = 4

    def __init__(
        self, model_name: str, device: str, max_tokens: int, tokenizer: Optional[RewardModelTokenizer] = None,
    ) -> None:
        """Load the tokenizer and the weights (bfloat16, one output) in eval mode."""
        self._device = torch.device(device)
        self._tokenizer = tokenizer or RewardModelTokenizer(model_name, max_tokens)
        # The auto class's stub has no return type; it builds a PreTrainedModel.
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
        return min(len(item[0]) + len(item[1]), self._tokenizer.max_tokens * self._CHARS_PER_TOKEN)

    def process(self, items: list[tuple[str, str]], budgets: list) -> list[float]:
        """One score per ``(prompt, response)`` pair, in the order given."""
        input_ids, attention_mask = self._tokenizer.encode(
            [prompt for prompt, _ in items], [response for _, response in items])
        with torch.no_grad():
            logits: torch.Tensor = self._model(
                input_ids=input_ids.to(self._device), attention_mask=attention_mask.to(self._device)).logits
        return logits[:, 0].float().cpu().tolist()

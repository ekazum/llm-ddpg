"""``CriticModel`` — the critic Q: the reward model's weights, trainable, read at every token position."""

import logging
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel

logger = logging.getLogger(__name__)


class CriticModel(nn.Module):
    """Q(prompt, first t response tokens) at every position t: the reward model's score head on each hidden state.

    Loaded in float32, so small updates are not rounded away; the forward pass runs under bfloat16 autocast.
    """

    def __init__(self, model_name: str, device: str) -> None:
        """Load the reward model's weights (one output) on ``device``, in train mode.

        The score head is read at given positions, not at the last non-padding token, so no pad id is needed.
        """
        super().__init__()
        # The auto class's stub has no return type; it builds a PreTrainedModel.
        self._model: PreTrainedModel = cast(Any, AutoModelForSequenceClassification).from_pretrained(
            model_name, dtype=torch.float32, num_labels=1)
        self.to(torch.device(device))
        self.train()
        logger.info("CriticModel: %s on %s", model_name, device)

    @property
    def embedding_weight(self) -> torch.Tensor:
        """``(vocab, hidden)``: the input embedding table."""
        # The input embeddings of a Qwen3 model are an nn.Embedding; the stub returns a plain nn.Module.
        embedding = cast(nn.Embedding, self._model.get_input_embeddings())
        return embedding.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` float32: the value after each token of the input."""
        hidden = self._model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self._score(hidden)

    def values_from_embeddings(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` float32: the value after each position, from input embeddings instead of token ids."""
        hidden = self._model.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        return self._score(hidden)

    def _score(self, hidden: torch.Tensor) -> torch.Tensor:
        """The score head on every hidden state."""
        # The score head is the sequence classifier's own linear layer; the model stub does not declare it.
        values: torch.Tensor = cast(Any, self._model).score(hidden)
        return values.squeeze(-1).float()

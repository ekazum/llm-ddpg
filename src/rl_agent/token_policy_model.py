"""``TokenPolicyModel`` — the chat policy: one user turn in, one generated response out."""

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TokenPolicyModel:
    """A causal language model with its tokenizer and chat template, loaded on one device.

    Frozen: bfloat16, eval mode. Trainable (the actor): float32 weights in train mode, generation under bfloat16
    autocast, and ``logits_at`` for the actor step.
    """

    def __init__(self, model_name: str, device: str, chat_template_kwargs: dict[str, Any],
                 trainable: bool = False) -> None:
        """Load the tokenizer and the weights from the Hugging Face cache."""
        self._device = torch.device(device)
        self._chat_template_kwargs = dict(chat_template_kwargs)
        self._autocast = trainable and self._device.type == "cuda"
        # The auto classes' stubs have no return type; they build these two types.
        self._tokenizer: PreTrainedTokenizerBase = cast(Any, AutoTokenizer).from_pretrained(model_name)
        self._tokenizer.padding_side = "left"
        self._model: PreTrainedModel = cast(Any, AutoModelForCausalLM).from_pretrained(
            model_name, dtype=torch.float32 if trainable else torch.bfloat16)
        self._model.to(self._device)
        self._model.train(trainable)
        logger.info("TokenPolicyModel: %s on %s, %s", model_name, self._device,
                    "trainable (float32)" if trainable else "frozen (bfloat16)")

    @property
    def module(self) -> nn.Module:
        """The model itself, for a gradient exchange between actor copies."""
        return self._model

    def parameters(self) -> Iterator[nn.Parameter]:
        """The model's weights, for the actor's optimizer."""
        return self._model.parameters()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The model's weights by name, for a snapshot."""
        return self._model.state_dict()

    def save(self, directory: Path, state_dict: dict[str, torch.Tensor]) -> None:
        """Write ``state_dict`` as a Hugging Face model directory with this model's config and tokenizer."""
        self._model.save_pretrained(directory, state_dict=state_dict)
        self._tokenizer.save_pretrained(directory)

    def load_weights(self, directory: Path) -> None:
        """Load the weights of a model directory written by ``save`` into this model, in its own dtype."""
        started = time.perf_counter()
        # The auto class's stub has no return type; it builds a PreTrainedModel.
        loaded: PreTrainedModel = cast(Any, AutoModelForCausalLM).from_pretrained(
            directory, dtype=self._model.dtype)
        self._model.load_state_dict(loaded.state_dict())
        logger.info("TokenPolicyModel: weights loaded from %s in %.0f s", directory, time.perf_counter() - started)

    def logits_at(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, positions: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, K, vocab)``: the next-token logits at the input indices ``positions`` ``(B, K)``, with gradients."""
        hidden = self._model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        picked = hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logits: torch.Tensor = self._model.get_output_embeddings()(picked)
        return logits

    @property
    def pad_token_id(self) -> int:
        """The id that pads a response after its end."""
        return int(self._tokenizer.pad_token_id)

    def generate(self, prompts: list[str], max_new_tokens: int, temperature: float) -> list[tuple[str, int]]:
        """One ``(response text, response tokens)`` per prompt; greedy when ``temperature`` is 0."""
        new_tokens = self._new_tokens(prompts, max_new_tokens, temperature)
        counts: list[int] = (new_tokens != self.pad_token_id).sum(dim=1).tolist()
        responses: list[str] = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [(text.strip(), int(count)) for text, count in zip(responses, counts)]

    def sample(self, prompts: list[str], max_new_tokens: int, temperature: float) -> list[tuple[list[int], str]]:
        """One ``(response token ids, response text)`` per prompt; the ids stop at the first padding id."""
        new_tokens = self._new_tokens(prompts, max_new_tokens, temperature).cpu()
        counts: list[int] = (new_tokens != self.pad_token_id).sum(dim=1).tolist()
        ids: list[list[int]] = [row[:count].tolist() for row, count in zip(new_tokens, counts)]
        return list(zip(ids, self.decode(ids)))

    def decode(self, ids: list[list[int]]) -> list[str]:
        """The text of each response, special tokens dropped; safe to call from any thread."""
        texts: list[str] = self._tokenizer.batch_decode(ids, skip_special_tokens=True)
        return [text.strip() for text in texts]

    def _new_tokens(self, prompts: list[str], max_new_tokens: int, temperature: float) -> torch.Tensor:
        """The generated tokens of each prompt under the chat template, padded on the right."""
        texts: list[str] = [
            str(self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            ))
            for prompt in prompts
        ]
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = encoded.to(self._device)
        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        )
        with torch.no_grad(), torch.autocast(device_type=self._device.type, dtype=torch.bfloat16,
                                             enabled=self._autocast):
            # generate comes from GenerationMixin, which the PreTrainedModel stub does not carry.
            output: torch.Tensor = cast(Any, self._model).generate(
                **encoded, max_new_tokens=max_new_tokens, pad_token_id=self.pad_token_id, **sampling)
        return output[:, encoded["input_ids"].shape[1]:]

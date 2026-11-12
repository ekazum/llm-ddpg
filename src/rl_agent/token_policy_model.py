"""``TokenPolicyModel`` — the chat policy: one user turn in, one generated response out."""

import logging
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TokenPolicyModel:
    """A causal language model with its tokenizer and chat template, loaded on one device in eval mode."""

    def __init__(self, model_name: str, device: str, chat_template_kwargs: dict[str, Any]) -> None:
        """Load the tokenizer and the weights (bfloat16) from the Hugging Face cache."""
        self._device = torch.device(device)
        self._chat_template_kwargs = dict(chat_template_kwargs)
        # The auto classes' stubs have no return type; they build these two types.
        self._tokenizer: PreTrainedTokenizerBase = cast(Any, AutoTokenizer).from_pretrained(model_name)
        self._tokenizer.padding_side = "left"
        self._model: PreTrainedModel = cast(Any, AutoModelForCausalLM).from_pretrained(
            model_name, dtype=torch.bfloat16)
        self._model.to(self._device)
        self._model.eval()
        logger.info("TokenPolicyModel: %s on %s", model_name, self._device)

    def generate(self, prompts: list[str], max_new_tokens: int, temperature: float) -> list[tuple[str, int]]:
        """One ``(response text, response tokens)`` per prompt; greedy when ``temperature`` is 0."""
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
        pad_id = self._tokenizer.pad_token_id
        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        )
        with torch.no_grad():
            # generate comes from GenerationMixin, which the PreTrainedModel stub does not carry.
            output: torch.Tensor = cast(Any, self._model).generate(
                **encoded, max_new_tokens=max_new_tokens, pad_token_id=pad_id, **sampling)
        new_tokens = output[:, encoded["input_ids"].shape[1]:]
        counts: list[int] = (new_tokens != pad_id).sum(dim=1).tolist()
        responses: list[str] = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [(text.strip(), int(count)) for text, count in zip(responses, counts)]

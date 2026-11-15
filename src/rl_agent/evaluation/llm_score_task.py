"""``LLMScoreTask`` — scores the current policy on one split: generate, reward, record."""

import logging
import os
import time
from pathlib import Path

import pandas as pd
from rl_core.eval_store import EvalStore
from rl_core.score_task import ScoreTask

from pipeline.prompt_schema import PromptSchema
from src.rl_agent.evaluation.prompt_reader import PromptReader
from src.rl_agent.evaluation.response_schema import ResponseSchema
from src.rl_agent.reward_model_client import RewardModelClient
from src.rl_agent.token_policy_model import TokenPolicyModel

logger = logging.getLogger(__name__)


class LLMScoreTask(ScoreTask):
    """One response per prompt from the policy, each scored by the frozen reward model.

    Writes ``responses_<split>.parquet`` (under ``update_<update>/`` for a training snapshot) and the metrics to
    ``eval.db``: a dev row keyed by the update and checkpoint scored, or the test metrics.
    """

    def __init__(
        self,
        policy: TokenPolicyModel,
        reward_client: RewardModelClient,
        store: EvalStore,
        prompts: PromptReader,
        output_dir: Path,
        *,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        """Bind the policy, the reward client, the store, the prompts and the generation settings."""
        self._policy = policy
        self._reward_client = reward_client
        self._store = store
        self._prompts = prompts
        self._output_dir = output_dir
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def run(self, split: str, update: int = 0, checkpoint: Path | None = None) -> dict:
        """Score the policy on ``split`` (dev or test), write its responses file, record and return its metrics.

        A dev row is keyed by ``update`` and names ``checkpoint``, the weights the policy holds.
        """
        if split not in ("dev", "test"):
            raise ValueError(f"LLMScoreTask: split must be dev or test; got {split!r}")
        if split == "dev" and checkpoint is None:
            raise ValueError("LLMScoreTask: a dev pass needs the checkpoint it scores")
        started = time.monotonic()
        prompts = self._prompts.read(split)
        batch_size = self._batch_size
        rows: list[tuple[str, str, float, int]] = []
        for lo in range(0, len(prompts), batch_size):
            chunk = prompts.iloc[lo:lo + batch_size]
            texts: list[str] = chunk[PromptSchema.PROMPT].tolist()
            generated = self._policy.generate(texts, self._max_new_tokens, self._temperature)
            responses = [text for text, _ in generated]
            rewards = self._reward_client.predict(texts, responses)
            keys: list[str] = chunk[PromptSchema.KEY].tolist()
            rows.extend(
                (key, response, reward, tokens)
                for key, (response, tokens), reward in zip(keys, generated, rewards)
            )
            logger.info("LLMScoreTask: %s %d/%d prompts", split, len(rows), len(prompts))
        frame = pd.DataFrame(rows, columns=list(ResponseSchema.COLUMNS[:4]))
        frame[ResponseSchema.SPLIT] = split
        self._write(frame, split, update)
        metrics = {
            "mean_reward": float(frame[ResponseSchema.REWARD].mean()),
            "n_prompts": int(len(frame)),
            "mean_response_tokens": float(frame[ResponseSchema.RESPONSE_TOKENS].mean()),
            "eval_seconds": time.monotonic() - started,
        }
        if split == "dev" and checkpoint is not None:
            self._store.add_dev(update, checkpoint, metrics)
        else:
            self._store.add_test(metrics)
        logger.info("LLMScoreTask: %s mean reward %.4f over %d prompts, %.1f tokens per response, %.0f s",
                    split, metrics["mean_reward"], metrics["n_prompts"], metrics["mean_response_tokens"],
                    metrics["eval_seconds"])
        return metrics

    def _write(self, frame: pd.DataFrame, split: str, update: int) -> None:
        """Write the responses file through a temporary file and a rename; a snapshot's under ``update_<update>/``."""
        directory = self._output_dir / f"update_{update:07d}" if update > 0 else self._output_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ResponseSchema.file_name(split)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)

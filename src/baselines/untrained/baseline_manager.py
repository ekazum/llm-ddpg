"""``BaselineManager`` — the baseline job: reward check, then the untrained policy on dev and on test.

Builds every part, starts the reward model thread, runs the steps in order on the calling thread and stops
the thread. One run directory under ``records_dir`` holds ``eval.db``, the two responses files and ``meta.json``.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from ai_runtime import ThreadStages
from rl_core.eval_store import EvalStore

from pipeline.prompt_schema import PromptSchema
from src.rl_agent.evaluation.llm_eval_config import LLMEvalConfig
from src.rl_agent.evaluation.llm_score_task import LLMScoreTask
from src.rl_agent.evaluation.prompt_reader import PromptReader
from src.rl_agent.evaluation.run_record import RunRecord
from src.rl_agent.reward_model_client import RewardModelClient
from src.rl_agent.reward_model_worker import RewardModelWorker
from src.rl_agent.sequence_reward_model import SequenceRewardModel
from src.rl_agent.token_policy_model import TokenPolicyModel

logger = logging.getLogger(__name__)


class BaselineManager:
    """Owns the baseline job's parts, its one thread stage and the order of its steps."""

    KIND = "baseline"
    # The dev column the best checkpoint is chosen by.
    CRITERION = "mean_reward"
    _RM_STAGE = "rm-worker"
    # Seconds between checks of the shutdown event while the reward model loads.
    _READY_POLL_S = 5.0

    def __init__(
        self,
        config: LLMEvalConfig,
        policy: Optional[TokenPolicyModel] = None,
        reward_model: Optional[SequenceRewardModel] = None,
    ) -> None:
        """Check the inputs, open the run record, load the reward model and wire its thread."""
        self._config = config
        self._prompts = PromptReader(config.prompts_dir, config.limit)
        self._prompts.check()
        self._policy = policy
        self._shutdown_event = threading.Event()
        request_queue: queue.Queue = queue.Queue()
        self._worker = RewardModelWorker(
            reward_model or SequenceRewardModel(config.reward_model, config.reward_device, config.reward_max_tokens),
            request_queue,
            self._shutdown_event,
            config.reward_window_size,
            config.reward_idle_timeout_s,
            config.reward_batch_chars,
            name=self._RM_STAGE,
        )
        self._client = RewardModelClient(request_queue, self._shutdown_event)
        self._stages = ThreadStages("BaselineManager", self._shutdown_event)
        self._stages.stage(self._RM_STAGE, self._worker.run)
        # Opened last, so a run directory exists only once the reward model has loaded.
        self._record = RunRecord(config.records_dir, self.KIND, config.policy_model)
        self._store = EvalStore(self._record.directory, self.CRITERION)

    def run(self) -> None:
        """Start the reward model thread, run the checks and the two score passes, stop the thread."""
        logger.info("BaselineManager: run %s in %s", self._record.run_id, self._record.directory)
        status = "failed"
        self._stages.start()
        try:
            self._wait_ready()
            self._check_batching()
            self._reward_check()
            policy = self._policy or TokenPolicyModel(
                self._config.policy_model, self._config.policy_device, self._config.chat_template_kwargs)
            score = LLMScoreTask(
                policy, self._client, self._store, self._prompts, self._record.directory,
                batch_size=self._config.batch_size, max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature)
            score.run("dev", checkpoint=Path(self._config.policy_model))
            score.run("test")
            status = "done"
        finally:
            self._finish(status)
        logger.info("BaselineManager: run %s done", self._record.run_id)

    def _finish(self, status: str) -> None:
        """Stop the reward model thread, then record how the run ended."""
        try:
            self._stages.stop([self._RM_STAGE])
        finally:
            self._record.close(status)

    def _wait_ready(self) -> None:
        """Block until the reward model is loaded; raise if its thread stopped first."""
        while not self._worker.ready.wait(self._READY_POLL_S):
            if self._shutdown_event.is_set():
                raise RuntimeError("BaselineManager: the reward model thread stopped before it was ready")

    def _check_batching(self) -> None:
        """Score a few dev pairs one by one and as one batch; raise if padding changes the scores."""
        frame = self._prompts.read("dev").head(self._config.reward_batch_check_pairs)
        prompts: list[str] = frame[PromptSchema.PROMPT].tolist()
        replies: list[str] = frame[PromptSchema.CHOSEN].tolist()
        single = [self._client.predict([prompt], [reply])[0] for prompt, reply in zip(prompts, replies)]
        batched = self._client.predict(prompts, replies)
        difference = float(np.max(np.abs(np.array(single) - np.array(batched)))) if single else 0.0
        self._store.set_pairs("reward_check", {"batch_pairs": len(single), "batch_max_difference": difference})
        logger.info("BaselineManager: batched and single reward scores differ by at most %.4f over %d pairs",
                    difference, len(single))
        if difference > self._config.reward_batch_tolerance:
            raise RuntimeError(
                f"BaselineManager: batched reward scores differ from single ones by {difference:.4f}, "
                f"more than reward_batch_tolerance {self._config.reward_batch_tolerance}")

    def _reward_check(self) -> None:
        """Score the dev prompts' chosen and rejected replies; record how often chosen scores higher."""
        frame = self._prompts.read("dev")
        prompts: list[str] = frame[PromptSchema.PROMPT].tolist()
        chosen = np.array(self._client.predict(prompts, frame[PromptSchema.CHOSEN].tolist()))
        rejected = np.array(self._client.predict(prompts, frame[PromptSchema.REJECTED].tolist()))
        result = {
            "pairs": int(len(frame)),
            "chosen_win_rate": float(np.mean(chosen > rejected)),
            "mean_chosen": float(chosen.mean()),
            "mean_rejected": float(rejected.mean()),
        }
        self._store.set_pairs("reward_check", result)
        logger.info("BaselineManager: reward check over %d dev pairs: chosen wins %.3f, mean chosen %.3f, "
                    "mean rejected %.3f", result["pairs"], result["chosen_win_rate"], result["mean_chosen"],
                    result["mean_rejected"])

"""``ActorProcessor`` — owns one actor batch from prompt rows to replay rows.

Per batch: the prompt text of the rows, responses from ``ActorGPUWorker``, their rewards from
``RewardModelClient``, one ``TrajectoryBatch`` on the write queue and one ``actor_batches`` stats row.
"""

import logging
import queue
import threading
import time

import numpy as np
from ai_runtime import SHUTDOWN_SENTINEL, BatchRequest

from src.rl_agent.agent.prompt_sampler import PromptSampler
from src.rl_agent.agent.trajectory_batch import TrajectoryBatch
from src.rl_agent.reward_model_client import RewardModelClient
from src.rl_agent.telemetry.telemetry import Telemetry

logger = logging.getLogger(__name__)


class ActorProcessor:
    """Processor thread: batch queue in, write queue out; generation and reward go to their own threads."""

    # Queue poll interval; bounds shutdown latency while idle.
    _QUEUE_POLL_TIMEOUT: float = 0.1
    # Seconds between checks of the shutdown event while waiting for responses or a free write slot.
    _WAIT_POLL_S: float = 1.0

    def __init__(
        self,
        sampler: PromptSampler,
        batch_queue: queue.Queue,
        sample_queue: queue.Queue,
        reward_client: RewardModelClient,
        write_queue: queue.Queue,
        telemetry: Telemetry,
        max_new_tokens: int,
        pad_token_id: int,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the prompt table, the three queues, the reward client and the counters."""
        self._sampler = sampler
        self._batch_queue = batch_queue
        self._sample_queue = sample_queue
        self._reward_client = reward_client
        self._write_queue = write_queue
        self._telemetry = telemetry
        self._max_new_tokens = max_new_tokens
        self._pad_token_id = pad_token_id
        self._shutdown_event = shutdown_event

    def run(self) -> None:
        """Process batches until shutdown; a failure sets the shutdown event."""
        logger.info("ActorProcessor: started")
        try:
            while not self._shutdown_event.is_set():
                try:
                    rows = self._batch_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                if rows is SHUTDOWN_SENTINEL:
                    break
                self._run_batch(rows)
        except Exception:
            self._shutdown_event.set()
            raise
        logger.info("ActorProcessor: exiting, %d batches written", self._telemetry.actor_batches)

    def _run_batch(self, rows: list[int]) -> None:
        """Generate, reward, write; a batch cut short by shutdown is dropped."""
        prompts = self._sampler.prompts(rows)
        started = time.perf_counter()
        samples = self._sample(prompts)
        if samples is None:
            return
        generated = time.perf_counter()
        rewards = self._reward_client.predict(prompts, [text for _, text in samples])
        rewarded = time.perf_counter()
        batch = self._batch(rows, [ids for ids, _ in samples], rewards)
        if not self._put(batch):
            return
        index = self._telemetry.add_actor_batch()
        self._telemetry.log_actor_batch({
            "actor_batch": index,
            "rows": batch.num_rows,
            "mean_reward": float(batch.reward.mean()),
            "mean_response_tokens": float(batch.response_len.mean()),
            "truncated_frac": float((batch.response_len >= self._max_new_tokens).mean()),
            "generate_seconds": generated - started,
            "reward_seconds": rewarded - generated,
            "update_count": self._telemetry.update_count,
        })
        logger.info("ActorProcessor: batch %d, %d rows, mean reward %.3f, %.0f tokens per response, "
                    "generate %.1f s, reward %.1f s", index, batch.num_rows, float(batch.reward.mean()),
                    float(batch.response_len.mean()), generated - started, rewarded - generated)

    def _sample(self, prompts: list[str]) -> list[tuple[list[int], str]] | None:
        """One ``(response ids, response text)`` per prompt from the actor GPU worker; None on shutdown."""
        count = len(prompts)
        sink: queue.Queue = queue.Queue()
        self._sample_queue.put(BatchRequest(keys=list(range(count)), items=prompts, sink=sink, budgets=[0] * count))
        samples: list[tuple[list[int], str]] = [([], "")] * count
        received = 0
        while received < count:
            try:
                key, sample = sink.get(timeout=self._WAIT_POLL_S)
            except queue.Empty:
                if self._shutdown_event.is_set():
                    return None
                continue
            samples[key] = sample
            received += 1
        return samples

    def _batch(self, rows: list[int], ids: list[list[int]], rewards: list[float]) -> TrajectoryBatch:
        """The batch's transitions; responses padded to ``max_new_tokens``."""
        count = len(rows)
        response_ids = np.full((count, self._max_new_tokens), self._pad_token_id, dtype=np.int32)
        lengths = np.zeros(count, dtype=np.int16)
        for i, row_ids in enumerate(ids):
            response_ids[i, :len(row_ids)] = row_ids
            lengths[i] = len(row_ids)
        return TrajectoryBatch(
            prompt_row=np.asarray(rows, dtype=np.int32),
            response_ids=response_ids,
            response_len=lengths,
            reward=np.asarray(rewards, dtype=np.float32),
            done=np.ones(count, dtype=np.bool_),
        )

    def _put(self, batch: TrajectoryBatch) -> bool:
        """Bounded put on the write queue that observes shutdown; False on shutdown."""
        while not self._shutdown_event.is_set():
            try:
                self._write_queue.put(batch, timeout=self._WAIT_POLL_S)
                return True
            except queue.Full:
                continue
        return False

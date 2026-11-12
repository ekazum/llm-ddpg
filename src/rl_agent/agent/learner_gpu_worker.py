"""``LearnerGPUWorker`` — the learner thread. Phase 4 skeleton: batches reach the GPU, no gradient step yet.

Per batch: the tensors are copied to the learner device, the slot goes back to the pool once the copies have
finished, the update count advances, and every ``stats_every_updates`` updates a ``CheckpointEvent`` asks
``StatsProcessor`` for a buffer stats row. Phase 5 adds the critic step.
"""

import logging
import queue
import threading

import torch
from ai_runtime import SHUTDOWN_SENTINEL
from rl_core.events.checkpoint_event import CheckpointEvent

from src.rl_agent.agent.learner_batch import LearnerBatch
from src.rl_agent.telemetry.telemetry import Telemetry

logger = logging.getLogger(__name__)


class LearnerGPUWorker:
    """Learner thread: tensor queue in; the device copies, the update count and the stats requests out."""

    # Queue poll and bounded-put retry interval; bounds shutdown latency.
    _QUEUE_POLL_TIMEOUT: float = 0.1
    _QUEUE_PUT_TIMEOUT: float = 0.5

    def __init__(
        self,
        tensor_queue: queue.Queue,
        slot_pool: queue.Queue,
        stats_queue: queue.Queue,
        telemetry: Telemetry,
        device: str,
        stats_every_updates: int,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the two queues, the slot pool, the counters and the device."""
        self._tensor_queue = tensor_queue
        self._slot_pool = slot_pool
        self._stats_queue = stats_queue
        self._telemetry = telemetry
        self._device = torch.device(device)
        self._stats_every_updates = stats_every_updates
        self._shutdown_event = shutdown_event

    def run(self) -> None:
        """Take batches until shutdown; a failure sets the shutdown event."""
        logger.info("LearnerGPUWorker: started on %s", self._device)
        updates = self._telemetry.update_count
        try:
            while not self._shutdown_event.is_set():
                try:
                    batch = self._tensor_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                if batch is SHUTDOWN_SENTINEL:
                    break
                self._to_device(batch)
                self._slot_pool.put(batch.slot)
                updates += 1
                self._telemetry.note_update(updates)
                if updates % self._stats_every_updates == 0:
                    self._request_stats(updates)
        except Exception:
            self._shutdown_event.set()
            raise
        logger.info("LearnerGPUWorker: exiting at update %d", updates)

    def _to_device(self, batch: LearnerBatch) -> dict[str, torch.Tensor]:
        """The batch's tensors on the device; returns after the copies have finished, so the slot can be reused."""
        tensors = {
            "reward": batch.reward,
            "done": batch.done,
            "is_weights": batch.is_weights,
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
        }
        on_device = {name: tensor.to(self._device, non_blocking=True) for name, tensor in tensors.items()}
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()
        return on_device

    def _request_stats(self, updates: int) -> None:
        """Ask for a buffer stats row at this update; waits for room rather than dropping the request."""
        while not self._shutdown_event.is_set():
            try:
                self._stats_queue.put(CheckpointEvent(updates), timeout=self._QUEUE_PUT_TIMEOUT)
                return
            except queue.Full:
                continue

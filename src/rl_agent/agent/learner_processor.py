"""``LearnerProcessor`` — a drawn batch from the replay buffer to a ``LearnerBatch`` in a pinned slot.

Per draw: the rows are gathered into a free slot (one copy, under the buffer's shared lock), each response's own
token ids are placed in the reward model's template after its prompt, and the batch goes on the tensor queue. Several processors can drain the same prefetch queue; batch order does not matter.
"""

import logging
import queue
import threading

import numpy as np
from ai_runtime import SHUTDOWN_SENTINEL
from rl_core.prioritized_replay_buffer import PrioritizedReplayBuffer

from src.rl_agent.agent.learner_batch import LearnerBatch
from src.rl_agent.agent.pinned_slot import PinnedSlot
from src.rl_agent.agent.prompt_sampler import PromptSampler
from src.rl_agent.reward_model_tokenizer import RewardModelTokenizer

logger = logging.getLogger(__name__)


class LearnerProcessor:
    """Processor thread: prefetch queue in, tensor queue out."""

    # Queue poll and bounded-put retry interval; bounds shutdown latency.
    _QUEUE_POLL_TIMEOUT: float = 0.1
    _QUEUE_PUT_TIMEOUT: float = 0.5

    def __init__(
        self,
        replay_buffer: PrioritizedReplayBuffer,
        sampler: PromptSampler,
        critic_tokenizer: RewardModelTokenizer,
        prefetch_queue: queue.Queue,
        tensor_queue: queue.Queue,
        slot_pool: queue.Queue,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the buffer, the prompt table, the critic's tokenizer, the two queues and the slot pool."""
        self._buffer = replay_buffer
        self._sampler = sampler
        self._critic_tokenizer = critic_tokenizer
        self._prefetch_queue = prefetch_queue
        self._tensor_queue = tensor_queue
        self._slot_pool = slot_pool
        self._shutdown_event = shutdown_event

    def run(self) -> None:
        """Convert draws until shutdown; a failure sets the shutdown event."""
        logger.info("LearnerProcessor: started")
        converted = 0
        try:
            while not self._shutdown_event.is_set():
                try:
                    item = self._prefetch_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                if item is SHUTDOWN_SENTINEL:
                    break
                slot = self._take_slot()
                if slot is None:
                    break
                indices, is_weights = item
                if not self._put(self._convert(indices, is_weights, slot)):
                    break
                converted += 1
        except Exception:
            self._shutdown_event.set()
            raise
        logger.info("LearnerProcessor: exiting, %d batches converted", converted)

    def _convert(self, indices: np.ndarray, is_weights: np.ndarray, slot: PinnedSlot) -> LearnerBatch:
        """Gather the rows into ``slot`` and add the critic input of each, built from the stored token ids."""
        self._buffer.gather_into(indices, slot.replay_views())
        np.copyto(slot.is_weights.numpy(), is_weights)
        lengths: list[int] = slot.response_len.tolist()
        ids: list[list[int]] = [row[:length].tolist() for row, length in zip(slot.response_ids, lengths)]
        prompts = self._sampler.prompts(slot.prompt_row.tolist())
        input_ids, attention_mask, prompt_end, value_len = self._critic_tokenizer.encode_ids(prompts, ids)
        width = int(input_ids.shape[1])
        slot.input_ids[:, :width].copy_(input_ids)
        slot.attention_mask[:, :width].copy_(attention_mask)
        slot.prompt_end.copy_(prompt_end)
        slot.value_len.copy_(value_len)
        return LearnerBatch(
            reward=slot.reward,
            done=slot.done,
            is_weights=slot.is_weights,
            input_ids=slot.input_ids[:, :width],
            attention_mask=slot.attention_mask[:, :width],
            prompt_end=slot.prompt_end,
            value_len=slot.value_len,
            indices=indices,
            slot=slot,
        )

    def _take_slot(self) -> PinnedSlot | None:
        """A free slot from the pool; None on shutdown."""
        while not self._shutdown_event.is_set():
            try:
                return self._slot_pool.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                continue
        return None

    def _put(self, batch: LearnerBatch) -> bool:
        """Bounded put on the tensor queue that observes shutdown; False on shutdown."""
        while not self._shutdown_event.is_set():
            try:
                self._tensor_queue.put(batch, timeout=self._QUEUE_PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False

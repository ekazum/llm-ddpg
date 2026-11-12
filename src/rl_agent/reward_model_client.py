"""``RewardModelClient`` — the caller side of the reward model queue."""

import queue
import threading

from ai_runtime import BatchRequest


class RewardModelClient:
    """Submits (prompt, response) pairs to the ``RewardModelWorker`` and waits for their scores.

    ``predict`` is thread-safe: each call has its own reply queue.
    """

    # Seconds between checks of the shutdown event while waiting for scores.
    _POLL_S = 1.0

    def __init__(self, request_queue: queue.Queue, shutdown_event: threading.Event) -> None:
        self._request_queue = request_queue
        self._shutdown_event = shutdown_event

    def predict(self, prompts: list[str], responses: list[str]) -> list[float]:
        """The reward of each ``(prompt, response)`` pair, in the order given."""
        if len(prompts) != len(responses):
            raise ValueError(f"RewardModelClient: {len(prompts)} prompts but {len(responses)} responses")
        count = len(prompts)
        sink: queue.Queue = queue.Queue()
        self._request_queue.put(BatchRequest(
            keys=list(range(count)), items=list(zip(prompts, responses)), sink=sink, budgets=[0] * count))
        rewards = [0.0] * count
        received = 0
        while received < count:
            try:
                key, reward = sink.get(timeout=self._POLL_S)
            except queue.Empty:
                if self._shutdown_event.is_set():
                    raise RuntimeError("RewardModelClient: reward model serving stopped; scores not received")
                continue
            rewards[key] = float(reward)
            received += 1
        return rewards

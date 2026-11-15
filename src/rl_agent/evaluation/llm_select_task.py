"""``LLMSelectTask`` — scores each actor snapshot on dev on its own thread, and keeps the best and the latest.

``serve`` is the thread: ``(update, actor directory)`` items from the snapshot queue, in the order written. Each is
loaded into the eval policy and scored on the fixed dev sample; its row goes to ``eval.db``. Unless every snapshot
is kept, the earlier snapshots that are not the best dev row are deleted. The thread drains the queue to the
sentinel, so the last snapshot is scored before the test pass; ``abort_event`` stops it early.
"""

import logging
import queue
import threading
from pathlib import Path

from ai_runtime import SHUTDOWN_SENTINEL
from rl_core.eval_store import EvalStore
from rl_core.select_task import SelectTask

from src.rl_agent.agent.run_snapshot_store import RunSnapshotStore
from src.rl_agent.evaluation.llm_score_task import LLMScoreTask
from src.rl_agent.token_policy_model import TokenPolicyModel

logger = logging.getLogger(__name__)


class LLMSelectTask(SelectTask):
    """Dev scoring of each snapshot through ``LLMScoreTask``; retention of the best and the latest snapshot."""

    # Snapshot-queue poll interval; bounds how fast an abort is seen.
    _QUEUE_POLL_TIMEOUT: float = 0.5

    def __init__(
        self,
        policy: TokenPolicyModel,
        score: LLMScoreTask,
        store: EvalStore,
        snapshots: RunSnapshotStore,
        snapshot_queue: queue.Queue,
        abort_event: threading.Event,
        shutdown_event: threading.Event,
        keep_all: bool,
    ) -> None:
        """Bind the eval policy, the dev score task, the stores, the queue and the events.

        A failure sets ``shutdown_event`` so the training stops too; ``keep_all`` turns retention off.
        """
        self._policy = policy
        self._score = score
        self._store = store
        self._snapshots = snapshots
        self._snapshot_queue = snapshot_queue
        self._abort_event = abort_event
        self._shutdown_event = shutdown_event
        self._keep_all = keep_all
        self._kept: list[int] = []

    def run(self, update: int, checkpoint: Path) -> dict:
        """Load the actor snapshot taken at ``update`` and score it on dev; its dev row is written."""
        self._policy.load_weights(checkpoint)
        return self._score.run("dev", update, checkpoint)

    def serve(self) -> None:
        """Thread main loop: score snapshots until the sentinel or an abort; a failure stops the training."""
        logger.info("LLMSelectTask: started")
        scored = 0
        try:
            while not self._abort_event.is_set():
                try:
                    item = self._snapshot_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                if item is SHUTDOWN_SENTINEL:
                    break
                update, checkpoint = item
                self.run(update, checkpoint)
                scored += 1
                self._retain(update)
        except Exception:
            self._shutdown_event.set()
            raise
        logger.info("LLMSelectTask: exiting, %d snapshots scored, best %s", scored, self._store.champion())

    def _retain(self, update: int) -> None:
        """Keep the best dev snapshot and the one just scored; delete the other scored ones."""
        if self._keep_all:
            return
        champion = self._store.champion()
        best = champion[0] if champion is not None else update
        for kept in self._kept:
            if kept not in (best, update):
                self._snapshots.delete_update(kept)
        self._kept = sorted({best, update})

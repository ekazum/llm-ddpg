"""``RewardModelWorker`` — the reward model thread: loads the frozen model and scores queued pairs."""

from ai_runtime import GpuBatchStage


class RewardModelWorker(GpuBatchStage):
    """An ai-runtime ``GpuBatchStage`` over a ``SequenceRewardModel``: length-sorted batches under a budget.

    Callers reach it through ``RewardModelClient``; ``ready`` is set once the model is loaded.
    """

    def run(self) -> None:
        """Serve until shutdown; a failure sets the shutdown event so a waiting client stops waiting."""
        try:
            super().run()
        except Exception:
            self._shutdown_event.set()
            raise

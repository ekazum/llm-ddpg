"""``ActorGPUWorker`` — the actor's generation thread: no-grad sampling for queued prompts."""

from ai_runtime import GpuBatchStage


class ActorGPUWorker(GpuBatchStage):
    """An ai-runtime ``GpuBatchStage`` over a ``SamplingModel``: queued prompts are generated for in batches.

    ``ActorProcessor`` puts one ``BatchRequest`` per actor batch on the worker's queue.
    """

    def run(self) -> None:
        """Serve until shutdown; a failure sets the shutdown event so a waiting processor stops waiting."""
        try:
            super().run()
        except Exception:
            self._shutdown_event.set()
            raise

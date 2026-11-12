"""``RLManager`` — the training job: builds every part and queue, runs the threads, stops them in order.

Actor side: ``ActorFeeder`` -> ``ActorProcessor`` (responses from ``ActorGPUWorker``, rewards from
``RewardModelWorker``) -> ``ReplayWriter`` -> ``PrioritizedReplayBuffer``. Learner side: ``ReplayReader`` ->
``LearnerProcessor`` -> ``LearnerGPUWorker`` (Phase 4: no gradient step yet); ``StatsProcessor`` writes the buffer
stats rows. The job stops once ``max_actor_transitions`` are stored. The reward model thread has its own stage,
stopped last.
"""

import logging
import queue
import threading
from typing import Optional

import numpy as np
import torch
from ai_runtime import ThreadStages
from rl_core.actor_feeder import ActorFeeder
from rl_core.prioritized_replay_buffer import PrioritizedReplayBuffer
from rl_core.replay_reader import ReplayReader
from rl_core.replay_writer import ReplayWriter
from rl_core.stats_processor import StatsProcessor

from src.rl_agent.agent.actor_gpu_worker import ActorGPUWorker
from src.rl_agent.agent.actor_processor import ActorProcessor
from src.rl_agent.agent.learner_gpu_worker import LearnerGPUWorker
from src.rl_agent.agent.learner_processor import LearnerProcessor
from src.rl_agent.agent.pinned_slot import PinnedSlot
from src.rl_agent.agent.prompt_sampler import PromptSampler
from src.rl_agent.agent.sampling_model import SamplingModel
from src.rl_agent.agent.trajectory_codec import TrajectoryCodec
from src.rl_agent.evaluation.prompt_reader import PromptReader
from src.rl_agent.evaluation.run_record import RunRecord
from src.rl_agent.reward_model_client import RewardModelClient
from src.rl_agent.reward_model_tokenizer import RewardModelTokenizer
from src.rl_agent.reward_model_worker import RewardModelWorker
from src.rl_agent.sequence_reward_model import SequenceRewardModel
from src.rl_agent.telemetry.stats_store import StatsStore
from src.rl_agent.telemetry.telemetry import Telemetry
from src.rl_agent.token_policy_model import TokenPolicyModel
from src.rl_agent.train_config import TrainConfig

logger = logging.getLogger(__name__)


class RLManager:
    """Owns the training job's parts, its queues, its two thread stage groups and their order."""

    KIND = "train"
    _RM_STAGE = "rm-worker"
    _FEEDER_STAGE = "actor-feeder"
    _PROCESSOR_STAGE = "actor-processor"
    _ACTOR_GPU_STAGE = "actor-gpu-worker"
    _WRITER_STAGE = "replay-writer"
    _READER_STAGE = "replay-reader"
    _LEARNER_PROCESSOR_STAGE = "learner-processor"
    _LEARNER_STAGE = "learner"
    _STATS_STAGE = "stats-processor"
    # Producers are joined before their consumers.
    _JOIN_ORDER = [_FEEDER_STAGE, _PROCESSOR_STAGE, _ACTOR_GPU_STAGE, _READER_STAGE, _LEARNER_PROCESSOR_STAGE,
                   _LEARNER_STAGE, _WRITER_STAGE, _STATS_STAGE]
    # Actor batches waiting for a processor, pushed batches waiting for the writer, and stats events.
    _BATCH_QUEUE_SIZE = 4
    _WRITE_QUEUE_SIZE = 8
    _STATS_QUEUE_SIZE = 2048
    # Seconds between checks of the stop condition and of the reward model's readiness.
    _POLL_S = 5.0

    def __init__(
        self,
        config: TrainConfig,
        policy: Optional[TokenPolicyModel] = None,
        reward_model: Optional[SequenceRewardModel] = None,
        critic_tokenizer: Optional[RewardModelTokenizer] = None,
    ) -> None:
        """Check the inputs, load both models, open the run record, build the parts and wire the stages."""
        self._config = config
        PromptReader(config.prompts_dir, 0).check()
        sampler_seed, replay_seed = np.random.SeedSequence(config.seed).spawn(2)
        self._sampler = PromptSampler(config.prompts_dir, sampler_seed)

        self._rm_shutdown_event = threading.Event()
        reward_queue: queue.Queue = queue.Queue()
        self._rm_worker = RewardModelWorker(
            reward_model or SequenceRewardModel(config.reward_model, config.reward_device, config.reward_max_tokens),
            reward_queue,
            self._rm_shutdown_event,
            config.reward_window_size,
            config.reward_idle_timeout_s,
            config.reward_batch_chars,
            name=self._RM_STAGE,
        )
        policy = policy or TokenPolicyModel(config.policy_model, config.policy_device, config.chat_template_kwargs)
        # Its own instance: a tokenizer must not be called from two threads at once.
        critic_tokenizer = critic_tokenizer or RewardModelTokenizer(config.reward_model, config.reward_max_tokens)

        self._shutdown_event = threading.Event()
        batch_queue: queue.Queue = queue.Queue(maxsize=self._BATCH_QUEUE_SIZE)
        sample_queue: queue.Queue = queue.Queue()
        write_queue: queue.Queue = queue.Queue(maxsize=self._WRITE_QUEUE_SIZE)
        stats_queue: queue.Queue = queue.Queue(maxsize=self._STATS_QUEUE_SIZE)
        prefetch_queue: queue.Queue = queue.Queue(maxsize=config.prefetch_depth)
        tensor_queue: queue.Queue = queue.Queue(maxsize=config.prefetch_depth)
        slot_pool = self._slot_pool(config)
        actor_worker = ActorGPUWorker(
            SamplingModel(policy, config.max_new_tokens, config.temperature),
            sample_queue,
            self._shutdown_event,
            config.generate_batch_size,
            config.generate_idle_timeout_s,
            config.generate_batch_size,
            name=self._ACTOR_GPU_STAGE,
        )

        # Opened once both models have loaded, so a run directory exists only for a job that got this far.
        self._record = RunRecord(config.records_dir, self.KIND, config.policy_model)
        self._telemetry = Telemetry(StatsStore(self._record.directory))
        self._buffer = PrioritizedReplayBuffer(
            config.replay_capacity,
            TrajectoryCodec(config.max_new_tokens),
            config.per_alpha,
            config.per_beta_start,
            config.per_n_anneal,
            config.per_priority_floor,
            replay_seed,
            stats_q=stats_queue,
        )
        writer = ReplayWriter(self._buffer, write_queue, self._telemetry, self._shutdown_event)
        processor = ActorProcessor(
            self._sampler,
            batch_queue,
            sample_queue,
            RewardModelClient(reward_queue, self._rm_shutdown_event),
            write_queue,
            self._telemetry,
            config.max_new_tokens,
            policy.pad_token_id,
            self._shutdown_event,
        )
        feeder = ActorFeeder(
            self._sampler,
            batch_queue,
            self._telemetry,
            config.actor_batch_size,
            config.max_actor_transitions,
            config.push_per_update_ratio,
            self._shutdown_event,
        )
        reader = ReplayReader(
            self._buffer,
            prefetch_queue,
            config.learner_batch_size,
            config.min_buffer_size,
            config.block_batches,
            self._shutdown_event,
        )
        learner_processor = LearnerProcessor(
            self._buffer,
            self._sampler,
            policy,
            critic_tokenizer,
            prefetch_queue,
            tensor_queue,
            slot_pool,
            self._shutdown_event,
        )
        learner = LearnerGPUWorker(
            tensor_queue,
            slot_pool,
            stats_queue,
            self._telemetry,
            config.learner_device,
            config.stats_every_updates,
            self._shutdown_event,
        )
        stats = StatsProcessor(stats_queue, self._buffer, self._telemetry, config.replay_capacity, self._shutdown_event)

        self._rm_stages = ThreadStages("RLManager.rm", self._rm_shutdown_event)
        self._rm_stages.stage(self._RM_STAGE, self._rm_worker.run)
        # Stages start in the order added: services before producers.
        self._stages = ThreadStages("RLManager", self._shutdown_event)
        self._stages.stage(self._STATS_STAGE, stats.run, inbox=stats_queue, sentinel_after=self._LEARNER_STAGE)
        self._stages.stage(self._ACTOR_GPU_STAGE, actor_worker.run)
        self._stages.stage(self._WRITER_STAGE, writer.run, inbox=write_queue)
        self._stages.stage(self._PROCESSOR_STAGE, processor.run, inbox=batch_queue)
        self._stages.stage(self._FEEDER_STAGE, feeder.run)
        self._stages.stage(self._READER_STAGE, reader.run)
        self._stages.stage(self._LEARNER_PROCESSOR_STAGE, learner_processor.run, inbox=prefetch_queue)
        self._stages.stage(self._LEARNER_STAGE, learner.run, inbox=tensor_queue)
        self._running = False

    @staticmethod
    def _slot_pool(config: TrainConfig) -> queue.Queue:
        """The pool of pinned slots, full; pinned only when the learner is on a GPU."""
        pin_memory = torch.device(config.learner_device).type == "cuda"
        pool: queue.Queue = queue.Queue(maxsize=config.pinned_slots)
        for _ in range(config.pinned_slots):
            pool.put(PinnedSlot(config.learner_batch_size, config.max_new_tokens, config.reward_max_tokens, pin_memory))
        return pool

    def run(self) -> None:
        """Start the reward model, then the pipeline; stop once enough transitions are stored."""
        logger.info("RLManager: run %s in %s", self._record.run_id, self._record.directory)
        status = "failed"
        self._rm_stages.start()
        try:
            self._wait_ready()
            self._stages.start()
            self._running = True
            self._wait_done()
            self._stop_pipeline()
            self._write_summary()
            status = "done"
        finally:
            self._finish(status)
        logger.info("RLManager: run %s done", self._record.run_id)

    def _wait_ready(self) -> None:
        """Block until the reward model is loaded; raise if its thread stopped first."""
        while not self._rm_worker.ready.wait(self._POLL_S):
            if self._rm_shutdown_event.is_set():
                raise RuntimeError("RLManager: the reward model thread stopped before it was ready")

    def _wait_done(self) -> None:
        """Return once ``max_actor_transitions`` are stored, or once a thread sets the shutdown event."""
        while not self._shutdown_event.wait(self._POLL_S):
            if self._telemetry.transitions_pushed >= self._config.max_actor_transitions:
                logger.info("RLManager: %d transitions stored; stopping", self._telemetry.transitions_pushed)
                return
        logger.info("RLManager: shutdown event set by a thread; stopping")

    def _stop_pipeline(self) -> None:
        """Stop the pipeline once; a thread's exception is re-raised here."""
        if self._running:
            self._running = False
            self._stages.stop(self._JOIN_ORDER)

    def _write_summary(self) -> None:
        """The run's totals."""
        self._telemetry.write_counters({
            "transitions_pushed": self._telemetry.transitions_pushed,
            "actor_batches": self._telemetry.actor_batches,
            "update_count": self._telemetry.update_count,
            "buffer_size": len(self._buffer),
            "prompt_passes": self._sampler.pass_counter,
        })

    def _finish(self, status: str) -> None:
        """Stop the pipeline if still running, then the reward model, then record how the run ended."""
        try:
            self._stop_pipeline()
        finally:
            try:
                self._rm_stages.stop([self._RM_STAGE])
            finally:
                self._record.close(status)

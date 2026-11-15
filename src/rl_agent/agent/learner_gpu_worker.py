"""``LearnerGPUWorker`` — the learner thread: rl-core's ``DDPGLearner`` with the per-token critic step.

The critic reads prompt + response once and gives values v_0 … v_T: v_k after the first k response tokens
(v_0 = V(s), at the prompt's end), v_T at the input's last token, where the reward model reads its score. Each row is
an episode of T token steps with reward 0 until the last; with γ = 1 the TD targets are::

    y_k = v'_{k+1}  (target critic), k < T        y_T = r
    loss = Σ_rows w · mean_k (v_k − y_k)² / B      priority = mean_k |v_k − y_k|

The batch is processed ``micro_batch`` rows at a time, gradients accumulated. Every ``stats_every_updates`` updates:
a ``learner_diag`` row (averages since the last one) and a ``CheckpointEvent`` for a buffer stats row. With one
worker per device, each takes its share of every draw and averages gradients through the ``GradientExchange``; only
rank 0 writes the diagnostics (of its own share) and the snapshots.
"""

import logging
import queue
import threading

import torch
from rl_core.critic_step import CriticStep
from rl_core.ddpg_learner import DDPGLearner
from rl_core.events.checkpoint_event import CheckpointEvent
from rl_core.gradient_exchange import GradientExchange
from torch import nn

from src.rl_agent.agent.learner_batch import LearnerBatch
from src.rl_agent.agent.q_batch import QBatch
from src.rl_agent.telemetry.telemetry import Telemetry

logger = logging.getLogger(__name__)


class LearnerGPUWorker(DDPGLearner):
    """Learner thread: tensor queue in; critic updates, priorities, snapshots and diagnostics out."""

    # Bounded-put retry interval for the stats queue.
    _STATS_PUT_TIMEOUT: float = 0.5

    def __init__(
        self,
        critic: nn.Module,
        critic_target: nn.Module,
        optimizer: torch.optim.Optimizer,
        tensor_queue: queue.Queue,
        slot_pool: queue.Queue,
        priority_queue: queue.Queue,
        write_queue: queue.Queue | None,
        stats_queue: queue.Queue,
        telemetry: Telemetry,
        shutdown_event: threading.Event,
        *,
        micro_batch: int,
        stats_every_updates: int,
        tau: float,
        lr_init: float,
        lr_min: float,
        max_updates: int,
        grad_clip: float,
        checkpoint_every: int,
        push_per_update_ratio: float,
        exchange: GradientExchange | None = None,
        rank: int = 0,
    ) -> None:
        """Bind the critic pair, the queues, the slot pool and the cadences; ``rank`` 0 has the write queue."""
        super().__init__(
            critic, critic_target, optimizer, tensor_queue, priority_queue, write_queue, telemetry, shutdown_event,
            tau=tau, lr_init=lr_init, lr_min=lr_min, max_updates=max_updates, grad_clip=grad_clip,
            checkpoint_every=checkpoint_every, push_per_update_ratio=push_per_update_ratio,
            exchange=exchange, rank=rank)
        self._slot_pool = slot_pool
        self._stats_queue = stats_queue
        self._diag = telemetry
        self._micro_batch = micro_batch
        self._stats_every_updates = stats_every_updates
        self._device = next(critic.parameters()).device
        self._sums: dict[str, float] = {}
        self._steps_summed = 0

    def _critic_backward(self, batch: LearnerBatch) -> CriticStep:
        """The per-token TD loss of the batch, backward a micro-batch at a time."""
        q = self._to_device(batch)
        n = q.num_rows
        td_abs: list[torch.Tensor] = []
        sums = {"loss": 0.0, "td_abs": 0.0, "end_abs_error": 0.0, "v0": 0.0, "reward": 0.0, "values_per_row": 0.0}
        for start in range(0, n, self._micro_batch):
            rows = slice(start, min(start + self._micro_batch, n))
            loss, row_td, chunk = self._micro_step(q, rows, n)
            loss.backward()
            td_abs.append(row_td)
            for name, value in chunk.items():
                sums[name] += value
        metrics = {name: value / n for name, value in sums.items()}
        metrics["loss"] = sums["loss"]
        return CriticStep(indices=q.indices, td_abs=torch.cat(td_abs).cpu().numpy(), metrics=metrics)

    def _micro_step(self, q: QBatch, rows: slice, n: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Loss share, per-row |TD error| and summed diagnostics of ``rows``."""
        attention_mask = q.attention_mask[rows]
        width = int(attention_mask.sum(dim=1).max())
        input_ids, attention_mask = q.input_ids[rows, :width], attention_mask[:, :width]
        value_len, reward = q.value_len[rows], q.reward[rows]
        positions, valid, last = self._positions(q.prompt_end[rows], value_len, attention_mask)
        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                next_values = self._critic_target(input_ids, attention_mask).gather(1, positions)
            values = self._critic(input_ids, attention_mask).gather(1, positions)
        targets = torch.cat([next_values[:, 1:], torch.zeros_like(next_values[:, :1])], dim=1)
        targets = torch.where(last, reward.unsqueeze(1), targets)
        error = values.sub(targets).mul(valid)
        row_mse = error.pow(2).sum(dim=1).div(value_len)
        loss = (q.is_weights[rows] * row_mse).sum() / n
        row_td = error.abs().sum(dim=1).div(value_len).detach()
        end_values = values.detach().gather(1, (value_len - 1).unsqueeze(1)).squeeze(1)
        chunk = {
            "loss": float(loss.detach()),
            "td_abs": float(row_td.sum()),
            "end_abs_error": float(end_values.sub(reward).abs().sum()),
            "v0": float(values.detach()[:, 0].sum()),
            "reward": float(reward.sum()),
            "values_per_row": float(value_len.sum()),
        }
        return loss, row_td, chunk

    @staticmethod
    def _positions(
        prompt_end: torch.Tensor, value_len: torch.Tensor, attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B, K)`` input index of each value, which values exist, and which is the last (v_T)."""
        k = torch.arange(int(value_len.max()), device=value_len.device).unsqueeze(0)
        input_end = (attention_mask.sum(dim=1) - 1).unsqueeze(1)
        last = k == (value_len - 1).unsqueeze(1)
        valid = k < value_len.unsqueeze(1)
        positions = torch.where(last | ~valid, input_end, prompt_end.unsqueeze(1) + k)
        return positions, valid, last

    def _to_device(self, batch: LearnerBatch) -> QBatch:
        """The batch's tensors on the learner device; the copies are ordered before the step on the same stream."""
        return QBatch(
            reward=self._move(batch.reward),
            is_weights=self._move(batch.is_weights),
            input_ids=self._move(batch.input_ids),
            attention_mask=self._move(batch.attention_mask),
            prompt_end=self._move(batch.prompt_end),
            value_len=self._move(batch.value_len),
            indices=batch.indices,
        )

    def _move(self, tensor: torch.Tensor) -> torch.Tensor:
        """One host tensor on the learner device, copied without waiting."""
        return tensor.to(self._device, non_blocking=True)

    def _release(self, batch: LearnerBatch) -> None:
        """The slot goes back to the pool once its copies have finished."""
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()
        self._slot_pool.put(batch.slot)

    def _snapshot_state(self) -> dict:
        """The critic's weights as a bfloat16 CPU copy, half the size of the float32 training copy."""
        return {name: tensor.detach().to("cpu", dtype=torch.bfloat16).clone()
                for name, tensor in self._critic.state_dict().items()}

    def _on_step(self, update: int, metrics: dict[str, float]) -> None:
        """Rank 0: sum the step's diagnostics; every ``stats_every_updates`` updates, write and log their averages."""
        if not self.primary:
            return
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + value
        self._steps_summed += 1
        if update % self._stats_every_updates != 0:
            return
        means = {name: value / self._steps_summed for name, value in self._sums.items()}
        self._sums, self._steps_summed = {}, 0
        self._diag.log_learner_diag(update, means)
        logger.info(
            "LearnerGPUWorker: update %d, loss %.4f, |td| %.4f, |v_T - r| %.4f, v0 %.3f, reward %.3f, "
            "%.0f values per row, grad norm %.3f, lr %.2e",
            update, means["loss"], means["td_abs"], means["end_abs_error"], means["v0"], means["reward"],
            means["values_per_row"], means["grad_norm"], means["lr"])
        self._request_stats(update)

    def _request_stats(self, update: int) -> None:
        """Ask for a buffer stats row at this update; waits for room rather than dropping the request."""
        while not self._shutdown_event.is_set():
            try:
                self._stats_queue.put(CheckpointEvent(update), timeout=self._STATS_PUT_TIMEOUT)
                return
            except queue.Full:
                continue

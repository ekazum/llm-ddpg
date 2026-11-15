"""``LearnerGPUWorker`` — the learner thread: rl-core's ``DDPGLearner`` with the per-token critic step.

The critic reads prompt + response once and gives values v_0 … v_T: v_k after the first k response tokens
(v_0 = V(s), at the prompt's end), v_T at the input's last token, where the reward model reads its score. Each row is
an episode of T token steps with reward 0 until the last; with γ = 1 the TD targets are::

    y_k = v'_{k+1}  (target critic), k < T        y_T = r
    loss = Σ_rows w · mean_k (v_k − y_k)² / B      priority = mean_k |v_k − y_k|

The actor step (rank 0, once due) climbs the critic on the same batch. Each chosen token a_k enters the critic as a
straight-through embedding, so the forward pass reads the stored tokens and the gradient reaches the actor::

    e_k = Eᵀ (onehot(a_k) + p_k − sg(p_k)),  p_k = π(· | s_k)     actor loss = −Σ_rows w · mean_{k ≥ 1} v_k / B

The batch is processed ``micro_batch`` rows at a time, gradients accumulated. Every ``stats_every_updates`` updates:
a ``learner_diag`` row (averages since the last one) and a ``CheckpointEvent`` for a buffer stats row. With one
worker per device, each takes its share of every draw and averages gradients through the ``GradientExchange``; only
rank 0 writes the diagnostics (of its own share) and the snapshots.
"""

import logging
import math
import queue
import threading

import torch
from rl_core.critic_step import CriticStep
from rl_core.ddpg_learner import DDPGLearner
from rl_core.events.checkpoint_event import CheckpointEvent
from rl_core.gradient_exchange import GradientExchange
from torch import nn

from src.rl_agent.agent.critic_model import CriticModel
from src.rl_agent.agent.learner_batch import LearnerBatch
from src.rl_agent.agent.q_batch import QBatch
from src.rl_agent.agent.run_snapshot_store import RunSnapshotStore
from src.rl_agent.telemetry.telemetry import Telemetry
from src.rl_agent.token_policy_model import TokenPolicyModel

logger = logging.getLogger(__name__)


class LearnerGPUWorker(DDPGLearner):
    """Learner thread: tensor queue in; critic updates, priorities, snapshots and diagnostics out."""

    # Bounded-put retry interval for the stats queue.
    _STATS_PUT_TIMEOUT: float = 0.5
    # Metrics of an actor step; empty in a learner_diag row whose window had none.
    _ACTOR_METRICS = ("actor_entropy", "actor_logp", "actor_grad_norm", "actor_lr")

    def __init__(
        self,
        critic: CriticModel,
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
        checkpoint_every: int,
        push_per_update_ratio: float,
        exchange: GradientExchange | None = None,
        rank: int = 0,
        policy: TokenPolicyModel | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        actor_lr_init: float = 0.0,
        actor_lr_min: float = 0.0,
        actor_every: int = 1,
        actor_start_update: int = 0,
        actor_micro_batch: int = 1,
    ) -> None:
        """Bind the critic pair, the queues, the slot pool and the cadences; ``rank`` 0 has the write queue.

        ``policy`` and ``actor_optimizer`` (rank 0 only) turn the actor step on; the policy must be on this device.
        """
        super().__init__(
            critic, critic_target, optimizer, tensor_queue, priority_queue, write_queue, telemetry, shutdown_event,
            tau=tau, lr_init=lr_init, lr_min=lr_min, max_updates=max_updates,
            checkpoint_every=checkpoint_every, push_per_update_ratio=push_per_update_ratio,
            exchange=exchange, rank=rank, actor_optimizer=actor_optimizer, actor_lr_init=actor_lr_init,
            actor_lr_min=actor_lr_min, actor_every=actor_every, actor_start_update=actor_start_update)
        self._critic_model = critic
        self._policy = policy
        self._slot_pool = slot_pool
        self._stats_queue = stats_queue
        self._diag = telemetry
        self._micro_batch = micro_batch
        self._actor_micro_batch = actor_micro_batch
        self._stats_every_updates = stats_every_updates
        self._device = next(critic.parameters()).device
        self._on_device: QBatch | None = None
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _critic_backward(self, batch: LearnerBatch) -> CriticStep:
        """The per-token TD loss of the batch, backward a micro-batch at a time."""
        q = self._to_device(batch)
        self._on_device = q
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

    def _actor_backward(self, batch: LearnerBatch) -> dict[str, float]:
        """−Q of the actor's own tokens, back-propagated into the actor a micro-batch at a time.

        Metrics per row, averaged: the actor's mean entropy over the chosen positions, and the stored tokens' mean
        log-probability under it.
        """
        q = self._on_device if self._on_device is not None else self._to_device(batch)
        n = q.num_rows
        sums = {"actor_entropy": 0.0, "actor_logp": 0.0}
        for start in range(0, n, self._actor_micro_batch):
            loss, chunk = self._actor_micro_step(q, slice(start, min(start + self._actor_micro_batch, n)), n)
            loss.backward()
            for name, value in chunk.items():
                sums[name] += value
        return {name: value / n for name, value in sums.items()}

    def _actor_micro_step(self, q: QBatch, rows: slice, n: int) -> tuple[torch.Tensor, dict[str, float]]:
        """Actor loss share of ``rows``, and the sums over its rows of the actor metrics."""
        policy = self._policy
        if policy is None:
            raise RuntimeError("LearnerGPUWorker: the actor step needs the policy")
        attention_mask = q.attention_mask[rows]
        width = int(attention_mask.sum(dim=1).max())
        input_ids, attention_mask = q.input_ids[rows, :width], attention_mask[:, :width]
        prompt_end, value_len, action_len = q.prompt_end[rows], q.value_len[rows], q.action_len[rows]
        k = torch.arange(max(int(action_len.max()), 1), device=self._device).unsqueeze(0)
        acting = k < action_len.unsqueeze(1)
        token_at = (prompt_end.unsqueeze(1) + 1 + k).clamp(max=width - 1)
        weight = self._critic_model.embedding_weight
        positions, valid, _ = self._positions(prompt_end, value_len, attention_mask)
        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16):
            log_probs = policy.logits_at(input_ids, attention_mask, token_at - 1).float().log_softmax(dim=-1)
            probs = log_probs.exp()
            # Zero in the forward pass, so the critic reads the stored tokens; the gradient reaches the probabilities.
            straight = torch.matmul(probs.sub(probs.detach()), weight).mul(acting.unsqueeze(-1))
            embeds = nn.functional.embedding(input_ids, weight)
            embeds = embeds.scatter_add(
                1, token_at.unsqueeze(-1).expand(-1, -1, embeds.shape[-1]), straight.to(embeds.dtype))
            values = self._critic_model.values_from_embeddings(embeds, attention_mask).gather(1, positions)
        after = valid.clone()
        after[:, 0] = False
        row_value = values.mul(after).sum(dim=1).div((value_len - 1).clamp(min=1))
        loss = -(q.is_weights[rows] * row_value).sum() / n
        with torch.no_grad():
            per_action = action_len.clamp(min=1)
            entropy = probs.mul(log_probs).sum(dim=-1).neg().mul(acting).sum(dim=1).div(per_action)
            chosen = log_probs.gather(-1, input_ids.gather(1, token_at).unsqueeze(-1)).squeeze(-1)
            logp = chosen.mul(acting).sum(dim=1).div(per_action)
        return loss, {"actor_entropy": float(entropy.sum()), "actor_logp": float(logp.sum())}

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
            action_len=self._move(batch.action_len),
            indices=batch.indices,
        )

    def _move(self, tensor: torch.Tensor) -> torch.Tensor:
        """One host tensor on the learner device, copied without waiting."""
        return tensor.to(self._device, non_blocking=True)

    def _release(self, batch: LearnerBatch) -> None:
        """The slot goes back to the pool once its copies have finished."""
        self._on_device = None
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()
        self._slot_pool.put(batch.slot)

    def _snapshot_state(self) -> dict:
        """The critic's weights, and the actor's when it trains here, as bfloat16 CPU copies."""
        state = {RunSnapshotStore.CRITIC: self._bfloat16_copy(self._critic.state_dict())}
        if self._policy is not None:
            state[RunSnapshotStore.ACTOR] = self._bfloat16_copy(self._policy.state_dict())
        return state

    @staticmethod
    def _bfloat16_copy(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """A bfloat16 CPU copy of each tensor, half the size of the float32 training copy."""
        return {name: tensor.detach().to("cpu", dtype=torch.bfloat16).clone() for name, tensor in weights.items()}

    def _on_step(self, update: int, metrics: dict[str, float]) -> None:
        """Rank 0: sum the step's diagnostics; every ``stats_every_updates`` updates, write and log their averages."""
        if not self.primary:
            return
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + value
            self._counts[name] = self._counts.get(name, 0) + 1
        if update % self._stats_every_updates != 0:
            return
        means = {name: value / self._counts[name] for name, value in self._sums.items()}
        if self._policy is not None:
            # NaN is stored as NULL, in a column typed from the first row's floats.
            for name in self._ACTOR_METRICS:
                means.setdefault(name, math.nan)
        self._sums, self._counts = {}, {}
        self._diag.log_learner_diag(update, means)
        logger.info(
            "LearnerGPUWorker: update %d, loss %.4f, |td| %.4f, |v_T - r| %.4f, v0 %.3f, reward %.3f, "
            "%.0f values per row, grad norm %.3f, lr %.2e%s",
            update, means["loss"], means["td_abs"], means["end_abs_error"], means["v0"], means["reward"],
            means["values_per_row"], means["grad_norm"], means["lr"], self._actor_line(means))
        self._request_stats(update)

    @staticmethod
    def _actor_line(means: dict[str, float]) -> str:
        """The actor's part of the diagnostics log line; empty while it has not stepped."""
        entropy, logp, grad_norm, lr = (means.get(name, math.nan) for name in LearnerGPUWorker._ACTOR_METRICS)
        if math.isnan(entropy):
            return ""
        return (f"; actor entropy {entropy:.3f}, actor logp {logp:.3f}, actor grad norm {grad_norm:.3f}, "
                f"actor lr {lr:.2e}")

    def _request_stats(self, update: int) -> None:
        """Ask for a buffer stats row at this update; waits for room rather than dropping the request."""
        while not self._shutdown_event.is_set():
            try:
                self._stats_queue.put(CheckpointEvent(update), timeout=self._STATS_PUT_TIMEOUT)
                return
            except queue.Full:
                continue

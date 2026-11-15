"""``RunSnapshotStore`` — the critic and actor snapshots of a training run, and the latest training state.

rl-core's ``CheckpointWriter`` calls it (``SnapshotStoreProtocol``). Per snapshot: ``critic/critic_<update>.pt`` and
``actor/actor_<update>/`` (a Hugging Face model directory). Each is written under a temporary name and renamed, so a
path under its final name is complete; then ``(update, actor directory)`` goes on the snapshot queue for dev scoring.
"""

import logging
import os
import queue
import shutil
from pathlib import Path

import torch
from rl_core.checkpoint_state import CheckpointState

from src.rl_agent.token_policy_model import TokenPolicyModel

logger = logging.getLogger(__name__)


class RunSnapshotStore:
    """Snapshot files per update, ``critic/latest.pt`` for the full state; deletes a whole update's snapshot."""

    CRITIC_DIRNAME = "critic"
    ACTOR_DIRNAME = "actor"
    LATEST = "latest.pt"
    # Keys of the snapshot state dict the learner hands over.
    CRITIC = "critic"
    ACTOR = "actor"

    def __init__(self, run_directory: Path, saver: TokenPolicyModel, snapshot_queue: queue.Queue) -> None:
        """Create ``critic/`` and ``actor/``; ``saver`` gives the actor directory its config and tokenizer."""
        self._critic_dir = Path(run_directory) / self.CRITIC_DIRNAME
        self._actor_dir = Path(run_directory) / self.ACTOR_DIRNAME
        self._critic_dir.mkdir(parents=True, exist_ok=True)
        self._actor_dir.mkdir(parents=True, exist_ok=True)
        self._saver = saver
        self._snapshot_queue = snapshot_queue

    def write_snapshot(self, update: int, q_online_state_dict: dict, dims: dict) -> Path:
        """Write the critic (and the actor, when given) at ``update``; return the critic file's path."""
        path = self._critic_path(update)
        self._save({"update": update, "state_dict": q_online_state_dict[self.CRITIC], "dims": dims}, path)
        logger.info("RunSnapshotStore: %s, %.2f GB", path.name, path.stat().st_size / 1e9)
        if self.ACTOR in q_online_state_dict:
            self._write_actor(update, q_online_state_dict[self.ACTOR])
        return path

    def _write_actor(self, update: int, state_dict: dict[str, torch.Tensor]) -> None:
        """The actor directory of ``update``, then its ``(update, directory)`` on the snapshot queue."""
        directory = self.actor_path(update)
        partial = directory.with_name(directory.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        self._saver.save(partial, state_dict)
        os.replace(partial, directory)
        logger.info("RunSnapshotStore: %s written", directory.name)
        self._snapshot_queue.put((update, directory))

    def write_latest(self, state: CheckpointState) -> None:
        """Write the full training state for resume."""
        self._save(state.to_dict(), self._critic_dir / self.LATEST)

    @staticmethod
    def delete_snapshot(path: Path) -> None:
        """Remove one snapshot file."""
        path.unlink()
        logger.info("RunSnapshotStore: deleted %s", path.name)

    def delete_update(self, update: int) -> None:
        """Remove the critic file and the actor directory of ``update``."""
        critic = self._critic_path(update)
        if critic.exists():
            critic.unlink()
        shutil.rmtree(self.actor_path(update), ignore_errors=True)
        logger.info("RunSnapshotStore: deleted the snapshot at update %d", update)

    def actor_path(self, update: int) -> Path:
        """The actor directory of ``update``."""
        return self._actor_dir / f"actor_{update:07d}"

    def _critic_path(self, update: int) -> Path:
        """The critic file of ``update``."""
        return self._critic_dir / f"critic_{update:07d}.pt"

    @staticmethod
    def _save(payload: dict, path: Path) -> None:
        """``torch.save`` under a temporary name, then rename."""
        partial = path.with_name(path.name + ".partial")
        torch.save(payload, partial)
        os.replace(partial, path)

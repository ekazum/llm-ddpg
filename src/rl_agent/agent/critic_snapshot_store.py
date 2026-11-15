"""``CriticSnapshotStore`` — critic snapshots and the latest training state, in the run directory's ``critic/``.

rl-core's ``CheckpointWriter`` calls it (``SnapshotStoreProtocol``). Each file is written under a temporary name and
renamed, so a file under its final name is complete.
"""

import logging
import os
from pathlib import Path

import torch
from rl_core.checkpoint_state import CheckpointState

logger = logging.getLogger(__name__)


class CriticSnapshotStore:
    """``critic_<update>.pt`` per snapshot, ``latest.pt`` for the full state."""

    DIRNAME = "critic"
    LATEST = "latest.pt"

    def __init__(self, run_directory: Path) -> None:
        """Create ``critic/`` under the run directory."""
        self._directory = Path(run_directory) / self.DIRNAME
        self._directory.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, update: int, q_online_state_dict: dict, dims: dict) -> Path:
        """Write the critic's weights at ``update``; return the file's path."""
        path = self._directory / f"critic_{update:07d}.pt"
        self._save({"update": update, "state_dict": q_online_state_dict, "dims": dims}, path)
        logger.info("CriticSnapshotStore: %s, %.2f GB", path.name, path.stat().st_size / 1e9)
        return path

    def write_latest(self, state: CheckpointState) -> None:
        """Write the full training state for resume."""
        self._save(state.to_dict(), self._directory / self.LATEST)

    @staticmethod
    def delete_snapshot(path: Path) -> None:
        """Remove one snapshot file."""
        path.unlink()
        logger.info("CriticSnapshotStore: deleted %s", path.name)

    @staticmethod
    def _save(payload: dict, path: Path) -> None:
        """``torch.save`` under a temporary name, then rename."""
        partial = path.with_name(path.name + ".partial")
        torch.save(payload, partial)
        os.replace(partial, path)

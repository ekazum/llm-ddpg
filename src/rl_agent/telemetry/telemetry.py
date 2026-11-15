"""``Telemetry`` — the run's progress counters and its stats rows, behind one object.

Writers advance the counters (``ReplayWriter``: transitions, ``ActorProcessor``: actor batches, the learner:
updates); readers take them without waiting. The ``log_*`` methods write rows to the ``StatsStore``.
"""

from ai_runtime import LockedCounter

from src.rl_agent.telemetry.stats_store import StatsStore


class Telemetry:
    """Shared counters of a training run and the writer of its ``stats.db`` rows."""

    def __init__(self, store: StatsStore) -> None:
        self._store = store
        self._transitions = LockedCounter()
        self._actor_batches = LockedCounter()
        # Single writer (the learner); an int read is not torn.
        self._update_count = 0

    @property
    def transitions_pushed(self) -> int:
        """Transitions stored in the replay buffer so far."""
        return self._transitions.value

    @property
    def update_count(self) -> int:
        """Learner updates done so far."""
        return self._update_count

    @property
    def actor_batches(self) -> int:
        """Actor batches written so far."""
        return self._actor_batches.value

    def note_transitions(self, n: int) -> None:
        """Count ``n`` more stored transitions (``ReplayWriter`` only)."""
        self._transitions.add(n)

    def note_update(self, count: int) -> None:
        """Set the learner's update count (the learner only)."""
        self._update_count = count

    def add_actor_batch(self) -> int:
        """Count one more actor batch; return its index, from 1."""
        return self._actor_batches.add(1)

    def log_actor_batch(self, metrics: dict) -> None:
        """Write one ``actor_batches`` row, keyed by ``actor_batch``."""
        self._store.append(StatsStore.ACTOR_BATCHES, [metrics], key="actor_batch")

    def log_buffer_stats(self, episode_index: int, metrics: dict) -> None:
        """Write one ``buffer`` row; ``episode_index`` is the learner update count it is keyed by."""
        self._store.append(StatsStore.BUFFER, [{"update_count": episode_index, **metrics}], key="update_count")

    def log_learner_diag(self, update_count: int, metrics: dict) -> None:
        """Write one ``learner_diag`` row, keyed by ``update_count``."""
        self._store.append(StatsStore.LEARNER_DIAG, [{"update_count": update_count, **metrics}], key="update_count")

    def write_counters(self, values: dict) -> None:
        """Upsert the run's totals."""
        self._store.set_pairs(StatsStore.COUNTERS, values)

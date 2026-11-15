"""``StatsStore`` — one training run's telemetry, in the run directory's ``stats.db``.

Rows land as the run produces them. Tables: ``actor_batches`` (one row per actor batch), ``buffer`` (replay
buffer stats, keyed by learner update), ``learner_diag`` (critic step averages, keyed by learner update) and
``counters`` (the run's totals).
"""

from ai_runtime import SqliteStore


class StatsStore(SqliteStore):
    """The training-side store; tables are created from their first row."""

    FILENAME = "stats.db"
    ACTOR_BATCHES = "actor_batches"
    BUFFER = "buffer"
    LEARNER_DIAG = "learner_diag"
    COUNTERS = "counters"

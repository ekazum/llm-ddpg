"""``RunRecord`` — one run's directory under the records root, with the ``meta.json`` a runs dashboard reads."""

import json
import os
from datetime import datetime
from pathlib import Path


class RunRecord:
    """Creates ``<records_dir>/<kind>_<timestamp>/`` and keeps its ``meta.json`` current."""

    META = "meta.json"

    def __init__(self, records_dir: str, kind: str, label: str) -> None:
        created = datetime.now().astimezone()
        self._run_id = f"{kind}_{created.strftime('%Y%m%d_%H%M%S')}"
        self._dir = Path(records_dir) / self._run_id
        self._dir.mkdir(parents=True, exist_ok=False)
        self._meta = {
            "run_id": self._run_id,
            "kind": kind,
            "label": label,
            "created_at": created.isoformat(timespec="seconds"),
            "status": "running",
        }
        self._write()

    @property
    def run_id(self) -> str:
        """The run's id, also its directory name."""
        return self._run_id

    @property
    def directory(self) -> Path:
        """The run's directory."""
        return self._dir

    def close(self, status: str) -> None:
        """Record how the run ended."""
        self._meta["status"] = status
        self._meta["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self._write()

    def _write(self) -> None:
        """Write ``meta.json`` through a temporary file and a rename."""
        tmp = self._dir / f"{self.META}.tmp"
        tmp.write_text(json.dumps(self._meta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self._dir / self.META)

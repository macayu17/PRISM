from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class PauseRequested(RuntimeError):
    """Raised when a training run should pause after saving progress."""


class StopRequested(RuntimeError):
    """Raised when a training run should stop gracefully."""


@dataclass(frozen=True)
class TrainingPaths:
    root: Path
    run_name: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_name

    @property
    def state_path(self) -> Path:
        return self.run_dir / "run_state.json"

    @property
    def pause_flag(self) -> Path:
        return self.run_dir / "pause.flag"

    @property
    def stop_flag(self) -> Path:
        return self.run_dir / "stop.flag"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"


class TrainingRunController:
    """Shared manifest/checkpoint controller for resumable training runs."""

    def __init__(self, base_dir: Path | str, run_name: str):
        self.paths = TrainingPaths(Path(base_dir), run_name)
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state = _load_json(
            self.paths.state_path,
            {
                "run_name": run_name,
                "status": "created",
                "created_at": _iso_now(),
                "updated_at": _iso_now(),
                "selected_models": [],
                "config": {},
                "current": {},
                "models": {},
                "events": [],
            },
        )
        self._flush()

    def _flush(self) -> None:
        self.state["updated_at"] = _iso_now()
        _atomic_write_json(self.paths.state_path, self.state)

    def initialize(self, selected_models: Iterable[str], config: Dict[str, Any], resume: bool = False) -> None:
        if not resume or not self.state.get("selected_models"):
            self.state["selected_models"] = list(selected_models)
            self.state["config"] = config
        self._flush()

    def get_status(self) -> str:
        return str(self.state.get("status", "created"))

    def mark_running(self, stage: str, model_name: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        self.state["status"] = "running"
        current = {"stage": stage}
        if model_name:
            current["model_name"] = model_name
        if extra:
            current.update(extra)
        self.state["current"] = current
        self._append_event("running", current)
        self._flush()

    def mark_paused(self, reason: str = "pause requested") -> None:
        self.state["status"] = "paused"
        self._append_event("paused", {"reason": reason})
        self._flush()

    def mark_stopped(self, reason: str = "stop requested") -> None:
        self.state["status"] = "stopped"
        self._append_event("stopped", {"reason": reason})
        self._flush()

    def mark_completed(self) -> None:
        self.state["status"] = "completed"
        self.state["completed_at"] = _iso_now()
        self._append_event("completed", {})
        self.clear_pause()
        self.clear_stop()
        self._flush()

    def mark_failed(self, error_message: str) -> None:
        self.state["status"] = "failed"
        self.state["last_error"] = error_message
        self._append_event("failed", {"error": error_message})
        self._flush()

    def _append_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        events = self.state.setdefault("events", [])
        events.append({"at": _iso_now(), "type": event_type, **payload})
        if len(events) > 100:
            del events[:-100]

    def update_model_state(self, model_name: str, **fields: Any) -> None:
        models = self.state.setdefault("models", {})
        model_state = models.setdefault(model_name, {})
        model_state.update(fields)
        model_state["updated_at"] = _iso_now()
        self._flush()

    def append_trial_result(self, model_name: str, result: Dict[str, Any]) -> None:
        models = self.state.setdefault("models", {})
        model_state = models.setdefault(model_name, {})
        trial_results = model_state.setdefault("trial_results", [])
        trial_results.append(result)
        model_state["updated_at"] = _iso_now()
        self._flush()

    def write_metrics_file(self, filename: str, payload: Dict[str, Any]) -> Path:
        target = self.paths.metrics_dir / filename
        _atomic_write_json(target, payload)
        return target

    def request_pause(self) -> None:
        self.paths.pause_flag.parent.mkdir(parents=True, exist_ok=True)
        self.paths.pause_flag.write_text("pause\n", encoding="utf-8")

    def clear_pause(self) -> None:
        if self.paths.pause_flag.exists():
            self.paths.pause_flag.unlink()

    def request_stop(self) -> None:
        self.paths.stop_flag.parent.mkdir(parents=True, exist_ok=True)
        self.paths.stop_flag.write_text("stop\n", encoding="utf-8")

    def clear_stop(self) -> None:
        if self.paths.stop_flag.exists():
            self.paths.stop_flag.unlink()

    def raise_if_requested(self) -> None:
        if self.paths.stop_flag.exists():
            raise StopRequested("Stop requested via stop.flag")
        if self.paths.pause_flag.exists():
            raise PauseRequested("Pause requested via pause.flag")

    def load_checkpoint_state(self, model_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        path = self.paths.checkpoints_dir / f"{model_name}.json"
        return _load_json(path, default)

    def save_checkpoint_state(self, model_name: str, payload: Dict[str, Any]) -> Path:
        path = self.paths.checkpoints_dir / f"{model_name}.json"
        _atomic_write_json(path, payload)
        return path

    def status_summary(self) -> Dict[str, Any]:
        return {
            "run_name": self.state.get("run_name"),
            "status": self.state.get("status"),
            "selected_models": self.state.get("selected_models", []),
            "current": self.state.get("current", {}),
            "models": self.state.get("models", {}),
            "paths": {
                "run_dir": str(self.paths.run_dir),
                "state_path": str(self.paths.state_path),
                "pause_flag": str(self.paths.pause_flag),
                "stop_flag": str(self.paths.stop_flag),
                "checkpoints_dir": str(self.paths.checkpoints_dir),
                "metrics_dir": str(self.paths.metrics_dir),
            },
        }

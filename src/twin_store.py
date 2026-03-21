from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from twin_schema import DigitalTwin


class TwinStore:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            db_path = str(project_root / "data" / "digital_twins.sqlite3")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS twins (
                    twin_id TEXT PRIMARY KEY,
                    patient_label TEXT NOT NULL,
                    source_patno INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    current_cohort_estimate TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot_count INTEGER NOT NULL,
                    profile_json TEXT NOT NULL,
                    current_state_json TEXT NOT NULL,
                    forecast_json TEXT NOT NULL,
                    prediction_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS twin_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    twin_id TEXT NOT NULL,
                    snapshot_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    FOREIGN KEY (twin_id) REFERENCES twins(twin_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_twin_snapshots_twin_id ON twin_snapshots(twin_id)"
            )
            conn.commit()

    def upsert_twin(self, twin: DigitalTwin) -> Dict[str, Any]:
        payload = twin.to_dict()
        profile = payload["profile"]
        state = payload["current_state"]
        forecast = payload["forecast"]
        prediction = payload.get("prediction_summary", {})
        snapshots = payload["snapshots"]

        with closing(self._connect()) as conn:
            existing = conn.execute(
                "SELECT twin_id FROM twins WHERE twin_id = ?",
                (profile["twin_id"],),
            ).fetchone()

            values = (
                profile["twin_id"],
                profile.get("patient_label") or profile["twin_id"],
                profile.get("source_patno"),
                profile["created_at"],
                state["computed_at"],
                state["current_cohort_estimate"],
                float(state.get("confidence") or 0.0),
                len(snapshots),
                json.dumps(profile),
                json.dumps(state),
                json.dumps(forecast),
                json.dumps(prediction),
            )

            if existing:
                conn.execute(
                    """
                    UPDATE twins
                    SET patient_label = ?, source_patno = ?, created_at = ?, updated_at = ?,
                        current_cohort_estimate = ?, confidence = ?, snapshot_count = ?,
                        profile_json = ?, current_state_json = ?, forecast_json = ?, prediction_json = ?
                    WHERE twin_id = ?
                    """,
                    (
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[9],
                        values[10],
                        values[11],
                        values[0],
                    ),
                )
                conn.execute(
                    "DELETE FROM twin_snapshots WHERE twin_id = ?",
                    (profile["twin_id"],),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO twins (
                        twin_id, patient_label, source_patno, created_at, updated_at,
                        current_cohort_estimate, confidence, snapshot_count,
                        profile_json, current_state_json, forecast_json, prediction_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )

            for index, snapshot in enumerate(snapshots):
                conn.execute(
                    """
                    INSERT INTO twin_snapshots (
                        snapshot_id, twin_id, snapshot_index, created_at, snapshot_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot["snapshot_id"],
                        profile["twin_id"],
                        index,
                        snapshot.get("visit_date") or profile["created_at"],
                        json.dumps(snapshot),
                    ),
                )
            conn.commit()

        stored = self.get_twin(profile["twin_id"])
        if stored is None:
            raise RuntimeError("Failed to persist digital twin")
        return stored

    def list_twins(self) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT twin_id, patient_label, source_patno, created_at, updated_at,
                       current_cohort_estimate, confidence, snapshot_count
                FROM twins
                ORDER BY updated_at DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def get_twin(self, twin_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT twin_id, patient_label, source_patno, created_at, updated_at,
                       current_cohort_estimate, confidence, snapshot_count,
                       profile_json, current_state_json, forecast_json, prediction_json
                FROM twins
                WHERE twin_id = ?
                """,
                (twin_id,),
            ).fetchone()

            if row is None:
                return None

            snapshots = conn.execute(
                """
                SELECT snapshot_json
                FROM twin_snapshots
                WHERE twin_id = ?
                ORDER BY snapshot_index ASC
                """,
                (twin_id,),
            ).fetchall()

        return {
            "profile": json.loads(row["profile_json"]),
            "current_state": json.loads(row["current_state_json"]),
            "forecast": json.loads(row["forecast_json"]),
            "prediction_summary": json.loads(row["prediction_json"]),
            "snapshots": [json.loads(snapshot["snapshot_json"]) for snapshot in snapshots],
            "summary": {
                "twin_id": row["twin_id"],
                "patient_label": row["patient_label"],
                "source_patno": row["source_patno"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "current_cohort_estimate": row["current_cohort_estimate"],
                "confidence": row["confidence"],
                "snapshot_count": row["snapshot_count"],
            },
        }

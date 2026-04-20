"""
Progression Modeling — PPMI Trajectory Clustering.

Extracts longitudinal PPMI visit data, clusters patients into
slow / moderate / fast progressors using k-means on velocity
vectors, and provides cluster-weighted forecast predictions.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cluster labels in order of severity (index 0 = slowest)
# ---------------------------------------------------------------------------
CLUSTER_LABELS = ["slow", "moderate", "fast"]

# Yearly progression defaults per cluster (UPDRS3 gain/yr, MoCA loss/yr, HY gain/yr)
DEFAULT_CLUSTER_PROFILES = {
    "slow":     {"updrs3_gain": 1.8, "moca_loss": 0.3, "hy_gain": 0.10},
    "moderate": {"updrs3_gain": 3.5, "moca_loss": 0.7, "hy_gain": 0.25},
    "fast":     {"updrs3_gain": 6.5, "moca_loss": 1.3, "hy_gain": 0.45},
}


class ProgressionModel:
    """PPMI-trained trajectory clustering for Parkinson's progression."""

    def __init__(self) -> None:
        self.fitted = False
        self.centroids: Optional[np.ndarray] = None
        self.cluster_profiles: Dict[str, Dict[str, float]] = dict(DEFAULT_CLUSTER_PROFILES)
        self.silhouette_score_: Optional[float] = None
        self.k = 3
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, csv_path: str) -> "ProgressionModel":
        """Fit the model from a PPMI CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, low_memory=False)
            return self._fit_from_dataframe(df)
        except Exception as exc:
            logger.warning("ProgressionModel.fit failed: %s — using defaults", exc)
            self.fitted = False
            return self

    def _fit_from_dataframe(self, df: "pd.DataFrame") -> "ProgressionModel":
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        required = {"PATNO", "YEAR", "updrs3_score", "moca"}
        if not required.issubset(set(df.columns)):
            logger.warning("Missing columns for progression model; using defaults.")
            return self

        # Keep only rows with valid data
        sub = df[["PATNO", "YEAR", "updrs3_score", "moca", "hy"]].dropna(
            subset=["PATNO", "YEAR", "updrs3_score"]
        ).copy()
        sub["YEAR"] = pd.to_numeric(sub["YEAR"], errors="coerce")
        sub["updrs3_score"] = pd.to_numeric(sub["updrs3_score"], errors="coerce")
        sub["moca"] = pd.to_numeric(sub["moca"], errors="coerce")
        sub = sub.dropna(subset=["YEAR", "updrs3_score"])

        # Compute per-patient velocity vectors
        velocities: List[List[float]] = []
        patnos: List[int] = []
        for patno, grp in sub.groupby("PATNO"):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("YEAR")
            years = grp["YEAR"].values
            span = years[-1] - years[0]
            if span < 0.5:
                continue
            delta_updrs = (grp["updrs3_score"].values[-1] - grp["updrs3_score"].values[0]) / span
            moca_vals = grp["moca"].dropna()
            if len(moca_vals) >= 2:
                delta_moca = (moca_vals.values[-1] - moca_vals.values[0]) / span
            else:
                delta_moca = 0.0
            velocities.append([delta_updrs, delta_moca])
            patnos.append(int(patno))

        if len(velocities) < 10:
            logger.warning("Too few longitudinal patients (%d); using defaults.", len(velocities))
            return self

        X = np.array(velocities)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler_mean = scaler.mean_
        self._scaler_std = scaler.scale_

        # Try k=3, fall back to k=2 if silhouette < 0.3
        best_k = 3
        best_score = -1.0
        best_km = None

        for k in [3, 2]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            try:
                from sklearn.metrics import silhouette_score as _sil
                score = _sil(X_scaled, labels)
            except Exception:
                score = 0.0
            logger.info("ProgressionModel k=%d silhouette=%.3f", k, score)
            if score > best_score:
                best_score = score
                best_k = k
                best_km = km

        if best_score < 0.3 and best_k == 3:
            # Retry with k=2
            km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels2 = km2.fit_predict(X_scaled)
            try:
                from sklearn.metrics import silhouette_score as _sil
                s2 = _sil(X_scaled, labels2)
            except Exception:
                s2 = 0.0
            if s2 > best_score:
                best_score = s2
                best_k = 2
                best_km = km2

        self.k = best_k
        self.silhouette_score_ = float(best_score)
        self.centroids = best_km.cluster_centers_  # type: ignore[union-attr]

        # Sort clusters by ascending UPDRS velocity (index 0 in velocity vec)
        centroid_means = scaler.inverse_transform(self.centroids)
        order = np.argsort(centroid_means[:, 0])
        self.centroids = self.centroids[order]
        centroid_means = centroid_means[order]

        # Build profiles from centroids
        labels_used = CLUSTER_LABELS[: self.k]
        self.cluster_profiles = {}
        for idx, label in enumerate(labels_used):
            updrs_vel = max(centroid_means[idx, 0], 0.5)
            moca_vel = abs(centroid_means[idx, 1]) if centroid_means.shape[1] > 1 else 0.3
            hy_gain = 0.10 + idx * 0.15
            self.cluster_profiles[label] = {
                "updrs3_gain": round(float(updrs_vel), 2),
                "moca_loss": round(float(moca_vel), 2),
                "hy_gain": round(float(hy_gain), 2),
            }

        self.fitted = True
        logger.info(
            "ProgressionModel fitted: k=%d, silhouette=%.3f, profiles=%s",
            self.k, self.silhouette_score_, self.cluster_profiles,
        )
        return self

    def assign_cluster(
        self,
        snapshots: List[Dict[str, Any]],
        patient_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Assign a patient to a progression cluster.

        Returns (cluster_id, cluster_label).
        - If ≥ 2 snapshots with enough time span: use velocity vector.
        - Otherwise: use current feature vector as proxy.
        """
        if not self.fitted or self.centroids is None:
            return self._heuristic_assign(snapshots, patient_data)

        # Try velocity-based assignment (≥ 2 visits)
        if len(snapshots) >= 2:
            cluster = self._velocity_assign(snapshots)
            if cluster is not None:
                return cluster

        # Feature-based fallback
        return self._feature_assign(snapshots, patient_data)

    def get_cluster_profile(self, cluster_label: str) -> Dict[str, float]:
        """Return progression rates for a cluster."""
        return self.cluster_profiles.get(cluster_label, DEFAULT_CLUSTER_PROFILES.get("moderate", {}))

    def cluster_weighted_forecast(
        self,
        cluster_label: str,
        current_updrs3: float,
        current_moca: float,
        current_hy: float,
        horizon_months: int,
        duration_years: float = 0.0,
    ) -> Dict[str, Optional[float]]:
        """Compute cluster-weighted forecast for a given horizon."""
        profile = self.get_cluster_profile(cluster_label)
        years = horizon_months / 12.0

        # Duration-dependent acceleration factor
        accel = 1.0 + duration_years * 0.02

        pred_updrs3 = current_updrs3 + profile["updrs3_gain"] * years * accel
        pred_moca = max(0, min(30, current_moca - profile["moca_loss"] * years))
        pred_hy = max(0, min(5, current_hy + profile["hy_gain"] * years))
        pred_total = pred_updrs3 * 1.7 + 8

        return {
            "predicted_updrs3": round(pred_updrs3, 2),
            "predicted_total_updrs": round(pred_total, 2),
            "predicted_moca": round(pred_moca, 2),
            "predicted_hy": round(pred_hy, 2),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _velocity_assign(self, snapshots: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
        """Assign using velocity vector from snapshot history."""
        first = snapshots[0]
        last = snapshots[-1]

        y0 = self._get_year(first)
        y1 = self._get_year(last)
        if y0 is None or y1 is None or (y1 - y0) < 0.5:
            return None

        span = y1 - y0
        u0 = self._get_updrs3(first)
        u1 = self._get_updrs3(last)
        if u0 is None or u1 is None:
            return None

        delta_updrs = (u1 - u0) / span

        m0 = self._get_moca(first)
        m1 = self._get_moca(last)
        delta_moca = ((m1 - m0) / span) if (m0 is not None and m1 is not None) else 0.0

        vec = np.array([[delta_updrs, delta_moca]])

        if self._scaler_mean is not None and self._scaler_std is not None:
            vec = (vec - self._scaler_mean) / np.clip(self._scaler_std, 1e-8, None)

        distances = np.linalg.norm(self.centroids - vec, axis=1)
        idx = int(np.argmin(distances))
        labels = CLUSTER_LABELS[: self.k]
        return (str(idx), labels[idx])

    def _feature_assign(
        self,
        snapshots: List[Dict[str, Any]],
        patient_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Assign using current feature vector as proxy."""
        latest = snapshots[-1] if snapshots else (patient_data or {})
        updrs3 = self._get_updrs3(latest)
        moca = self._get_moca(latest)

        # Simple heuristic proxy velocity from current scores
        updrs_proxy = (updrs3 or 10.0) / max(self._get_duration(latest) or 3.0, 1.0)
        moca_proxy = -((moca or 26.0) - 28.0) / max(self._get_duration(latest) or 3.0, 1.0)

        vec = np.array([[updrs_proxy, moca_proxy]])
        if self._scaler_mean is not None and self._scaler_std is not None:
            vec = (vec - self._scaler_mean) / np.clip(self._scaler_std, 1e-8, None)

        if self.centroids is not None:
            distances = np.linalg.norm(self.centroids - vec, axis=1)
            idx = int(np.argmin(distances))
        else:
            idx = 1  # moderate default

        labels = CLUSTER_LABELS[: self.k]
        return (str(idx), labels[min(idx, len(labels) - 1)])

    def _heuristic_assign(
        self,
        snapshots: List[Dict[str, Any]],
        patient_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Fallback heuristic when model is not fitted."""
        latest = snapshots[-1] if snapshots else (patient_data or {})
        updrs3 = self._get_updrs3(latest) or 0.0
        if updrs3 >= 25:
            return ("2", "fast")
        elif updrs3 >= 12:
            return ("1", "moderate")
        return ("0", "slow")

    # ------------------------------------------------------------------
    @staticmethod
    def _get_updrs3(d: Dict[str, Any]) -> Optional[float]:
        val = d.get("motor", d).get("updrs3_score") if isinstance(d.get("motor"), dict) else d.get("updrs3_score")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_moca(d: Dict[str, Any]) -> Optional[float]:
        val = d.get("cognition", d).get("moca") if isinstance(d.get("cognition"), dict) else d.get("moca")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_year(d: Dict[str, Any]) -> Optional[float]:
        val = d.get("year_index")
        if val is None:
            val = d.get("YEAR")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_duration(d: Dict[str, Any]) -> Optional[float]:
        val = d.get("duration_years") or d.get("duration_yrs")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

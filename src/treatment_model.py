"""
Treatment Response Model — LEDD → UPDRS Dose-Response Curve.

Extracts PPMI treatment data (patients on PD treatment with ON/OFF
UPDRS3 scores), fits a log-decay relationship between LEDD dose and
UPDRS3 improvement, and predicts treatment effect for new patients.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default coefficients for the dose-response curve when PPMI fit fails.
# UPDRS_reduction = a * log(1 + LEDD) + b * disease_duration + c
_DEFAULT_COEFS = {"a": 3.5, "b": -0.3, "c": -2.0}


class TreatmentModel:
    """Dose-response model: LEDD → UPDRS3 reduction."""

    def __init__(self) -> None:
        self.fitted = False
        self.coefs: Dict[str, float] = dict(_DEFAULT_COEFS)
        self.r_squared: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, csv_path: str) -> "TreatmentModel":
        """Fit the dose-response model from PPMI CSV data."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, low_memory=False)
            return self._fit_from_dataframe(df)
        except Exception as exc:
            logger.warning("TreatmentModel.fit failed: %s — using defaults", exc)
            self.fitted = False
            return self

    def _fit_from_dataframe(self, df: "pd.DataFrame") -> "TreatmentModel":
        import pandas as pd

        required = {"PDTRTMNT", "LEDD", "updrs3_score", "updrs3_score_on"}
        if not required.issubset(set(df.columns)):
            logger.warning("Missing columns for treatment model; using defaults.")
            return self

        # Filter: patients on treatment with valid ON/OFF UPDRS3
        sub = df[df["PDTRTMNT"] == 1].copy()
        for col in ["LEDD", "updrs3_score", "updrs3_score_on"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=["LEDD", "updrs3_score", "updrs3_score_on"])
        sub = sub[sub["LEDD"] > 0]

        if len(sub) < 20:
            logger.warning("Too few treatment records (%d); using defaults.", len(sub))
            return self

        # Compute UPDRS reduction (OFF - ON; positive = improvement)
        sub["updrs_reduction"] = sub["updrs3_score"] - sub["updrs3_score_on"]

        # Prepare features: log(1 + LEDD), duration
        sub["log_ledd"] = np.log1p(sub["LEDD"].values)
        if "duration_yrs" in sub.columns:
            sub["dur"] = pd.to_numeric(sub["duration_yrs"], errors="coerce").fillna(0)
        else:
            sub["dur"] = 0.0

        y = sub["updrs_reduction"].values
        X = np.column_stack([
            sub["log_ledd"].values,
            sub["dur"].values,
            np.ones(len(sub)),
        ])

        # Ordinary least squares
        try:
            coefs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning("TreatmentModel OLS failed; using defaults.")
            return self

        self.coefs = {
            "a": round(float(coefs[0]), 4),
            "b": round(float(coefs[1]), 4),
            "c": round(float(coefs[2]), 4),
        }

        # R²
        y_pred = X @ coefs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = round(1 - ss_res / max(ss_tot, 1e-10), 4)

        self.fitted = True
        logger.info(
            "TreatmentModel fitted: coefs=%s, R²=%.4f, n=%d",
            self.coefs, self.r_squared, len(sub),
        )
        return self

    def predict_treatment_effect(
        self,
        ledd: Optional[float],
        duration_years: float = 0.0,
    ) -> float:
        """
        Predict UPDRS3 reduction for given LEDD and disease duration.

        Returns a positive number meaning points of improvement.
        """
        if ledd is None or ledd <= 0:
            return 0.0

        a = self.coefs["a"]
        b = self.coefs["b"]
        c = self.coefs["c"]

        reduction = a * math.log1p(ledd) + b * duration_years + c
        # Clamp to non-negative (treatment can't make things worse in this model)
        return round(max(0.0, reduction), 2)

    def apply_treatment_effect(
        self,
        updrs3_off: float,
        ledd: Optional[float],
        duration_years: float = 0.0,
    ) -> float:
        """Return predicted ON-medication UPDRS3 score."""
        reduction = self.predict_treatment_effect(ledd, duration_years)
        return round(max(0.0, updrs3_off - reduction), 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitted": self.fitted,
            "coefs": self.coefs,
            "r_squared": self.r_squared,
        }

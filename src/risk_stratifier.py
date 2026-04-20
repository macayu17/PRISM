"""
Risk Stratifier — MDS Prodromal Markers + Bootstrap CI.

Maps available PPMI features to MDS prodromal markers, computes a
risk score, and produces bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MDS_MARKERS = {
    "rem":       {"weight": 2.5, "threshold_fn": "binary"},
    "upsit":     {"weight": 2.0, "threshold_fn": "upsit_low"},
    "pigd":      {"weight": 1.5, "threshold_fn": "pigd_present"},
    "gds":       {"weight": 1.2, "threshold_fn": "gds_high"},
    "fampd_bin": {"weight": 1.0, "threshold_fn": "binary"},
}


def _check_marker(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    spec = MDS_MARKERS.get(name)
    if spec is None:
        return None
    fn, w = spec["threshold_fn"], spec["weight"]
    if fn == "binary":
        return w if value >= 1.0 else 0.0
    if fn == "upsit_low":
        return w if value <= 22.0 else 0.0
    if fn == "pigd_present":
        return w if value >= 1.0 else 0.0
    if fn == "gds_high":
        return w if value >= 5.0 else 0.0
    return 0.0


class RiskStratifier:
    """MDS criteria risk stratification with bootstrap CIs."""

    def __init__(self, n_bootstrap: int = 100) -> None:
        self.n_bootstrap = n_bootstrap

    def stratify(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        marker_values: Dict[str, Optional[float]] = {}
        for marker in MDS_MARKERS:
            marker_values[marker] = self._extract(patient_data, marker)

        contributions: Dict[str, Optional[float]] = {}
        available: List[str] = []
        for m, v in marker_values.items():
            s = _check_marker(m, v)
            contributions[m] = s
            if s is not None:
                available.append(m)

        total = sum(v for v in contributions.values() if v is not None)
        max_p = sum(MDS_MARKERS[m]["weight"] for m in available) if available else 1.0
        raw_conf = total / max(max_p, 1e-6)
        category = self._categorize(raw_conf, patient_data)
        ci_lo, ci_hi = self._bootstrap_ci(marker_values, available)

        return {
            "category": category,
            "confidence": round(float(raw_conf), 4),
            "ci_lower": round(float(ci_lo), 4),
            "ci_upper": round(float(ci_hi), 4),
            "marker_contributions": contributions,
            "total_score": round(float(total), 4),
            "max_possible_score": round(float(max_p), 4),
        }

    def _bootstrap_ci(self, marker_values, available):
        if not available:
            return (0.0, 0.0)
        rng = np.random.RandomState(42)
        weights = np.array([MDS_MARKERS[m]["weight"] for m in available])
        base = np.array([_check_marker(m, marker_values[m]) or 0.0 for m in available])
        scores = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(available), size=len(available), replace=True)
            conf = base[idx].sum() / max(weights[idx].sum(), 1e-6)
            conf += rng.normal(0, 0.02)
            scores.append(float(np.clip(conf, 0, 1)))
        return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))

    def _categorize(self, confidence, patient_data):
        sym_total = sum(
            float(patient_data.get(s) or (patient_data.get("motor", {}) or {}).get(s) or 0)
            for s in ("sym_tremor", "sym_rigid", "sym_brady", "sym_posins")
        )
        updrs3 = (patient_data.get("motor", {}) or {}).get("updrs3_score") or patient_data.get("updrs3_score")
        if confidence >= 0.55 or sym_total >= 5 or (updrs3 is not None and float(updrs3) >= 20):
            return "PD"
        if confidence >= 0.30:
            return "Prodromal PD"
        if confidence >= 0.15:
            return "SWEDD"
        return "HC"

    @staticmethod
    def _extract(data: Dict[str, Any], marker: str) -> Optional[float]:
        val = data.get(marker)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        for sec in ("non_motor", "motor", "cognition", "autonomic"):
            sub = data.get(sec)
            if isinstance(sub, dict) and marker in sub and sub[marker] is not None:
                try:
                    return float(sub[marker])
                except (TypeError, ValueError):
                    pass
        return None

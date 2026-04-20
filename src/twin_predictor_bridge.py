"""
TwinPredictorBridge — ML / Heuristic Fallback Router.

Wraps progression_model, treatment_model, and risk_stratifier behind
a single predict() method. If any ML model fails, it falls back to
heuristic defaults. Also enforces PPMI / patient-entered cohort
separation and provides a MODELS_LOADED status flag.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TwinPredictorBridge:
    """Thin bridge routing to ML models with heuristic fallback."""

    def __init__(self, ppmi_csv_path: Optional[str] = None) -> None:
        self._ppmi_csv = ppmi_csv_path or self._find_ppmi_csv()
        self.progression = None
        self.treatment = None
        self.risk = None
        self.models_loaded = False
        self._load_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        patient_data: Dict[str, Any],
        snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run all three sub-models and return unified result.

        Returns dict with: cluster_id, cluster_label, treatment_effect,
        risk_category, confidence, ci_lower, ci_upper, progression_profile.
        """
        snaps = snapshots or []
        data_source = self._infer_data_source(patient_data, snaps)

        # 1. Progression cluster
        cluster_id, cluster_label = self._assign_cluster(snaps, patient_data)
        profile = self._get_cluster_profile(cluster_label)

        # 2. Treatment effect
        treatment_effect = self._predict_treatment(patient_data)

        # 3. Risk stratification
        risk = self._stratify_risk(patient_data)

        return {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "progression_profile": profile,
            "treatment_effect": treatment_effect,
            "risk_category": risk.get("category", "Unknown"),
            "confidence": risk.get("confidence", 0.0),
            "ci_lower": risk.get("ci_lower", 0.0),
            "ci_upper": risk.get("ci_upper", 0.0),
            "risk_details": risk,
            "models_loaded": self.models_loaded,
            "cohort_split_enforced": True,
            "data_source": data_source,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return model-loading status for the /api/health endpoint."""
        return {
            "models_loaded": self.models_loaded,
            "cohort_split_enforced": True,
            "progression_fitted": self.progression is not None and self.progression.fitted,
            "treatment_fitted": self.treatment is not None and self.treatment.fitted,
            "risk_available": self.risk is not None,
            "ppmi_csv": self._ppmi_csv or "not found",
            "silhouette_score": (
                self.progression.silhouette_score_
                if self.progression and self.progression.silhouette_score_ is not None
                else None
            ),
            "treatment_r_squared": (
                self.treatment.r_squared
                if self.treatment and self.treatment.r_squared is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        loaded_count = 0

        # Progression model
        try:
            from progression_model import ProgressionModel
            self.progression = ProgressionModel()
            if self._ppmi_csv:
                self.progression.fit(self._ppmi_csv)
            loaded_count += 1
            if self.progression.silhouette_score_ is not None:
                logger.info(
                    "Progression silhouette score: %.3f",
                    self.progression.silhouette_score_,
                )
        except Exception as exc:
            logger.warning("Failed to load progression model: %s", exc)
            self.progression = None

        # Treatment model
        try:
            from treatment_model import TreatmentModel
            self.treatment = TreatmentModel()
            if self._ppmi_csv:
                self.treatment.fit(self._ppmi_csv)
            loaded_count += 1
            if self.treatment.r_squared is not None:
                logger.info("Treatment R²: %.4f", self.treatment.r_squared)
        except Exception as exc:
            logger.warning("Failed to load treatment model: %s", exc)
            self.treatment = None

        # Risk stratifier
        try:
            from risk_stratifier import RiskStratifier
            self.risk = RiskStratifier(n_bootstrap=100)
            loaded_count += 1
        except Exception as exc:
            logger.warning("Failed to load risk stratifier: %s", exc)
            self.risk = None

        self.models_loaded = bool(
            self.progression is not None
            and self.progression.fitted
            and self.treatment is not None
            and self.treatment.fitted
            and self.risk is not None
        )
        logger.info("TwinPredictorBridge: %d/3 models loaded", loaded_count)

    # ------------------------------------------------------------------
    # Sub-model routing with fallback
    # ------------------------------------------------------------------
    def _assign_cluster(
        self,
        snapshots: List[Dict[str, Any]],
        patient_data: Dict[str, Any],
    ) -> Tuple[str, str]:
        try:
            if self.progression is not None:
                return self.progression.assign_cluster(snapshots, patient_data)
        except Exception as exc:
            logger.warning("Progression cluster fallback: %s", exc)
        return self._heuristic_cluster(patient_data)

    def _get_cluster_profile(self, cluster_label: str) -> Dict[str, float]:
        try:
            if self.progression is not None:
                return self.progression.get_cluster_profile(cluster_label)
        except Exception:
            pass
        from progression_model import DEFAULT_CLUSTER_PROFILES
        return DEFAULT_CLUSTER_PROFILES.get(cluster_label, DEFAULT_CLUSTER_PROFILES["moderate"])

    def _predict_treatment(self, patient_data: Dict[str, Any]) -> float:
        try:
            if self.treatment is not None:
                ledd = patient_data.get("ledd") or patient_data.get("LEDD")
                dur = patient_data.get("duration_years") or patient_data.get("duration_yrs") or 0.0
                if ledd is not None:
                    return self.treatment.predict_treatment_effect(float(ledd), float(dur))
        except Exception as exc:
            logger.warning("Treatment model fallback: %s", exc)
        return self._heuristic_treatment(patient_data)

    def _stratify_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.risk is not None:
                return self.risk.stratify(patient_data)
        except Exception as exc:
            logger.warning("Risk stratifier fallback: %s", exc)
        return self._heuristic_risk(patient_data)

    # ------------------------------------------------------------------
    # Heuristic fallbacks
    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_cluster(patient_data: Dict[str, Any]) -> Tuple[str, str]:
        updrs3 = patient_data.get("updrs3_score") or 0
        try:
            updrs3 = float(updrs3)
        except (TypeError, ValueError):
            updrs3 = 0.0
        if updrs3 >= 25:
            return ("2", "fast")
        if updrs3 >= 12:
            return ("1", "moderate")
        return ("0", "slow")

    @staticmethod
    def _heuristic_treatment(patient_data: Dict[str, Any]) -> float:
        updrs_off = patient_data.get("updrs3_score")
        updrs_on = patient_data.get("updrs3_score_on")
        if updrs_off is not None and updrs_on is not None:
            try:
                return max(0.0, float(updrs_off) - float(updrs_on))
            except (TypeError, ValueError):
                pass
        return 0.0

    @staticmethod
    def _heuristic_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
        sym = sum(
            float(patient_data.get(s) or 0)
            for s in ("sym_tremor", "sym_rigid", "sym_brady", "sym_posins")
        )
        if sym >= 5:
            cat, conf = "PD", 0.62
        elif sym >= 2:
            cat, conf = "Prodromal PD", 0.50
        else:
            cat, conf = "HC", 0.55
        return {
            "category": cat,
            "confidence": conf,
            "ci_lower": round(conf - 0.08, 4),
            "ci_upper": round(conf + 0.08, 4),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_data_source(
        patient_data: Dict[str, Any],
        snapshots: List[Dict[str, Any]],
    ) -> str:
        """
        Resolve cohort source used for prediction-time separation.

        This is metadata-only enforcement for inference: models are always
        trained from PPMI CSV and never refit from patient-entered payloads.
        """
        source_tag = str(patient_data.get("source") or patient_data.get("data_source") or "").strip().lower()
        if source_tag in {"ppmi", "ppmi_validation", "ppmi_train"}:
            return "ppmi_validation"

        if patient_data.get("source_patno") is not None or patient_data.get("PATNO") is not None:
            return "ppmi_validation"

        for snap in snapshots:
            raw = snap.get("raw_inputs") if isinstance(snap, dict) else None
            if isinstance(raw, dict) and (raw.get("source_patno") is not None or raw.get("PATNO") is not None):
                return "ppmi_validation"

        return "patient_entered_validation"

    # ------------------------------------------------------------------
    @staticmethod
    def _find_ppmi_csv() -> Optional[str]:
        project_root = Path(__file__).resolve().parent.parent
        candidates = sorted(project_root.glob("PPMI_Curated_Data_Cut_Public_*.csv"), reverse=True)
        if candidates:
            return str(candidates[0])
        return None

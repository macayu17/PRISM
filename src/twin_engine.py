from __future__ import annotations

import logging
import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from twin_schema import (
    DigitalTwin,
    TwinForecastPoint,
    TwinSimulation,
    TwinSnapshot,
    TwinState,
    TwinStaticProfile,
)
from twin_store import TwinStore

logger = logging.getLogger(__name__)

# Lazy singleton for the predictor bridge
_bridge_instance = None


def _get_bridge():
    global _bridge_instance
    if _bridge_instance is None:
        try:
            from twin_predictor_bridge import TwinPredictorBridge
            _bridge_instance = TwinPredictorBridge()
            status = _bridge_instance.get_status()
            logger.info("TwinPredictorBridge initialized: %s", status)
            if status.get("silhouette_score") is not None:
                print(f"[TWIN] Progression silhouette score: {status['silhouette_score']:.3f}")
            if status.get("treatment_r_squared") is not None:
                print(f"[TWIN] Treatment model R²: {status['treatment_r_squared']:.4f}")
        except Exception as exc:
            logger.warning("Failed to init TwinPredictorBridge: %s", exc)
            _bridge_instance = None
    return _bridge_instance


CLASS_NAMES = ["Healthy Control", "Parkinson's Disease", "SWEDD", "Prodromal PD"]
FORECAST_HORIZONS_MONTHS = [3, 6, 12]


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    try:
        stripped = str(value).strip()
        if stripped == "":
            return None
        parsed = float(stripped)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clamp(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    return max(low, min(high, value))


def _round_optional(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _parse_date(date_string: Optional[str]) -> Optional[datetime]:
    if not date_string:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_string[: len(fmt)], fmt)
        except ValueError:
            continue
    return None


def _scale(value: Optional[float], maximum: float) -> Optional[float]:
    if value is None or maximum <= 0:
        return None
    return _clamp(value / maximum, 0, 1)


def _inverse_scale(value: Optional[float], maximum: float) -> Optional[float]:
    if value is None or maximum <= 0:
        return None
    return _clamp((maximum - value) / maximum, 0, 1)


def _mean_defined(values: List[Optional[float]]) -> float:
    defined = [value for value in values if value is not None]
    if not defined:
        return 0.0
    return sum(defined) / len(defined)


class DigitalTwinEngine:
    def __init__(self, store: Optional[TwinStore] = None, db_path: Optional[str] = None):
        self.store = store or TwinStore(db_path=db_path)
        self.bridge = _get_bridge()

    def list_twins(self) -> List[Dict[str, Any]]:
        return self.store.list_twins()

    def get_twin(self, twin_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_twin(twin_id)

    def create_twin(
        self,
        patient_data: Dict[str, Any],
        patient_label: Optional[str] = None,
        source_patno: Optional[int] = None,
        predictor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        created_at = _iso_now()
        twin_id = f"twin_{uuid4().hex[:12]}"
        profile = self._build_profile(
            twin_id=twin_id,
            patient_data=patient_data,
            patient_label=patient_label,
            source_patno=source_patno,
            created_at=created_at,
        )
        snapshot = self._build_snapshot(patient_data, snapshot_index=0)
        prediction_summary = self._predict_current_state(patient_data, predictor)
        bridge_result = self._bridge_predict(patient_data, [snapshot.to_dict()])
        state = self._build_state(profile, [snapshot], prediction_summary, bridge_result)
        forecast = self._build_forecast(snapshot, state, bridge_result)
        twin = DigitalTwin(
            profile=profile,
            snapshots=[snapshot],
            current_state=state,
            forecast=forecast,
            prediction_summary=prediction_summary,
        )
        return self.store.upsert_twin(twin)

    def add_snapshot(
        self,
        twin_id: str,
        patient_data: Dict[str, Any],
        predictor: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        stored = self.store.get_twin(twin_id)
        if stored is None:
            return None

        snapshots = [self._snapshot_from_dict(item) for item in stored["snapshots"]]
        profile = self._profile_from_dict(stored["profile"])
        snapshots.append(self._build_snapshot(patient_data, snapshot_index=len(snapshots)))
        prediction_summary = self._predict_current_state(patient_data, predictor)
        snap_dicts = [s.to_dict() for s in snapshots]
        bridge_result = self._bridge_predict(patient_data, snap_dicts)
        state = self._build_state(profile, snapshots, prediction_summary, bridge_result)
        forecast = self._build_forecast(snapshots[-1], state, bridge_result)
        twin = DigitalTwin(
            profile=profile,
            snapshots=snapshots,
            current_state=state,
            forecast=forecast,
            prediction_summary=prediction_summary,
        )
        return self.store.upsert_twin(twin)

    def simulate(
        self,
        twin_id: str,
        overrides: Dict[str, Any],
        scenario_name: Optional[str] = None,
        predictor: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        stored = self.store.get_twin(twin_id)
        if stored is None or not stored["snapshots"]:
            return None

        profile = self._profile_from_dict(stored["profile"])
        history = [self._snapshot_from_dict(item) for item in stored["snapshots"]]
        base_snapshot = self._snapshot_from_dict(stored["snapshots"][-1])
        raw_inputs = deepcopy(base_snapshot.raw_inputs)
        raw_inputs.update(overrides)
        simulated_snapshot = self._build_snapshot(
            raw_inputs,
            snapshot_index=len(stored["snapshots"]),
            default_event_id="SIM",
        )
        prediction_summary = self._predict_current_state(raw_inputs, predictor)
        all_snaps = history + [simulated_snapshot]
        snap_dicts = [s.to_dict() for s in all_snaps]
        bridge_result = self._bridge_predict(raw_inputs, snap_dicts)
        state = self._build_state(profile, all_snaps, prediction_summary, bridge_result)
        forecast = self._build_forecast(simulated_snapshot, state, bridge_result)
        simulation = TwinSimulation(
            scenario_name=(scenario_name or "Scenario").strip() or "Scenario",
            overrides=overrides,
            simulated_snapshot=simulated_snapshot,
            state=state,
            forecast=forecast,
        )
        return simulation.to_dict()

    def _build_profile(
        self,
        twin_id: str,
        patient_data: Dict[str, Any],
        patient_label: Optional[str],
        source_patno: Optional[int],
        created_at: str,
    ) -> TwinStaticProfile:
        resolved_label = str(patient_label or patient_data.get("patient_id") or twin_id)
        return TwinStaticProfile(
            twin_id=twin_id,
            patient_label=resolved_label,
            source_patno=source_patno,
            created_at=created_at,
            enrollment_cohort=str(patient_data.get("COHORT") or "Unknown"),
            subgroup=_coerce_text(patient_data.get("subgroup")),
            sex=_safe_float(patient_data.get("SEX")),
            education_years=_safe_float(patient_data.get("EDUCYRS")),
            race=_safe_float(patient_data.get("race")),
            family_pd=_safe_float(patient_data.get("fampd")),
            family_pd_bin=_safe_float(patient_data.get("fampd_bin")),
            bmi=_safe_float(patient_data.get("BMI")),
            age_diag=_safe_float(patient_data.get("agediag")),
            age_onset=_safe_float(patient_data.get("ageonset")),
            dominant_side=_safe_float(patient_data.get("DOMSIDE")),
        )

    def _build_snapshot(
        self,
        patient_data: Dict[str, Any],
        snapshot_index: int,
        default_event_id: str = "MANUAL",
    ) -> TwinSnapshot:
        visit_date = _coerce_text(patient_data.get("visit_date")) or datetime.now().strftime("%Y-%m-%d")
        return TwinSnapshot(
            snapshot_id=f"snap_{uuid4().hex[:12]}",
            event_id=_coerce_text(patient_data.get("EVENT_ID")) or f"{default_event_id}_{snapshot_index + 1}",
            visit_date=visit_date,
            year_index=_safe_float(patient_data.get("YEAR")),
            age_at_visit=_safe_float(patient_data.get("age_at_visit") or patient_data.get("age")),
            duration_years=_safe_float(patient_data.get("duration_yrs")),
            treatment_flag=_safe_float(patient_data.get("PDTRTMNT")),
            ledd=_safe_float(patient_data.get("LEDD")),
            motor={
                "sym_tremor": _safe_float(patient_data.get("sym_tremor")),
                "sym_rigid": _safe_float(patient_data.get("sym_rigid")),
                "sym_brady": _safe_float(patient_data.get("sym_brady")),
                "sym_posins": _safe_float(patient_data.get("sym_posins")),
                "hy": _safe_float(patient_data.get("hy")),
                "hy_on": _safe_float(patient_data.get("hy_on")),
                "pigd": _safe_float(patient_data.get("pigd")),
                "td_pigd": _safe_float(patient_data.get("td_pigd")),
                "updrs1_score": _safe_float(patient_data.get("updrs1_score")),
                "updrs2_score": _safe_float(patient_data.get("updrs2_score")),
                "updrs3_score": _safe_float(patient_data.get("updrs3_score")),
                "updrs3_score_on": _safe_float(patient_data.get("updrs3_score_on")),
                "updrs4_score": _safe_float(patient_data.get("updrs4_score")),
                "updrs_totscore": _safe_float(patient_data.get("updrs_totscore")),
                "updrs_totscore_on": _safe_float(patient_data.get("updrs_totscore_on")),
            },
            cognition={
                "moca": _safe_float(patient_data.get("moca")),
                "bjlot": _safe_float(patient_data.get("bjlot")),
                "clockdraw": _safe_float(patient_data.get("clockdraw")),
                "hvlt_immediaterecall": _safe_float(patient_data.get("hvlt_immediaterecall")),
                "hvlt_retention": _safe_float(patient_data.get("hvlt_retention")),
                "hvlt_discrimination": _safe_float(patient_data.get("hvlt_discrimination")),
                "lexical": _safe_float(patient_data.get("lexical")),
                "lns": _safe_float(patient_data.get("lns")),
            },
            non_motor={
                "ess": _safe_float(patient_data.get("ess")),
                "rem": _safe_float(patient_data.get("rem")),
                "gds": _safe_float(patient_data.get("gds")),
                "stai": _safe_float(patient_data.get("stai")),
                "quip_any": _safe_float(patient_data.get("quip_any")),
                "NP1COG": _safe_float(patient_data.get("NP1COG")),
                "NP1DPRS": _safe_float(patient_data.get("NP1DPRS")),
                "NP1ANXS": _safe_float(patient_data.get("NP1ANXS")),
                "NP1APAT": _safe_float(patient_data.get("NP1APAT")),
                "NP1FATG": _safe_float(patient_data.get("NP1FATG")),
            },
            autonomic={
                "scopa": _safe_float(patient_data.get("scopa")),
                "orthostasis": _safe_float(patient_data.get("orthostasis")),
            },
            biomarkers={
                "abeta": _safe_float(patient_data.get("abeta")),
                "tau": _safe_float(patient_data.get("tau")),
                "ptau": _safe_float(patient_data.get("ptau")),
                "asyn": _safe_float(patient_data.get("asyn")),
                "nfl_serum": _safe_float(patient_data.get("nfl_serum")),
                "NFL_CSF": _safe_float(patient_data.get("NFL_CSF")),
            },
            imaging={
                "MIA_CAUDATE_mean": _safe_float(patient_data.get("MIA_CAUDATE_mean")),
                "MIA_PUTAMEN_mean": _safe_float(patient_data.get("MIA_PUTAMEN_mean")),
                "MIA_STRIATUM_mean": _safe_float(patient_data.get("MIA_STRIATUM_mean")),
            },
            raw_inputs=deepcopy(patient_data),
        )

    def _predict_current_state(
        self,
        patient_data: Dict[str, Any],
        predictor: Optional[Any],
    ) -> Dict[str, Any]:
        if predictor is not None:
            required_fields = [
                "age",
                "SEX",
                "EDUCYRS",
                "BMI",
                "sym_tremor",
                "sym_rigid",
                "sym_brady",
                "sym_posins",
            ]
            if all(_safe_float(patient_data.get(field)) is not None for field in required_fields):
                try:
                    prediction = predictor.predict_patient(patient_data)
                    class_index = int(prediction["ensemble_prediction"])
                    return {
                        "prediction": CLASS_NAMES[class_index],
                        "confidence": round(float(prediction.get("confidence") or 0.0), 3),
                        "probabilities": {
                            CLASS_NAMES[idx]: round(float(prob), 4)
                            for idx, prob in enumerate(prediction.get("ensemble_probabilities", []))
                        },
                        "source": "assessment_model",
                    }
                except Exception:
                    pass

        motor_score = sum(
            value or 0.0
            for value in (
                _safe_float(patient_data.get("sym_tremor")),
                _safe_float(patient_data.get("sym_rigid")),
                _safe_float(patient_data.get("sym_brady")),
                _safe_float(patient_data.get("sym_posins")),
            )
        )
        rem = _safe_float(patient_data.get("rem")) or 0.0
        moca = _safe_float(patient_data.get("moca"))

        if motor_score <= 1.0 and rem == 0 and (moca is None or moca >= 27):
            prediction = "Healthy Control"
            confidence = 0.58
        elif motor_score >= 5.0:
            prediction = "Parkinson's Disease"
            confidence = 0.62
        elif rem == 1 or (moca is not None and moca < 25):
            prediction = "Prodromal PD"
            confidence = 0.56
        else:
            prediction = "SWEDD"
            confidence = 0.52

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {prediction: confidence},
            "source": "heuristic_fallback",
        }

    def _bridge_predict(
        self,
        patient_data: Dict[str, Any],
        snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the TwinPredictorBridge (ML + fallback)."""
        if self.bridge is None:
            self.bridge = _get_bridge()
        if self.bridge is not None:
            try:
                return self.bridge.predict(patient_data, snapshots)
            except Exception as exc:
                logger.warning("Bridge predict failed: %s", exc)
        return {}

    def _build_state(
        self,
        profile: TwinStaticProfile,
        snapshots: List[TwinSnapshot],
        prediction_summary: Dict[str, Any],
        bridge_result: Optional[Dict[str, Any]] = None,
    ) -> TwinState:
        latest = snapshots[-1]
        motor_index = self._motor_burden_index(latest)
        cognitive_index = self._cognitive_burden_index(latest)
        non_motor_index = self._non_motor_burden_index(latest)
        progression_velocity = self._progression_velocity(snapshots)
        treatment_response_proxy = self._treatment_response_proxy(latest)
        br = bridge_result or {}
        evidence = self._build_evidence(
            profile=profile,
            snapshot=latest,
            motor_index=motor_index,
            cognitive_index=cognitive_index,
            non_motor_index=non_motor_index,
            progression_velocity=progression_velocity,
            prediction_summary=prediction_summary,
            bridge_result=br,
        )

        return TwinState(
            current_cohort_estimate=prediction_summary.get("prediction", "Unknown"),
            prediction_source=prediction_summary.get("source", "heuristic"),
            confidence=float(br.get("confidence") or prediction_summary.get("confidence") or 0.0),
            motor_burden_index=_round_optional(motor_index),
            cognitive_burden_index=_round_optional(cognitive_index),
            non_motor_burden_index=_round_optional(non_motor_index),
            progression_velocity=_round_optional(progression_velocity),
            treatment_response_proxy=_round_optional(treatment_response_proxy),
            computed_at=_iso_now(),
            cluster_id=br.get("cluster_id"),
            cluster_label=br.get("cluster_label"),
            treatment_effect=_round_optional(br.get("treatment_effect")),
            ci_lower=_round_optional(br.get("ci_lower")),
            ci_upper=_round_optional(br.get("ci_upper")),
            evidence=evidence,
        )

    def _build_forecast(
        self,
        snapshot: TwinSnapshot,
        state: TwinState,
        bridge_result: Optional[Dict[str, Any]] = None,
    ) -> List[TwinForecastPoint]:
        motor_index = state.motor_burden_index or 0.0
        cognitive_index = state.cognitive_burden_index or 0.0
        non_motor_index = state.non_motor_burden_index or 0.0
        duration_years = snapshot.duration_years or 0.0

        current_updrs3 = snapshot.motor.get("updrs3_score")
        if current_updrs3 is None:
            current_updrs3 = 8 + motor_index * 22

        current_total = snapshot.motor.get("updrs_totscore")
        if current_total is None:
            current_total = current_updrs3 * 1.7 + 8

        current_moca = snapshot.cognition.get("moca")
        if current_moca is None:
            current_moca = 30 - cognitive_index * 8

        current_hy = snapshot.motor.get("hy")
        if current_hy is None:
            current_hy = 1 + motor_index * 2.2

        # Use cluster-weighted profiles from the bridge if available
        br = bridge_result or {}
        profile = br.get("progression_profile", {})
        cluster_label = br.get("cluster_label", "moderate")
        treatment_effect = br.get("treatment_effect", 0.0) or 0.0

        if profile:
            yearly_updrs3_gain = profile.get("updrs3_gain", 3.5)
            yearly_moca_loss = profile.get("moca_loss", 0.7)
            yearly_hy_gain = profile.get("hy_gain", 0.25)
        else:
            velocity = state.progression_velocity or 0.0
            yearly_updrs3_gain = 1.5 + motor_index * 3.5 + duration_years * 0.2 + velocity * 2.0
            yearly_moca_loss = 0.4 + cognitive_index * 0.9 + velocity * 0.2
            yearly_hy_gain = 0.15 + motor_index * 0.35 + velocity * 0.05

        yearly_total_gain = yearly_updrs3_gain * 1.8 + non_motor_index * 1.2

        # Apply treatment effect: reduce UPDRS gains
        treatment_offset = min(treatment_effect, yearly_updrs3_gain * 0.8)

        forecast: List[TwinForecastPoint] = []
        for months in FORECAST_HORIZONS_MONTHS:
            years = months / 12.0
            accel = 1.0 + duration_years * 0.02
            raw_updrs3 = current_updrs3 + (yearly_updrs3_gain * accel - treatment_offset) * years
            predicted_updrs3 = _round_optional(max(0, raw_updrs3))
            predicted_total = _round_optional(max(0, current_total + (yearly_total_gain * accel - treatment_offset * 1.5) * years))
            predicted_moca = _round_optional(_clamp(current_moca - yearly_moca_loss * years, 0, 30))
            predicted_hy = _round_optional(_clamp(current_hy + yearly_hy_gain * years, 0, 5))
            risk_level = self._risk_level(predicted_updrs3, predicted_moca, state.current_cohort_estimate)
            forecast.append(
                TwinForecastPoint(
                    horizon_months=months,
                    predicted_updrs3=predicted_updrs3,
                    predicted_total_updrs=predicted_total,
                    predicted_moca=predicted_moca,
                    predicted_hy=predicted_hy,
                    risk_level=risk_level,
                    uncertainty={
                        "updrs3_pm": _round_optional(1.5 + months * 0.4),
                        "total_updrs_pm": _round_optional(3.0 + months * 0.8),
                        "moca_pm": _round_optional(0.4 + months * 0.08),
                    },
                )
            )

        return forecast

    def _motor_burden_index(self, snapshot: TwinSnapshot) -> float:
        components = [
            _scale(snapshot.motor.get("sym_tremor"), 4),
            _scale(snapshot.motor.get("sym_rigid"), 4),
            _scale(snapshot.motor.get("sym_brady"), 4),
            _scale(snapshot.motor.get("sym_posins"), 4),
            _scale(snapshot.motor.get("updrs3_score"), 60),
            _scale(snapshot.motor.get("updrs_totscore"), 120),
            _scale(snapshot.motor.get("hy"), 5),
        ]
        return _mean_defined(components)

    def _cognitive_burden_index(self, snapshot: TwinSnapshot) -> float:
        components = [
            _inverse_scale(snapshot.cognition.get("moca"), 30),
            _inverse_scale(snapshot.cognition.get("bjlot"), 30),
            _inverse_scale(snapshot.cognition.get("clockdraw"), 4),
            _inverse_scale(snapshot.cognition.get("hvlt_immediaterecall"), 36),
            _inverse_scale(snapshot.cognition.get("lns"), 21),
        ]
        return _mean_defined(components)

    def _non_motor_burden_index(self, snapshot: TwinSnapshot) -> float:
        components = [
            _scale(snapshot.non_motor.get("ess"), 24),
            _scale(snapshot.non_motor.get("gds"), 15),
            _scale((_safe_float(snapshot.non_motor.get("stai")) or 20) - 20, 60),
            _scale(snapshot.non_motor.get("rem"), 1),
            _scale(snapshot.non_motor.get("quip_any"), 1),
            _scale(snapshot.autonomic.get("scopa"), 39),
            _scale(snapshot.autonomic.get("orthostasis"), 1),
            _scale(snapshot.non_motor.get("NP1DPRS"), 4),
            _scale(snapshot.non_motor.get("NP1ANXS"), 4),
            _scale(snapshot.non_motor.get("NP1APAT"), 4),
            _scale(snapshot.non_motor.get("NP1FATG"), 4),
        ]
        return _mean_defined(components)

    def _progression_velocity(self, snapshots: List[TwinSnapshot]) -> Optional[float]:
        if len(snapshots) < 2:
            return None

        first = snapshots[0]
        last = snapshots[-1]
        first_date = _parse_date(first.visit_date)
        last_date = _parse_date(last.visit_date)
        if first_date is None or last_date is None or last_date <= first_date:
            return None

        delta_years = (last_date - first_date).days / 365.25
        if delta_years <= 0:
            return None

        first_composite = (
            self._motor_burden_index(first)
            + self._cognitive_burden_index(first)
            + self._non_motor_burden_index(first)
        ) / 3.0
        last_composite = (
            self._motor_burden_index(last)
            + self._cognitive_burden_index(last)
            + self._non_motor_burden_index(last)
        ) / 3.0
        return (last_composite - first_composite) / delta_years

    def _treatment_response_proxy(self, snapshot: TwinSnapshot) -> Optional[float]:
        updrs_off = snapshot.motor.get("updrs3_score")
        updrs_on = snapshot.motor.get("updrs3_score_on")
        if updrs_off is not None and updrs_on is not None:
            return updrs_off - updrs_on

        hy_off = snapshot.motor.get("hy")
        hy_on = snapshot.motor.get("hy_on")
        if hy_off is not None and hy_on is not None:
            return hy_off - hy_on
        return None

    def _build_evidence(
        self,
        profile: TwinStaticProfile,
        snapshot: TwinSnapshot,
        motor_index: float,
        cognitive_index: float,
        non_motor_index: float,
        progression_velocity: Optional[float],
        prediction_summary: Dict[str, Any],
        bridge_result: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        br = bridge_result or {}
        evidence = [
            f"Current cohort estimate uses {prediction_summary.get('source', 'heuristic')} inference.",
            "Forecasts use cluster-weighted trajectory prediction (v2) and should be treated as decision support only.",
        ]

        data_source = br.get("data_source")
        if data_source:
            evidence.append(f"Cohort split enforced at inference with source: {data_source}.")

        cluster_label = br.get("cluster_label")
        if cluster_label:
            evidence.append(f"Assigned progression cluster: {cluster_label} progressor.")

        treatment_effect = br.get("treatment_effect")
        if treatment_effect and treatment_effect > 0:
            evidence.append(f"Estimated treatment effect (LEDD): {treatment_effect:.1f} UPDRS3 point reduction.")

        ci_lo = br.get("ci_lower")
        ci_hi = br.get("ci_upper")
        if ci_lo is not None and ci_hi is not None:
            evidence.append(f"Risk confidence interval (bootstrap 100): [{ci_lo:.2f}, {ci_hi:.2f}].")

        if snapshot.ledd is not None:
            evidence.append(f"Medication context captured via LEDD {snapshot.ledd:.1f}.")
        if profile.subgroup:
            evidence.append(f"Patient subgroup context: {profile.subgroup}.")
        if motor_index >= 0.55:
            evidence.append("Motor burden is elevated relative to the entered symptom profile.")
        if cognitive_index >= 0.4:
            evidence.append("Cognitive burden suggests closer monitoring of executive and memory measures.")
        if non_motor_index >= 0.45:
            evidence.append("Non-motor burden is material and likely to affect quality of life trajectory.")
        if progression_velocity is not None:
            evidence.append(f"Estimated progression velocity across snapshots: {progression_velocity:.2f} burden units/year.")

        return evidence

    def _risk_level(
        self,
        predicted_updrs3: Optional[float],
        predicted_moca: Optional[float],
        cohort_estimate: str,
    ) -> str:
        if cohort_estimate == "Parkinson's Disease" and (predicted_updrs3 or 0) >= 20:
            return "high"
        if predicted_moca is not None and predicted_moca < 24:
            return "high"
        if (predicted_updrs3 or 0) >= 10:
            return "medium"
        return "low"

    def _profile_from_dict(self, payload: Dict[str, Any]) -> TwinStaticProfile:
        return TwinStaticProfile(**payload)

    def _snapshot_from_dict(self, payload: Dict[str, Any]) -> TwinSnapshot:
        return TwinSnapshot(**payload)

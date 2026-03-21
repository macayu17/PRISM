from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TwinStaticProfile:
    twin_id: str
    patient_label: str
    source_patno: Optional[int]
    created_at: str
    enrollment_cohort: str
    subgroup: Optional[str]
    sex: Optional[float]
    education_years: Optional[float]
    race: Optional[float]
    family_pd: Optional[float]
    family_pd_bin: Optional[float]
    bmi: Optional[float]
    age_diag: Optional[float]
    age_onset: Optional[float]
    dominant_side: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinSnapshot:
    snapshot_id: str
    event_id: str
    visit_date: str
    year_index: Optional[float]
    age_at_visit: Optional[float]
    duration_years: Optional[float]
    treatment_flag: Optional[float]
    ledd: Optional[float]
    motor: Dict[str, Optional[float]]
    cognition: Dict[str, Optional[float]]
    non_motor: Dict[str, Optional[float]]
    autonomic: Dict[str, Optional[float]]
    biomarkers: Dict[str, Optional[float]] = field(default_factory=dict)
    imaging: Dict[str, Optional[float]] = field(default_factory=dict)
    raw_inputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinState:
    current_cohort_estimate: str
    prediction_source: str
    confidence: float
    motor_burden_index: Optional[float]
    cognitive_burden_index: Optional[float]
    non_motor_burden_index: Optional[float]
    progression_velocity: Optional[float]
    treatment_response_proxy: Optional[float]
    computed_at: str
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinForecastPoint:
    horizon_months: int
    predicted_updrs3: Optional[float]
    predicted_total_updrs: Optional[float]
    predicted_moca: Optional[float]
    predicted_hy: Optional[float]
    risk_level: str
    uncertainty: Dict[str, Optional[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinSimulation:
    scenario_name: str
    overrides: Dict[str, Any]
    simulated_snapshot: TwinSnapshot
    state: TwinState
    forecast: List[TwinForecastPoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "overrides": self.overrides,
            "simulated_snapshot": self.simulated_snapshot.to_dict(),
            "state": self.state.to_dict(),
            "forecast": [point.to_dict() for point in self.forecast],
        }


@dataclass
class DigitalTwin:
    profile: TwinStaticProfile
    snapshots: List[TwinSnapshot]
    current_state: TwinState
    forecast: List[TwinForecastPoint]
    prediction_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "current_state": self.current_state.to_dict(),
            "forecast": [point.to_dict() for point in self.forecast],
            "prediction_summary": self.prediction_summary,
        }

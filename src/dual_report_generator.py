"""
Dual report generator for patient-friendly and clinician-facing outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime


CLASS_NAMES = ['Healthy Control', "Parkinson's Disease", 'SWEDD', 'Prodromal PD']


@dataclass
class PatientReportGenerator:
    def generate_report(self, prediction_results: Dict[str, Any], patient_data: Dict[str, Any]) -> str:
        pred_idx = int(prediction_results.get('ensemble_prediction', 0))
        pred_label = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else 'Unknown'
        confidence = float(prediction_results.get('confidence', 0.0)) * 100.0

        lines = [
            'PATIENT SUMMARY REPORT',
            '=' * 40,
            f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '',
            f'Primary assessment: {pred_label}',
            f'Confidence: {confidence:.1f}%',
            '',
            'What this means:',
            '- This is an AI-assisted assessment from questionnaire/clinical inputs.',
            '- It is NOT a final diagnosis.',
            '- Please consult a neurologist for clinical confirmation.',
            '',
            'Next steps:',
            '- Book specialist follow-up',
            '- Track symptom progression',
            '- Bring this report to your clinician',
            '',
            'Disclaimer: Educational/support use only. Not medical advice.'
        ]
        return '\n'.join(lines)


@dataclass
class DoctorReportGenerator:
    def generate_report(self, prediction_results: Dict[str, Any], patient_data: Dict[str, Any], literature_insights: str = '') -> str:
        pred_idx = int(prediction_results.get('ensemble_prediction', 0))
        pred_label = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else 'Unknown'
        confidence = float(prediction_results.get('confidence', 0.0)) * 100.0
        probs = prediction_results.get('ensemble_probabilities', [0, 0, 0, 0])

        lines = [
            'CLINICAL REPORT',
            '=' * 40,
            f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '',
            f'Predicted class: {pred_label}',
            f'Model confidence: {confidence:.2f}%',
            'Probability distribution:',
            f"- HC: {float(probs[0]) * 100:.2f}%" if len(probs) > 0 else '- HC: N/A',
            f"- PD: {float(probs[1]) * 100:.2f}%" if len(probs) > 1 else '- PD: N/A',
            f"- SWEDD: {float(probs[2]) * 100:.2f}%" if len(probs) > 2 else '- SWEDD: N/A',
            f"- PRODROMAL: {float(probs[3]) * 100:.2f}%" if len(probs) > 3 else '- PRODROMAL: N/A',
            '',
            'Patient features submitted:',
        ]

        for k, v in patient_data.items():
            lines.append(f'- {k}: {v}')

        if literature_insights:
            lines.extend(['', 'Literature insights:', literature_insights])

        lines.extend(['', 'Disclaimer: Decision support only; correlate clinically.'])
        return '\n'.join(lines)


class DualReportManager:
    def __init__(self):
        self.patient_generator = PatientReportGenerator()
        self.doctor_generator = DoctorReportGenerator()

    def generate_both_reports(self, prediction_results: Dict[str, Any], patient_data: Dict[str, Any], literature_insights: str = '') -> Dict[str, str]:
        return {
            'patient_report': self.patient_generator.generate_report(prediction_results, patient_data),
            'doctor_report': self.doctor_generator.generate_report(prediction_results, patient_data, literature_insights),
        }

    def save_reports(self, reports: Dict[str, str], report_dir: str, patient_id: str) -> Dict[str, str]:
        import os
        from pathlib import Path
        os.makedirs(report_dir, exist_ok=True)

        safe_patient_id = Path(str(patient_id or "patient")).stem
        safe_patient_id = "".join(
            ch if ch.isalnum() or ch in "._- " else "_" for ch in safe_patient_id
        ).strip(" .") or "patient"

        patient_report_path = os.path.join(report_dir, f'patient_report_{safe_patient_id}.txt')
        doctor_report_path = os.path.join(report_dir, f'clinical_report_{safe_patient_id}.txt')

        with open(patient_report_path, 'w', encoding='utf-8') as f:
            f.write(reports.get('patient_report', ''))

        with open(doctor_report_path, 'w', encoding='utf-8') as f:
            f.write(reports.get('doctor_report', ''))

        return {
            'patient_report_path': patient_report_path,
            'doctor_report_path': doctor_report_path,
        }

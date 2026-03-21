import os
import tempfile
import unittest
from unittest import mock

from src import web_interface
from src.twin_engine import DigitalTwinEngine


class FakeGenerator:
    def __init__(self):
        self.last_patient_data = None
        self.prediction_calls = 0
        self.report_calls = 0

    def predict_patient(self, patient_data):
        self.prediction_calls += 1
        self.last_patient_data = patient_data
        return {
            "ensemble_prediction": 1,
            "ensemble_probabilities": [0.1, 0.7, 0.1, 0.1],
            "traditional_predictions": {"lightgbm": 1},
            "transformer_predictions": {},
            "confidence": 0.7,
        }

    def generate_full_report(self, patient_data, patient_id=None):
        self.report_calls += 1
        return "CLINICAL REPORT\n\n**Summary**\n- Stable"


class WebSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        web_interface.app.config["TESTING"] = True

    def setUp(self):
        self.client = web_interface.app.test_client()

    def test_model_metrics_summary_endpoint(self):
        response = self.client.get("/api/model_metrics_summary")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("models", payload)
        self.assertIsInstance(payload["models"], list)

    def test_documents_endpoint_returns_summaries(self):
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        documents = payload.get("documents", [])
        for document in documents:
            self.assertNotIn("content", document)
            self.assertIn("preview", document)

    def test_invalid_report_download_is_rejected(self):
        response = self.client.get("/api/download_report/..secret.txt")
        self.assertEqual(response.status_code, 400)

    def test_predict_normalizes_family_history_and_derives_binary_code(self):
        fake_generator = FakeGenerator()
        with mock.patch.object(
            web_interface, "_ensure_system_initialized", return_value=fake_generator
        ):
            response = self.client.post(
                "/api/predict",
                json={
                    "age": 62,
                    "SEX": 1,
                    "EDUCYRS": 16,
                    "BMI": 24.5,
                    "sym_tremor": 2,
                    "sym_rigid": 1,
                    "sym_brady": 2,
                    "sym_posins": 0,
                    "fampd": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_generator.last_patient_data["fampd"], 1.0)
        self.assertEqual(fake_generator.last_patient_data["fampd_bin"], 1.0)

    def test_pdf_generation_backfills_prediction_and_report(self):
        fake_generator = FakeGenerator()
        with mock.patch.object(
            web_interface, "_ensure_system_initialized", return_value=fake_generator
        ):
            response = self.client.post(
                "/api/generate_report_pdf",
                json={
                    "patient_data": {
                        "age": 62,
                        "SEX": 1,
                        "EDUCYRS": 16,
                        "BMI": 24.5,
                        "sym_tremor": 2,
                        "sym_rigid": 1,
                        "sym_brady": 2,
                        "sym_posins": 0,
                    },
                    "patient_id": "demo/unsafe",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertGreaterEqual(fake_generator.prediction_calls, 1)
        self.assertGreaterEqual(fake_generator.report_calls, 1)

    def test_create_twin_endpoint_persists_digital_twin(self):
        fake_generator = FakeGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "digital_twins.sqlite3")
            twin_engine = DigitalTwinEngine(db_path=db_path)
            with mock.patch.object(web_interface, "digital_twin_engine", twin_engine):
                with mock.patch.object(
                    web_interface, "_get_twin_predictor", return_value=fake_generator
                ):
                    response = self.client.post(
                        "/api/twins",
                        json={
                            "patient_data": {
                                "patient_id": "Twin-001",
                                "age": 62,
                                "SEX": 1,
                                "EDUCYRS": 16,
                                "BMI": 24.5,
                                "sym_tremor": 2,
                                "sym_rigid": 1,
                                "sym_brady": 2,
                                "sym_posins": 0,
                                "moca": 27,
                            }
                        },
                    )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("twin_id", payload)
        self.assertEqual(
            payload["twin"]["current_state"]["current_cohort_estimate"],
            "Parkinson's Disease",
        )

    def test_twin_snapshot_and_simulation_routes_work(self):
        fake_generator = FakeGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "digital_twins.sqlite3")
            twin_engine = DigitalTwinEngine(db_path=db_path)
            with mock.patch.object(web_interface, "digital_twin_engine", twin_engine):
                with mock.patch.object(
                    web_interface, "_get_twin_predictor", return_value=fake_generator
                ):
                    create_response = self.client.post(
                        "/api/twins",
                        json={
                            "patient_data": {
                                "patient_id": "Twin-002",
                                "age": 59,
                                "SEX": 0,
                                "EDUCYRS": 14,
                                "BMI": 22.8,
                                "sym_tremor": 1,
                                "sym_rigid": 1,
                                "sym_brady": 1,
                                "sym_posins": 0,
                                "moca": 28,
                                "visit_date": "2026-01-01",
                            }
                        },
                    )
                    twin_id = create_response.get_json()["twin_id"]

                    snapshot_response = self.client.post(
                        f"/api/twins/{twin_id}/snapshot",
                        json={
                            "patient_data": {
                                "patient_id": "Twin-002",
                                "age": 60,
                                "SEX": 0,
                                "EDUCYRS": 14,
                                "BMI": 22.8,
                                "sym_tremor": 2,
                                "sym_rigid": 2,
                                "sym_brady": 1,
                                "sym_posins": 1,
                                "moca": 27,
                                "visit_date": "2027-01-01",
                            }
                        },
                    )

                    simulate_response = self.client.post(
                        f"/api/twins/{twin_id}/simulate",
                        json={
                            "scenario_name": "Higher symptom burden",
                            "overrides": {
                                "sym_tremor": 4,
                                "sym_rigid": 4,
                                "moca": 24,
                            },
                        },
                    )

        self.assertEqual(snapshot_response.status_code, 200)
        snapshot_payload = snapshot_response.get_json()
        self.assertEqual(len(snapshot_payload["twin"]["snapshots"]), 2)

        self.assertEqual(simulate_response.status_code, 200)
        simulation_payload = simulate_response.get_json()
        self.assertEqual(
            simulation_payload["simulation"]["scenario_name"], "Higher symptom burden"
        )
        self.assertEqual(len(simulation_payload["simulation"]["forecast"]), 3)


if __name__ == "__main__":
    unittest.main()

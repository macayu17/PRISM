import tempfile
import unittest
from pathlib import Path

from src.train_model_suite import (
    _configure_transformer_runtime,
    _detect_gpu_execution_profile,
    _parse_selected_models,
)
from src.training_runtime import PauseRequested, StopRequested, TrainingRunController


class TrainingRuntimeTests(unittest.TestCase):
    def test_controller_persists_pause_stop_and_checkpoint_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = TrainingRunController(tmpdir, "demo_run")
            controller.initialize(["lightgbm", "biogpt"], {"epochs": 5}, resume=False)
            controller.update_model_state("lightgbm", status="running", current_trial=1)
            controller.save_checkpoint_state("lightgbm", {"phase": "search", "trial": 1})

            checkpoint = controller.load_checkpoint_state("lightgbm")
            self.assertEqual(checkpoint["trial"], 1)

            controller.request_pause()
            with self.assertRaises(PauseRequested):
                controller.raise_if_requested()

            controller.clear_pause()
            controller.request_stop()
            with self.assertRaises(StopRequested):
                controller.raise_if_requested()

            status = controller.status_summary()
            self.assertEqual(status["run_name"], "demo_run")
            self.assertEqual(status["models"]["lightgbm"]["status"], "running")
            self.assertTrue(Path(status["paths"]["state_path"]).exists())

    def test_model_parser_maps_aliases_to_the_six_base_models(self):
        self.assertEqual(_parse_selected_models("all"), ["lightgbm", "xgboost", "svm", "pubmedbert", "biogpt", "clinical_t5"])
        self.assertEqual(_parse_selected_models("lgbm,xgb,svm,pubmed,bio,t5"), ["lightgbm", "xgboost", "svm", "pubmedbert", "biogpt", "clinical_t5"])

    def test_transformer_runtime_requires_cuda_by_default(self):
        with self.assertRaises(RuntimeError):
            _configure_transformer_runtime(["biogpt"], allow_cpu_transformers=False, cuda_available=False)
        self.assertEqual(
            _configure_transformer_runtime(["biogpt"], allow_cpu_transformers=True, cuda_available=False),
            "cpu",
        )
        self.assertEqual(
            _configure_transformer_runtime(["biogpt"], allow_cpu_transformers=False, cuda_available=True),
            "cuda",
        )

    def test_a4000_profile_detection_uses_higher_vram_settings(self):
        profile = _detect_gpu_execution_profile(
            requested_profile="auto",
            cuda_available=True,
            device_name="NVIDIA RTX A4000",
            total_memory_gb=16.0,
        )
        self.assertEqual(profile.name, "rtx-a4000")
        self.assertEqual(profile.train_batch_by_model["pubmedbert"], 16)
        self.assertEqual(profile.train_batch_by_model["biogpt"], 10)
        self.assertEqual(profile.eval_batch_by_model["clinical_t5"], 24)
        self.assertEqual(profile.num_workers, 4)


if __name__ == "__main__":
    unittest.main()

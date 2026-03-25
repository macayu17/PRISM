from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent
DATA_FILES = [
    ROOT / "PPMI_Curated_Data_Cut_Public_20240129.csv",
    ROOT / "PPMI_Curated_Data_Cut_Public_20241211.csv",
    ROOT / "PPMI_Curated_Data_Cut_Public_20250321.csv",
    ROOT / "PPMI_Curated_Data_Cut_Public_20250714.csv",
]
DOCS_DIR = ROOT / "medical_docs"
OUTPUT_TARGETS = [
    ROOT / "models" / "saved",
    ROOT / "evaluation_results",
]


@dataclass
class CheckResult:
    name: str
    status: str
    message: str


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _run_command(command: List[str]) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None


def _recommended_profile(gpu_name: str, vram_gb: float) -> str:
    lower = gpu_name.lower()
    if "a4000" in lower or vram_gb >= 15.0:
        return "rtx-a4000"
    if vram_gb >= 11.0:
        return "high-vram"
    return "compat"


def _record(results: List[CheckResult], name: str, passed: bool, message: str) -> None:
    results.append(CheckResult(name=name, status="PASS" if passed else "FAIL", message=message))


def _record_warn(results: List[CheckResult], name: str, message: str) -> None:
    results.append(CheckResult(name=name, status="WARN", message=message))


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for RTX A4000 training.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument(
        "--min-disk-gb",
        type=float,
        default=20.0,
        help="Minimum recommended free disk space on the project drive.",
    )
    parser.add_argument(
        "--allow-no-rag-docs",
        action="store_true",
        help="Do not fail if medical_docs is empty or missing.",
    )
    args = parser.parse_args()

    results: List[CheckResult] = []
    summary = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "root": str(ROOT),
    }

    python_ok = sys.version_info >= (3, 10)
    _record(
        results,
        "python",
        python_ok,
        f"Python {platform.python_version()} detected" if python_ok else "Python 3.10+ is recommended for training",
    )

    available_data = [path.name for path in DATA_FILES if path.exists()]
    missing_data = [path.name for path in DATA_FILES if not path.exists()]
    summary["available_dataset_files"] = available_data
    if len(available_data) == len(DATA_FILES):
        _record(results, "dataset", True, "All expected PPMI CSV files are present")
    elif available_data:
        _record_warn(
            results,
            "dataset",
            f"Found {len(available_data)}/{len(DATA_FILES)} expected dataset files; missing: {', '.join(missing_data)}",
        )
    else:
        _record(
            results,
            "dataset",
            False,
            "No expected PPMI CSV files were found in the project root",
        )

    docs_ok = DOCS_DIR.exists() and any(DOCS_DIR.iterdir())
    if docs_ok:
        _record(results, "medical_docs", True, f"RAG docs found in {DOCS_DIR.name}")
    elif args.allow_no_rag_docs:
        _record_warn(results, "medical_docs", "medical_docs is missing or empty; training must run with --no-rag")
    else:
        _record(results, "medical_docs", False, "medical_docs is missing or empty; default training uses RAG contexts")

    disk_free_gb = _free_gb(ROOT)
    summary["free_disk_gb"] = round(disk_free_gb, 2)
    _record(
        results,
        "disk",
        disk_free_gb >= args.min_disk_gb,
        f"{disk_free_gb:.1f} GB free on project drive",
    )

    writable_failures = []
    for target in OUTPUT_TARGETS:
        check_path = target if target.exists() else target.parent
        if not os.access(check_path, os.W_OK):
            writable_failures.append(str(check_path))
    _record(
        results,
        "write_access",
        not writable_failures,
        "Output paths are writable" if not writable_failures else f"Not writable: {', '.join(writable_failures)}",
    )

    smi_result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if smi_result and smi_result.returncode == 0:
        smi_line = smi_result.stdout.strip().splitlines()[0] if smi_result.stdout.strip() else ""
        summary["nvidia_smi"] = smi_line
        _record(results, "nvidia_smi", True, smi_line or "nvidia-smi responded")
    else:
        _record(results, "nvidia_smi", False, "nvidia-smi not available or NVIDIA driver not ready")

    torch = None
    try:
        import torch as torch_module

        torch = torch_module
        summary["torch"] = torch.__version__
        _record(results, "torch", True, f"torch {torch.__version__}")
    except Exception as exc:
        _record(results, "torch", False, f"PyTorch import failed: {exc}")

    try:
        import transformers

        _record(results, "transformers", True, f"transformers {transformers.__version__}")
        summary["transformers"] = transformers.__version__
    except Exception as exc:
        _record(results, "transformers", False, f"Transformers import failed: {exc}")

    try:
        import sacremoses

        sacremoses_version = getattr(sacremoses, "__version__", "installed")
        _record(results, "sacremoses", True, f"sacremoses {sacremoses_version}")
        summary["sacremoses"] = sacremoses_version
    except Exception as exc:
        _record(results, "sacremoses", False, f"BioGPT tokenizer dependency missing: {exc}")

    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        summary["cuda_available"] = cuda_available
        if not cuda_available:
            _record(results, "cuda", False, "torch.cuda.is_available() is False")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_version = torch.version.cuda or "unknown"
            profile = _recommended_profile(gpu_name, total_vram_gb)
            summary["gpu_name"] = gpu_name
            summary["cuda_version"] = cuda_version
            summary["gpu_capability"] = f"{capability[0]}.{capability[1]}"
            summary["gpu_vram_gb"] = round(total_vram_gb, 2)
            summary["recommended_profile"] = profile
            _record(
                results,
                "gpu",
                profile == "rtx-a4000",
                f"{gpu_name} | {total_vram_gb:.1f} GB VRAM | CUDA {cuda_version} | recommended profile: {profile}",
            )

    passed = all(result.status != "FAIL" for result in results)
    report = {
        "passed": passed,
        "summary": summary,
        "results": [asdict(result) for result in results],
        "recommended_train_command": (
            "python src/train_model_suite.py train --run-name a4000_full "
            "--gpu-profile rtx-a4000 --epochs 30 --patience 8 "
            "--traditional-trials 6 --transformer-trials 6 "
            "--transformer-loss focal --focal-gamma 1.5"
        ),
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 72)
        print("PRISM RTX A4000 Preflight")
        print("=" * 72)
        for result in results:
            print(f"[{result.status}] {result.name:<12} {result.message}")
        print("-" * 72)
        print(f"Project root : {ROOT}")
        print(f"Free disk GB : {disk_free_gb:.1f}")
        if "torch" in summary:
            print(f"PyTorch      : {summary['torch']}")
        if "gpu_name" in summary:
            print(f"GPU          : {summary['gpu_name']}")
            print(f"CUDA         : {summary['cuda_version']}")
            print(f"Profile      : {summary['recommended_profile']}")
        print("-" * 72)
        print("Recommended command:")
        print(report["recommended_train_command"])

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

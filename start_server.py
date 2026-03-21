"""
Start the Parkinson's Disease Assessment Web Application.

This launcher:
- verifies required dependencies without importing heavy packages eagerly
- imports the Flask app via package-style imports from `src`
- optionally skips model initialization with `--skip-init` / `-s`
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_project_on_path() -> None:
    """Add the project root to sys.path so `src.*` imports work reliably."""
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def has_module(module_name: str) -> bool:
    """Check whether a module is available without importing it."""
    return importlib.util.find_spec(module_name) is not None


def check_dependencies() -> None:
    """Validate the minimum runtime dependencies needed to start the server."""
    print("Checking dependencies...")

    required_modules = {
        "flask": "Flask",
        "flask_cors": "Flask-CORS",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "joblib": "Joblib",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "torch": "PyTorch",
        "transformers": "Transformers",
        "PyPDF2": "PyPDF2",
        "reportlab": "ReportLab",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
    }

    missing: list[str] = []

    for module_name, label in required_modules.items():
        if has_module(module_name):
            print(f"[OK] {label} installed")
        else:
            print(f"[ERROR] {label} not installed")
            missing.append(module_name)

    if missing:
        print()
        print("Missing required dependencies.")
        print("Install them with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print()
    print("All dependencies found!")
    print()


def print_header() -> None:
    print("=" * 70)
    print("Parkinson's Disease Assessment Portal")
    print("=" * 70)
    print()


def print_server_info() -> None:
    print("Starting Web Server...")
    print("=" * 70)
    print()
    print("The application will be available at:")
    print("  -> http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()


def main() -> None:
    ensure_project_on_path()

    print_header()
    check_dependencies()
    print_server_info()

    from src.web_interface import app, initialize_system

    skip_init = "--skip-init" in sys.argv or "-s" in sys.argv

    if skip_init:
        print("[INFO] Skipping initial model loading.")
        print("[INFO] Models will load automatically on the first prediction.")
        print()
    else:
        print("Initializing AI models...")
        print("[INFO] This may take 1-2 minutes the first time...")
        try:
            if initialize_system():
                print("[OK] System initialized successfully")
                print()
            else:
                print("[WARNING] System initialization had some issues")
                print(
                    "[WARNING] The app will still run and load models on first prediction"
                )
                print()
        except KeyboardInterrupt:
            print("\n[INFO] Initialization interrupted by user")
            print(
                "[INFO] Starting server anyway (models will load on first prediction)"
            )
            print()
        except Exception as exc:
            print(f"[ERROR] Initialization error: {exc}")
            print(
                "[INFO] Starting server anyway (models will load on first prediction)"
            )
            print()

    print("=" * 70)
    print("SERVER STARTING...")
    print("=" * 70)
    print()

    try:
        app.run(
            debug=False,
            host="0.0.0.0",
            port=5000,
            use_reloader=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Server stopped by user")
        print("=" * 70)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()

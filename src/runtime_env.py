"""Runtime environment guards for local scripts and app startup."""

from __future__ import annotations

import os
from pathlib import Path


def configure_loky_cpu_count() -> bool:
    """Set a stable Joblib/Loky CPU count on Windows when one is not provided."""

    if os.environ.get("LOKY_MAX_CPU_COUNT"):
        return False

    if os.name != "nt":
        return False

    logical_count = os.cpu_count() or 1
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, logical_count - 1))
    return True


def sanitize_ssl_keylogfile() -> bool:
    """Remove SSLKEYLOGFILE when it points to a path Python cannot write.

    Some ML dependencies import networking stacks during module import. Python's
    SSL module honors SSLKEYLOGFILE while building default contexts, so a stale
    or protected keylog path can crash the whole app before it starts.
    """

    keylog_path = os.environ.get("SSLKEYLOGFILE")
    if not keylog_path:
        return True

    path = Path(keylog_path)
    try:
        if path.exists() and path.is_dir():
            raise IsADirectoryError(str(path))

        parent = path.parent
        if parent and not parent.exists():
            raise FileNotFoundError(str(parent))

        with path.open("a", encoding="utf-8"):
            pass
    except OSError:
        os.environ.pop("SSLKEYLOGFILE", None)
        return False

    return True


def prepare_runtime_environment() -> dict[str, bool]:
    """Apply safe defaults needed before importing heavyweight ML libraries."""

    return {
        "ssl_keylogfile_ok": sanitize_ssl_keylogfile(),
        "loky_cpu_count_set": configure_loky_cpu_count(),
    }

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class RuntimeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.original_ssl_keylogfile = os.environ.get("SSLKEYLOGFILE")
        self.original_loky_cpu_count = os.environ.get("LOKY_MAX_CPU_COUNT")

    def tearDown(self):
        if self.original_ssl_keylogfile is None:
            os.environ.pop("SSLKEYLOGFILE", None)
        else:
            os.environ["SSLKEYLOGFILE"] = self.original_ssl_keylogfile

        if self.original_loky_cpu_count is None:
            os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        else:
            os.environ["LOKY_MAX_CPU_COUNT"] = self.original_loky_cpu_count

    def test_bad_ssl_keylogfile_path_is_removed_before_heavy_imports(self):
        from src.runtime_env import sanitize_ssl_keylogfile

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["SSLKEYLOGFILE"] = temp_dir

            self.assertFalse(sanitize_ssl_keylogfile())
            self.assertNotIn("SSLKEYLOGFILE", os.environ)

    def test_writable_ssl_keylogfile_path_is_preserved(self):
        from src.runtime_env import sanitize_ssl_keylogfile

        with tempfile.TemporaryDirectory() as temp_dir:
            keylog_path = Path(temp_dir) / "sslkeys.log"
            os.environ["SSLKEYLOGFILE"] = str(keylog_path)

            self.assertTrue(sanitize_ssl_keylogfile())
            self.assertEqual(os.environ["SSLKEYLOGFILE"], str(keylog_path))

    def test_windows_loky_cpu_count_default_is_set_when_missing(self):
        from src.runtime_env import configure_loky_cpu_count

        os.environ.pop("LOKY_MAX_CPU_COUNT", None)
        with mock.patch("src.runtime_env.os.name", "nt"):
            with mock.patch("src.runtime_env.os.cpu_count", return_value=8):
                self.assertTrue(configure_loky_cpu_count())

        self.assertEqual(os.environ["LOKY_MAX_CPU_COUNT"], "7")

    def test_existing_loky_cpu_count_is_preserved(self):
        from src.runtime_env import configure_loky_cpu_count

        os.environ["LOKY_MAX_CPU_COUNT"] = "3"
        with mock.patch("src.runtime_env.os.name", "nt"):
            self.assertFalse(configure_loky_cpu_count())

        self.assertEqual(os.environ["LOKY_MAX_CPU_COUNT"], "3")


if __name__ == "__main__":
    unittest.main()

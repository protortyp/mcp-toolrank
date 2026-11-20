from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class BundleBuilderTests(unittest.TestCase):
    def test_corrupt_pinned_wheel_does_not_replace_existing_bundle(self):
        builder = Path(__file__).resolve().parents[2] / "bundle/build.py"
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            wheel = directory / "toolrank_cpu-0.3.0-py3-none-any.whl"
            wheel.write_bytes(b"corrupt")
            existing = directory / "bundle.mcpb"
            existing.write_bytes(b"keep this output")
            result = subprocess.run([sys.executable, str(builder), str(existing), "--gateway-wheel", str(wheel)],
                                    capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Gateway wheel does not match the bundle lockfile", result.stderr)
            self.assertEqual(existing.read_bytes(), b"keep this output")

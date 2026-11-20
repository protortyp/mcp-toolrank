from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from toolrank_cpu import cli


class PackageTests(unittest.TestCase):
    def test_checksums_reject_corrupt_bytes(self):
        with TemporaryDirectory() as temp:
            fingerprint = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
            cached = Path(temp) / "data"
            cached.write_bytes(b"hello")
            self.assertTrue(cli.matches(cached, fingerprint))
            cached.write_bytes(b"changed")
            self.assertFalse(cli.matches(cached, fingerprint))

    def test_offline_does_not_attempt_download(self):
        with TemporaryDirectory() as temp, patch.dict("os.environ", {
            "XDG_CACHE_HOME": temp, "TOOLRANK_OFFLINE": "1",
        }), patch.object(cli, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "verified cached file"):
                cli.checkpoint()
            download.assert_not_called()

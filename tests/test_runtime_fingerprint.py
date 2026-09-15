"""Build-identity regression only: never opens a model or initializes Metal."""
from pathlib import Path
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
fingerprint = runpy.run_path(str(ROOT / "tools/runtime_fingerprint.py"))["fingerprint"]


class RuntimeFingerprintTest(unittest.TestCase):
    def test_generated_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("qw3.c", "qw3.h", "qw3_metal.m", "qw3_metal.h",
                         "tools/runtime_fingerprint.py"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            header = root / "identity.h"
            command = [sys.executable, str(root / "tools/runtime_fingerprint.py"),
                       str(header), "cc", "-O3", "-fobjc-arc", "-lm"]
            subprocess.run(command, check=True)
            original = header.read_bytes()
            # An old timestamp makes accidental rewrites observable even on
            # filesystems with coarse timestamp resolution.
            os.utime(header, ns=(1000000000, 1000000000))
            subprocess.run(command, check=True)
            self.assertEqual(header.stat().st_mtime_ns, 1000000000)
            self.assertEqual(header.read_bytes(), original)
            subprocess.run(command + ["-DCHANGED=1"], check=True)
            self.assertNotEqual(header.read_bytes(), original)
            subprocess.run(command, check=True)
            self.assertEqual(header.read_bytes(), original)
            source = root / "qw3_metal.m"
            before = source.stat()
            source.write_bytes(source.read_bytes() + b"\n/* changed */\n")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            subprocess.run(command, check=True)
            self.assertNotEqual(header.read_bytes(), original)

    def test_content_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("qw3.c", "qw3.h", "qw3_metal.m", "qw3_metal.h",
                         "tools/runtime_fingerprint.py"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            config = ["cc", "-O3", "-fobjc-arc", "compiler version"]
            original = fingerprint(root, config)
            self.assertEqual(original, fingerprint(root, config))
            self.assertNotEqual(original, fingerprint(root, config + ["-DNEW=1"]))
            for name in ("qw3.c", "qw3.h", "qw3_metal.m", "qw3_metal.h"):
                path = root / name
                before = path.read_bytes()
                stat = path.stat()
                path.write_bytes(before + b"\n/* changed runtime */\n")
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertNotEqual(original, fingerprint(root, config), name)
                path.write_bytes(before)
                self.assertEqual(original, fingerprint(root, config))
            os.utime(root / "qw3.c", None)
            self.assertEqual(original, fingerprint(root, config))


if __name__ == "__main__":
    unittest.main()

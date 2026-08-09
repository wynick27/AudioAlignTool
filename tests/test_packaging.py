from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PortablePackagingTests(unittest.TestCase):
    def test_manual_and_ci_build_share_one_entrypoint(self) -> None:
        script = ROOT / "build-portable.ps1"
        workflow = ROOT / ".github" / "workflows" / "windows-portable.yml"

        self.assertTrue(script.is_file())
        workflow_text = workflow.read_text(encoding="utf-8")
        self.assertIn(".\\build-portable.ps1", workflow_text)
        self.assertNotIn("Inno", workflow_text)
        self.assertNotIn("choco", workflow_text)
        self.assertFalse((ROOT / "installer.iss").exists())

    def test_portable_build_uses_python_314_and_onedir_spec(self) -> None:
        script_text = (ROOT / "build-portable.ps1").read_text(encoding="utf-8")
        spec_text = (ROOT / "AudioAlignTool.spec").read_text(encoding="utf-8")
        workflow_text = (ROOT / ".github" / "workflows" / "windows-portable.yml").read_text(encoding="utf-8")

        self.assertIn("sys.version_info[:2] == (3, 14)", script_text)
        self.assertIn('python-version: "3.14"', workflow_text)
        self.assertIn("AudioAlignTool.spec", script_text)
        self.assertIn("Compress-Archive", script_text)
        self.assertIn("COLLECT(", spec_text)
        self.assertRegex(spec_text, re.compile(r"upx\s*=\s*False"))

    def test_portable_output_is_ignored(self) -> None:
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("artifacts/", ignore_text.splitlines())


if __name__ == "__main__":
    unittest.main()

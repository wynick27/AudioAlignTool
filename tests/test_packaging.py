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

    def test_portable_build_uses_python_313_and_onedir_spec(self) -> None:
        script_text = (ROOT / "build-portable.ps1").read_text(encoding="utf-8")
        spec_text = (ROOT / "AudioAlignTool.spec").read_text(encoding="utf-8")
        workflow_text = (ROOT / ".github" / "workflows" / "windows-portable.yml").read_text(encoding="utf-8")

        self.assertIn("sys.version_info[:2] == (3, 13)", script_text)
        self.assertIn('python-version: "3.13"', workflow_text)
        self.assertIn("AudioAlignTool.spec", script_text)
        self.assertIn("Compress-Archive", script_text)
        self.assertIn("COLLECT(", spec_text)
        self.assertIn('"runtime-packages\\runtime-index.json"', script_text)
        self.assertIn("Copy-Item -LiteralPath $sourceRuntimeIndex", script_text)
        self.assertIn('"pip"', spec_text)
        self.assertRegex(spec_text, re.compile(r"upx\s*=\s*False"))

    def test_runtime_catalog_is_local_and_not_a_github_release_index(self) -> None:
        index_text = (ROOT / "runtime-packages" / "runtime-index.json").read_text(encoding="utf-8")
        runtime_source = (ROOT / "src" / "audioalign" / "core" / "runtime_addons.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"packages"', index_text)
        self.assertNotIn("github.com", index_text.lower())
        self.assertNotIn("urlopen", runtime_source)
        self.assertFalse((ROOT / ".github" / "workflows" / "runtime-addons.yml").exists())

    def test_ci_validates_dependencies_and_skips_tests(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "windows-portable.yml").read_text(encoding="utf-8")

        self.assertIn("requirements-lock.txt", workflow_text)
        self.assertIn("Failed to install locked build dependencies.", workflow_text)
        self.assertIn("import numpy, PySide6, av, faster_whisper, markdown_it", workflow_text)
        self.assertIn("markdown_it", workflow_text)
        self.assertIn("QtWebEngineWidgets", workflow_text)
        self.assertNotIn("- name: Run tests", workflow_text)
        self.assertIn("-SkipInstall -SkipTests -Clean", workflow_text)

    def test_default_portable_excludes_optional_qwen_stack_and_cleans_bundle(self) -> None:
        script_text = (ROOT / "build-portable.ps1").read_text(encoding="utf-8")
        spec_text = (ROOT / "AudioAlignTool.spec").read_text(encoding="utf-8")
        lock_text = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
        qwen_lock_text = (ROOT / "requirements-qwen-lock.txt").read_text(encoding="utf-8")

        self.assertIn("[switch]$Standard", script_text)
        self.assertNotIn("IncludeQwen", script_text)
        self.assertIn("download.pytorch.org/whl/cpu", script_text)
        self.assertIn("Remove-BuildPath $bundleDir", script_text)
        self.assertIn('os.environ.get("AAT_INCLUDE_QWEN"', spec_text)
        self.assertIn('"torch"', spec_text)
        self.assertNotIn('collect_all("onnxruntime")', spec_text)
        self.assertNotIn('collect_all("pyqtgraph")', spec_text)
        self.assertNotIn("qwen-asr", lock_text)
        self.assertIn("qwen-asr==0.0.6", qwen_lock_text)

    def test_ci_builds_portable_and_cpu_qwen_standard_packages(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "windows-portable.yml").read_text(encoding="utf-8")

        self.assertIn("flavor: portable", workflow_text)
        self.assertIn("flavor: standard", workflow_text)
        self.assertIn('build_args: "-Standard"', workflow_text)
        self.assertIn("download.pytorch.org/whl/cpu", workflow_text)
        self.assertNotIn("-IncludeQwen", workflow_text)
        self.assertIn("AudioAlignTool-*-Windows-x64-*.zip", workflow_text)
        self.assertEqual(1, workflow_text.count("concurrency:"))
        self.assertLess(workflow_text.index("concurrency:"), workflow_text.index("jobs:"))

    def test_tag_build_publishes_the_portable_archive(self) -> None:
        workflow_text = (ROOT / ".github" / "workflows" / "windows-portable.yml").read_text(encoding="utf-8")

        self.assertIn("name: Publish GitHub Release", workflow_text)
        self.assertIn("contents: write", workflow_text)
        self.assertIn("actions/download-artifact@v4", workflow_text)
        self.assertIn('gh release create "$tag"', workflow_text)
        self.assertIn('gh release upload "$tag"', workflow_text)

    def test_portable_output_is_ignored(self) -> None:
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("artifacts/", ignore_text.splitlines())


if __name__ == "__main__":
    unittest.main()

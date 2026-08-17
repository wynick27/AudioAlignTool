from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from audioalign.__main__ import _runtime_probe
from audioalign.core.asr import RuntimeStatus
from audioalign.core.paths import ApplicationPaths
from audioalign.core.runtime_addons import (
    RuntimeComponent, activate_component, cleanup_inactive_ai_components,
    _install_pypi_packages,
    component_manifest,
    install_runtime_component,
    load_active_runtimes,
    load_runtime_index,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeAddonTests(unittest.TestCase):
    def test_runtime_probe_uses_codepage_independent_ascii_json(self) -> None:
        status = RuntimeStatus(
            False, False, False, False,
            "模型 Qwen3-ASR-0.6B 尚未下载",
            "qwen3-asr",
        )
        output = io.StringIO()
        with patch("audioalign.core.asr.runtime_status", return_value=status):
            with redirect_stdout(output):
                result = _runtime_probe(["qwen3-asr", "Qwen3-ASR-0.6B", "models"])

        self.assertEqual(0, result)
        encoded = output.getvalue().strip()
        encoded.encode("ascii")
        self.assertEqual(status.message, json.loads(encoded)["message"])

    def test_pip_installer_passes_environment_with_subprocess_env(self) -> None:
        component = RuntimeComponent(
            id="test-win-x64",
            group="test",
            variant="cpu",
            display_name="Test",
            packages=("example-package==1.0",),
        )
        captured: dict[str, object] = {}

        class FakeProcess:
            stdout = ["installed\n"]

            @staticmethod
            def wait() -> int:
                return 0

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as folder:
            with patch("audioalign.core.runtime_addons.subprocess.Popen", fake_popen):
                _install_pypi_packages(component, Path(folder), None)

        self.assertNotIn("environment", captured)
        self.assertIn("--progress-bar", captured["command"])
        self.assertIn("raw", captured["command"])
        environment = captured["env"]
        self.assertIsInstance(environment, dict)
        self.assertEqual("1", environment["PYTHONUTF8"])
        self.assertEqual("1", environment["PIP_NO_INPUT"])

    def test_pip_raw_download_progress_is_determinate_and_monotonic(self) -> None:
        component = RuntimeComponent(
            id="test-win-x64",
            group="test",
            variant="cpu",
            display_name="Test",
            packages=("example-package==1.0",),
        )
        events: list[tuple[float, str]] = []

        class FakeProcess:
            stdout = [
                "Collecting example-package\n",
                "Progress 0 of 104857600\n",
                "Progress 52428800 of 104857600\n",
                "Progress 104857600 of 104857600\n",
                "Progress 0 of 209715200\n",
                "Progress 209715200 of 209715200\n",
                "Installing collected packages: example-package\n",
                "Successfully installed example-package-1.0\n",
            ]

            @staticmethod
            def wait() -> int:
                return 0

        with tempfile.TemporaryDirectory() as folder:
            with patch(
                "audioalign.core.runtime_addons.subprocess.Popen",
                return_value=FakeProcess(),
            ):
                _install_pypi_packages(
                    component, Path(folder),
                    lambda value, message: events.append((value, message)),
                )

        fractions = [value for value, _message in events]
        self.assertTrue(all(value >= 0 for value in fractions))
        self.assertEqual(fractions, sorted(fractions))
        self.assertGreaterEqual(fractions[-1], 0.99)
        self.assertTrue(any("50.0/100.0 MB" in message for _value, message in events))

    def test_no_deps_packages_are_installed_in_a_separate_pip_phase(self) -> None:
        component = RuntimeComponent(
            id="test-win-x64",
            group="test",
            variant="cpu",
            display_name="Test",
            packages=("dependency==2.11",),
            no_deps_packages=("overridden-package==1.0",),
        )
        commands: list[list[str]] = []

        class FakeProcess:
            stdout = ["Successfully installed\n"]

            @staticmethod
            def wait() -> int:
                return 0

        def fake_popen(command, **_kwargs):
            commands.append(list(command))
            return FakeProcess()

        with tempfile.TemporaryDirectory() as folder:
            with patch("audioalign.core.runtime_addons.subprocess.Popen", fake_popen):
                _install_pypi_packages(component, Path(folder), None)

        self.assertEqual(2, len(commands))
        self.assertNotIn("--no-deps", commands[0])
        self.assertIn("--no-deps", commands[1])
        self.assertIn("--ignore-requires-python", commands[1])
        self.assertIn("overridden-package==1.0", commands[1])

    def test_bundled_index_uses_package_sources_and_no_github_assets(self) -> None:
        index_path = ROOT / "runtime-packages" / "runtime-index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["components"])
        for component in payload["components"]:
            self.assertTrue(component["packages"])
            self.assertNotIn("url", component)
            self.assertNotIn("archive", component)
            self.assertNotIn("github.com", json.dumps(component).lower())

    def test_bundled_ai_runtime_uses_one_torch_211_stack(self) -> None:
        index_path = ROOT / "runtime-packages" / "runtime-index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        components = payload["components"]

        self.assertEqual(
            {"ai-qwen-cpu-win-x64", "ai-full-cpu-win-x64", "ai-full-cuda-win-x64"},
            {component["id"] for component in components},
        )
        self.assertEqual({"ai"}, {component["group"] for component in components})
        for component in components:
            packages = set(component["packages"])
            self.assertIn("torch==2.11.0", packages)
            self.assertNotIn("torch==2.8.0", packages)
            if component["id"].startswith("ai-full-"):
                self.assertIn("torchaudio==2.11.0", packages)
                self.assertIn("torchvision==0.26.0", packages)
                self.assertIn("torchcodec==0.11.1", packages)
                self.assertEqual(["whisperx==3.8.6"], component["no_deps_packages"])

        source = (ROOT / "src" / "audioalign" / "core" / "runtime_addons.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("urlopen", source)
        self.assertNotIn("https://github", source.lower())

    def test_index_is_loaded_only_from_application_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = ApplicationPaths(Path(folder))
            paths.runtime_packages.mkdir(parents=True)
            paths.runtime_index.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "components": [
                            {
                                "id": "test-win-x64",
                                "group": "test",
                                "variant": "cpu",
                                "display_name": "Test",
                                "packages": ["example-package==1.0"],
                                "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                                "platform": "win_amd64",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            components = load_runtime_index(paths)
            self.assertEqual(components[0].packages, ("example-package==1.0",))
            self.assertEqual(components[0].index_url, "https://pypi.org/simple")

    def test_install_commits_complete_pypi_layer_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = ApplicationPaths(Path(folder))
            paths.ensure()
            paths.runtime_index.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "components": [
                            {
                                "id": "test-win-x64",
                                "group": "test",
                                "variant": "cpu",
                                "display_name": "Test",
                                "packages": ["example-package==1.0"],
                                "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                                "platform": "win_amd64",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            component = load_runtime_index(paths)[0]

            def fake_install(_component, target, progress) -> None:
                (target / "example_package.py").write_text("READY = True\n", encoding="utf-8")
                if progress:
                    progress(1.0, "done")

            with patch("audioalign.core.runtime_addons._install_pypi_packages", fake_install):
                target = install_runtime_component(component, paths)

            self.assertTrue((target / "site-packages" / "example_package.py").is_file())
            self.assertEqual(component_manifest(component.id, paths)["source"], "pypi")
            self.assertEqual(load_active_runtimes(paths), {"test": component.id})

    def test_reinstall_of_complete_component_only_reactivates_it(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = ApplicationPaths(Path(folder))
            paths.ensure()
            paths.runtime_index.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "components": [
                            {
                                "id": "test-cpu-win-x64",
                                "group": "test",
                                "variant": "cpu",
                                "display_name": "Test CPU",
                                "packages": ["example-package==1.0"],
                                "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                                "platform": "win_amd64",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            component = load_runtime_index(paths)[0]
            install_calls = 0

            def fake_install(_component, target, _progress) -> None:
                nonlocal install_calls
                install_calls += 1
                (target / "example_package.py").write_text("READY = True\n", encoding="utf-8")

            with patch("audioalign.core.runtime_addons._install_pypi_packages", fake_install):
                first = install_runtime_component(component, paths)
                second = install_runtime_component(component, paths)

            self.assertEqual(first, second)
            self.assertEqual(1, install_calls)
            self.assertEqual(load_active_runtimes(paths), {"test": component.id})

    def test_unified_ai_activation_replaces_legacy_groups_and_cleans_old_layers(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = ApplicationPaths(Path(folder))
            paths.ensure()
            components_root = paths.runtimes / "components"
            legacy = components_root / "qwen-cuda-win-x64"
            unified = components_root / "ai-cuda-win-x64"
            for directory, component_id, group, packages in (
                (legacy, "qwen-cuda-win-x64", "qwen", ["torch==2.11.0"]),
                (unified, "ai-cuda-win-x64", "ai", ["torch==2.11.0"]),
            ):
                directory.mkdir(parents=True)
                (directory / "site-packages").mkdir()
                (directory / "runtime.json").write_text(json.dumps({
                    "schema_version": 1,
                    "id": component_id,
                    "group": group,
                    "variant": "cuda128",
                    "kind": "python-layer",
                    "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                    "platform": "win_amd64",
                    "site_packages": "site-packages",
                    "packages": packages,
                }), encoding="utf-8")

            activate_component("qwen", "qwen-cuda-win-x64", paths)
            activate_component("ai", "ai-cuda-win-x64", paths)
            self.assertEqual({"ai": "ai-cuda-win-x64"}, load_active_runtimes(paths))
            self.assertEqual(("qwen-cuda-win-x64",), cleanup_inactive_ai_components(paths))
            self.assertFalse(legacy.exists())
            self.assertTrue(unified.exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from audioalign.core.paths import ApplicationPaths
from audioalign.core.runtime_addons import (
    component_manifest,
    install_runtime_component,
    load_active_runtimes,
    load_runtime_index,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeAddonTests(unittest.TestCase):
    def test_bundled_index_uses_package_sources_and_no_github_assets(self) -> None:
        index_path = ROOT / "runtime-packages" / "runtime-index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["components"])
        for component in payload["components"]:
            self.assertTrue(component["packages"])
            self.assertNotIn("url", component)
            self.assertNotIn("archive", component)
            self.assertNotIn("github.com", json.dumps(component).lower())

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


if __name__ == "__main__":
    unittest.main()

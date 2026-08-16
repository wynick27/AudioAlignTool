from __future__ import annotations

import os
import site
import sys
import sysconfig
from pathlib import Path


_DLL_HANDLES: list[object] = []
_REGISTERED_DLL_PATHS: list[str] = []
_ACTIVE_PACKAGE_PATHS: tuple[str, ...] = ()


def _candidate_site_packages() -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(Path(value) for value in _ACTIVE_PACKAGE_PATHS)
    configured = sysconfig.get_paths().get("purelib")
    if configured:
        candidates.append(Path(configured))
    try:
        candidates.extend(Path(value) for value in site.getsitepackages())
    except AttributeError:
        pass
    candidates.append(Path(sys.prefix) / "Lib" / "site-packages")
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.extend((Path(bundle_root), Path(bundle_root) / "_internal"))
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def activate_optional_runtimes() -> tuple[str, ...]:
    """Activate app-local runtime add-ons before importing model libraries."""
    global _ACTIVE_PACKAGE_PATHS
    if not _ACTIVE_PACKAGE_PATHS:
        from .runtime_addons import activate_runtime_paths

        _ACTIVE_PACKAGE_PATHS = activate_runtime_paths()
    return _ACTIVE_PACKAGE_PATHS


def native_runtime_directories() -> list[Path]:
    activate_optional_runtimes()
    relative = (
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cudnn" / "bin",
        Path("nvidia") / "cuda_runtime" / "bin",
        Path("torch") / "lib",
    )
    found: list[Path] = []
    for root in _candidate_site_packages():
        for suffix in relative:
            candidate = root / suffix
            if candidate.is_dir() and candidate not in found:
                found.append(candidate)
        # PyInstaller can flatten package DLLs into the bundle root.
        if (root / "cublas64_12.dll").is_file() or (root / "cudnn64_9.dll").is_file():
            if root not in found:
                found.append(root)
    return found


def bootstrap_native_runtime() -> tuple[str, ...]:
    """Register app-local NVIDIA DLL folders without changing system PATH."""
    activate_optional_runtimes()
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return tuple(_REGISTERED_DLL_PATHS)
    for directory in native_runtime_directories():
        value = str(directory)
        if value in _REGISTERED_DLL_PATHS:
            continue
        try:
            handle = os.add_dll_directory(value)
        except OSError:
            continue
        _DLL_HANDLES.append(handle)
        _REGISTERED_DLL_PATHS.append(value)
    return tuple(_REGISTERED_DLL_PATHS)


def subprocess_runtime_environment() -> dict[str, str]:
    """Return a process-only environment for model worker subprocesses."""
    paths = bootstrap_native_runtime()
    environment = dict(os.environ)
    if paths:
        environment["PATH"] = os.pathsep.join((*paths, environment.get("PATH", "")))
    return environment

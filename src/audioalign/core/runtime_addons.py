from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable

from .paths import ApplicationPaths


RUNTIME_SCHEMA_VERSION = 1
ProgressCallback = Callable[[float, str], None]
_PIP_RAW_PROGRESS = re.compile(r"^Progress\s+(\d+)\s+of\s+(\d+)$")


@dataclass(frozen=True, slots=True)
class RuntimeComponent:
    id: str
    group: str
    variant: str
    display_name: str
    packages: tuple[str, ...]
    no_deps_packages: tuple[str, ...] = ()
    index_url: str = "https://pypi.org/simple"
    extra_index_urls: tuple[str, ...] = ()
    size: int = 0
    python_abi: str = ""
    platform: str = "win_amd64"
    description: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "RuntimeComponent":
        return cls(
            id=str(payload["id"]),
            group=str(payload["group"]),
            variant=str(payload.get("variant", "")),
            display_name=str(payload.get("display_name", payload["id"])),
            packages=tuple(str(item) for item in payload.get("packages", ())),
            no_deps_packages=tuple(
                str(item) for item in payload.get("no_deps_packages", ())
            ),
            index_url=str(payload.get("index_url", "https://pypi.org/simple")),
            extra_index_urls=tuple(
                str(item) for item in payload.get("extra_index_urls", ())
            ),
            size=int(payload.get("size", 0)),
            python_abi=str(payload.get("python_abi", "")),
            platform=str(payload.get("platform", "win_amd64")),
            description=str(payload.get("description", "")),
        )


def _runtime_root(paths: ApplicationPaths | None = None) -> Path:
    return (paths or ApplicationPaths.current()).runtimes


def _active_file(paths: ApplicationPaths | None = None) -> Path:
    return _runtime_root(paths) / "active.json"


def load_active_runtimes(paths: ApplicationPaths | None = None) -> dict[str, str]:
    path = _active_file(paths)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text("utf-8"))
        return {str(key): str(value) for key, value in payload.get("active", {}).items()}
    except (OSError, ValueError, TypeError):
        return {}


def _write_active_runtimes(active: dict[str, str], paths: ApplicationPaths | None = None) -> None:
    root = _runtime_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    target = _active_file(paths)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "active": active}, indent=2), "utf-8")
    os.replace(temporary, target)


def component_directory(component_id: str, paths: ApplicationPaths | None = None) -> Path:
    return _runtime_root(paths) / "components" / component_id


def component_manifest(component_id: str, paths: ApplicationPaths | None = None) -> dict | None:
    manifest = component_directory(component_id, paths) / "runtime.json"
    if not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return value if value.get("id") == component_id else None


def active_runtime_manifest(group: str, paths: ApplicationPaths | None = None) -> dict | None:
    component_id = load_active_runtimes(paths).get(group)
    manifest = component_manifest(component_id, paths) if component_id else None
    if not manifest:
        return None
    abi = str(manifest.get("python_abi", ""))
    if manifest.get("kind") == "python-layer" and abi and abi != _host_python_abi():
        return None
    return manifest


def activate_component(group: str, component_id: str, paths: ApplicationPaths | None = None) -> None:
    manifest = component_manifest(component_id, paths)
    if manifest is None or manifest.get("group") != group:
        raise ValueError("运行时组件不存在或类型不匹配")
    active = load_active_runtimes(paths)
    if group == "ai":
        # The unified AI layer supersedes the former independent Qwen,
        # WhisperX and CTranslate2-CUDA layers.  Keeping any of those active
        # would put a second Torch/CUDA stack on sys.path.
        for legacy_group in ("qwen", "whisperx", "cuda"):
            active.pop(legacy_group, None)
    active[group] = component_id
    _write_active_runtimes(active, paths)


def deactivate_component(group: str, paths: ApplicationPaths | None = None) -> None:
    active = load_active_runtimes(paths)
    if active.pop(group, None) is not None:
        _write_active_runtimes(active, paths)


def _host_python_abi() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _validate_manifest(manifest: dict, expected: RuntimeComponent) -> None:
    if int(manifest.get("schema_version", 0)) != RUNTIME_SCHEMA_VERSION:
        raise ValueError("不支持的运行时组件格式")
    if manifest.get("id") != expected.id or manifest.get("group") != expected.group:
        raise ValueError("运行时组件清单与索引不一致")
    wanted_platform = str(manifest.get("platform", expected.platform))
    if wanted_platform == "win_amd64" and (os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}):
        raise ValueError("该运行时组件仅支持 Windows x64")
    abi = str(manifest.get("python_abi", ""))
    if manifest.get("kind") == "python-layer" and abi and abi != _host_python_abi():
        raise ValueError(f"组件需要 {abi}，当前程序为 {_host_python_abi()}")


def load_runtime_index(
    paths: ApplicationPaths | None = None,
) -> list[RuntimeComponent]:
    application_paths = paths or ApplicationPaths.current()
    index = application_paths.runtime_index
    if not index.is_file():
        raise FileNotFoundError(f"本地运行时索引不存在：{index}")
    payload = json.loads(index.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != RUNTIME_SCHEMA_VERSION:
        raise ValueError("不支持的运行时索引格式")
    components = [RuntimeComponent.from_dict(item) for item in payload.get("components", [])]
    for component in components:
        if not component.packages and not component.no_deps_packages:
            raise ValueError(f"运行时组件 {component.id} 没有声明 PyPI 包")
        if component.platform != "win_amd64":
            raise ValueError(f"运行时组件 {component.id} 的平台不受支持")
        sources = (component.index_url, *component.extra_index_urls)
        if any("github.com" in source.casefold() for source in sources):
            raise ValueError("运行时组件源不能指向 GitHub")
        declared_packages = (*component.packages, *component.no_deps_packages)
        if any("://" in package or package.startswith("git+") for package in declared_packages):
            raise ValueError("运行时组件只能声明包名和版本，不能包含直接下载地址")
    return components


def _pip_command() -> list[str]:
    """Use the bundled pip entry point in frozen builds, normal pip in source."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--runtime-pip"]
    return [sys.executable, "-m", "pip"]


def _install_pypi_packages(
    component: RuntimeComponent,
    target: Path,
    progress: ProgressCallback | None,
) -> None:
    def run(packages: tuple[str, ...], *, no_deps: bool, start: float, end: float) -> None:
        if not packages:
            return
        command = [
            *_pip_command(), "install", "--disable-pip-version-check", "--no-input",
            "--only-binary=:all:", "--progress-bar", "raw", "--target", str(target),
            "--index-url", component.index_url,
        ]
        for index_url in component.extra_index_urls:
            command.extend(("--extra-index-url", index_url))
        if no_deps:
            # WhisperX currently declares torch~=2.8 even though its current
            # pyannote stack works with newer Torch.  Its dependencies are
            # installed explicitly in the first phase; only the WhisperX
            # wheel itself bypasses the upstream Torch constraint here.
            command.extend(("--no-deps", "--ignore-requires-python"))
        command.extend(packages)
        if progress:
            progress(start, f"正在解析 {component.display_name} 的依赖")
        environment = dict(os.environ)
        environment["PYTHONUTF8"] = "1"
        environment["PIP_NO_INPUT"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=environment,
        )
        recent: list[str] = []
        displayed_progress = start
        download_index = -1
        previous_download: tuple[int, int] | None = None
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                recent.append(line)
                del recent[:-12]
                if progress:
                    matched = _PIP_RAW_PROGRESS.match(line)
                    if matched:
                        current, total = (int(value) for value in matched.groups())
                        if (
                            previous_download is None
                            or current < previous_download[0]
                            or (current == 0 and total != previous_download[1])
                        ):
                            download_index += 1
                        previous_download = (current, total)
                        file_fraction = current / total if total > 0 else 0.0
                        local_base = 0.85 * (1.0 - 0.5 ** max(0, download_index))
                        local_span = 0.85 * 0.5 ** (max(0, download_index) + 1)
                        local_progress = min(0.85, local_base + local_span * file_fraction)
                        displayed_progress = max(
                            displayed_progress,
                            start + (end - start) * local_progress,
                        )
                        if total > 0:
                            progress(
                                displayed_progress,
                                f"正在下载运行时文件 {current / 1024**2:.1f}/{total / 1024**2:.1f} MB",
                            )
                    else:
                        folded = line.casefold()
                        if "installing collected packages" in folded:
                            displayed_progress = max(displayed_progress, start + (end - start) * 0.90)
                        elif "successfully installed" in folded:
                            displayed_progress = max(displayed_progress, end)
                        progress(displayed_progress, line)
        return_code = process.wait()
        if return_code:
            detail = "\n".join(recent[-8:])
            raise RuntimeError(f"pip 安装失败（退出码 {return_code}）\n{detail}")

    regular_end = 0.94 if component.no_deps_packages else 0.99
    run(component.packages, no_deps=False, start=0.01, end=regular_end)
    run(component.no_deps_packages, no_deps=True, start=regular_end, end=0.99)


def install_runtime_component(
    component: RuntimeComponent,
    paths: ApplicationPaths | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    application_paths = paths or ApplicationPaths.current()
    existing = component_manifest(component.id, application_paths)
    if existing is not None:
        try:
            _validate_manifest(existing, component)
            package_root = component_directory(component.id, application_paths) / str(
                existing.get("site_packages", "site-packages")
            )
            if package_root.is_dir():
                activate_component(component.group, component.id, application_paths)
                if progress:
                    progress(1.0, f"{component.display_name} 已存在，已直接切换启用；重启后生效")
                return component_directory(component.id, application_paths)
        except ValueError:
            # A stale or incompatible component is replaced only after a new
            # installation has completed successfully below.
            pass
    root = _runtime_root(application_paths)
    components = root / "components"
    components.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{component.id}-", dir=components))
    try:
        site_packages = temporary / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        _install_pypi_packages(component, site_packages, progress)
        manifest = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "id": component.id,
            "group": component.group,
            "variant": component.variant,
            "display_name": component.display_name,
            "kind": "python-layer",
            "python_abi": component.python_abi or _host_python_abi(),
            "platform": component.platform,
            "site_packages": "site-packages",
            "packages": list(component.packages),
            "no_deps_packages": list(component.no_deps_packages),
            "source": "pypi",
        }
        (temporary / "runtime.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        _validate_manifest(manifest, component)
        target = component_directory(component.id, application_paths)
        old = target.with_name(target.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        if target.exists():
            os.replace(target, old)
        os.replace(temporary, target)
        if old.exists():
            shutil.rmtree(old)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    activate_component(component.group, component.id, application_paths)
    if progress:
        progress(1.0, f"{component.display_name} 已安装，重启后生效")
    return target


def remove_runtime_component(component_id: str, paths: ApplicationPaths | None = None) -> None:
    active = load_active_runtimes(paths)
    changed = False
    for group, active_id in tuple(active.items()):
        if active_id == component_id:
            del active[group]
            changed = True
    if changed:
        _write_active_runtimes(active, paths)
    target = component_directory(component_id, paths)
    if target.exists():
        shutil.rmtree(target)


def cleanup_inactive_ai_components(paths: ApplicationPaths | None = None) -> tuple[str, ...]:
    """Remove superseded AI layers after restart, when their DLLs are unused."""
    application_paths = paths or ApplicationPaths.current()
    components_root = _runtime_root(application_paths) / "components"
    if not components_root.is_dir():
        return ()
    active_ids = set(load_active_runtimes(application_paths).values())
    removed: list[str] = []
    for directory in components_root.iterdir():
        if not directory.is_dir() or directory.name in active_ids:
            continue
        manifest = component_manifest(directory.name, application_paths)
        if not manifest:
            continue
        packages = (
            *manifest.get("packages", ()),
            *manifest.get("no_deps_packages", ()),
        )
        package_names = {str(item).split("=", 1)[0].split("<", 1)[0].casefold() for item in packages}
        is_ai_layer = (
            manifest.get("group") in {"ai", "qwen", "whisperx", "cuda"}
            or bool(package_names & {"torch", "qwen-asr", "whisperx", "nvidia-cublas-cu12"})
        )
        if not is_ai_layer:
            continue
        try:
            shutil.rmtree(directory)
        except OSError:
            # A previous process may still be releasing a native DLL.  Keep
            # the complete layer and retry on the next clean startup.
            continue
        removed.append(directory.name)
    return tuple(removed)


def activate_runtime_paths(paths: ApplicationPaths | None = None) -> tuple[str, ...]:
    """Prepend active in-process package layers before any AI imports."""
    added: list[str] = []
    for group, component_id in load_active_runtimes(paths).items():
        manifest = component_manifest(component_id, paths)
        if not manifest or manifest.get("kind") != "python-layer":
            continue
        try:
            _validate_manifest(manifest, RuntimeComponent(
                component_id, group, str(manifest.get("variant", "")), component_id,
                (), python_abi=str(manifest.get("python_abi", "")),
            ))
        except ValueError:
            continue
        package_root = component_directory(component_id, paths) / str(manifest.get("site_packages", "site-packages"))
        if package_root.is_dir() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
            added.append(str(package_root))
    return tuple(added)

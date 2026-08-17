from __future__ import annotations

from dataclasses import asdict
import faulthandler
import json
import sys


_CRASH_LOG = None


def _enable_crash_diagnostics() -> None:
    """Keep a useful traceback when a native model or driver terminates Python."""
    global _CRASH_LOG
    try:
        from audioalign.core.paths import application_root

        log_directory = application_root() / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        _CRASH_LOG = (log_directory / "native-crash.log").open("a", encoding="utf-8")
        faulthandler.enable(file=_CRASH_LOG, all_threads=True)
    except (OSError, RuntimeError):
        _CRASH_LOG = None


def _runtime_probe(arguments: list[str]) -> int:
    if len(arguments) != 3:
        return 2
    from audioalign.core.asr import runtime_status

    backend, model, model_root = arguments
    status = runtime_status(model, model_root, backend)
    # Frozen windowed applications can ignore PYTHONIOENCODING and attach
    # stdout using the active Windows code page.  An ASCII-only JSON envelope
    # keeps the probe protocol independent of either process' text encoding;
    # json.loads restores the original Unicode message in the GUI process.
    print(json.dumps(asdict(status), ensure_ascii=True), flush=True)
    return 0


def _runtime_pip(arguments: list[str]) -> int:
    """Private entry point used by the frozen app's runtime installer."""
    if getattr(sys, "frozen", False):
        # distlib's registry knows the standard source/zip loaders but not
        # PyInstaller's frozen loader.  Register its public filesystem finder
        # before importing pip's CLI, which imports distlib.scripts and scans
        # the bundled Windows launcher stubs immediately.
        import pip._vendor.distlib as distlib
        from pip._vendor.distlib import resources

        resources.register_finder(distlib.__loader__, resources.ResourceFinder)
    from pip._internal.cli.main import main as pip_main

    return int(pip_main(arguments))


def main() -> int:
    _enable_crash_diagnostics()
    if len(sys.argv) > 1 and sys.argv[1] == "--runtime-pip":
        return _runtime_pip(sys.argv[2:])

    from audioalign.core.runtime_addons import cleanup_inactive_ai_components
    from audioalign.core.runtime import activate_optional_runtimes, bootstrap_native_runtime

    cleanup_inactive_ai_components()
    activate_optional_runtimes()
    bootstrap_native_runtime()
    if len(sys.argv) > 1 and sys.argv[1] == "--runtime-probe":
        return _runtime_probe(sys.argv[2:])
    try:
        from audioalign.gui.app import run
    except ImportError as exc:
        print(
            "无法启动图形界面。请先安装依赖：pip install -e .\n"
            f"详细错误：{exc}",
            file=sys.stderr,
        )
        return 2
    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

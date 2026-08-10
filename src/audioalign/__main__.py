from __future__ import annotations

from dataclasses import asdict
import json
import sys


def _runtime_probe(arguments: list[str]) -> int:
    if len(arguments) != 3:
        return 2
    from audioalign.core.asr import runtime_status

    backend, model, model_root = arguments
    status = runtime_status(model, model_root, backend)
    print(json.dumps(asdict(status), ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    from audioalign.core.runtime import bootstrap_native_runtime

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

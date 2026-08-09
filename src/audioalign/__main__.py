from __future__ import annotations

import sys


def main() -> int:
    from audioalign.core.runtime import bootstrap_native_runtime

    bootstrap_native_runtime()
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

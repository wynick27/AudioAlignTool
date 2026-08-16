from __future__ import annotations

from pathlib import Path
import sys
import zipfile


def main() -> int:
    source, target = (Path(value).resolve() for value in sys.argv[1:3])
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

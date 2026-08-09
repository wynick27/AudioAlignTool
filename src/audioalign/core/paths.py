from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def sanitize_project_name(name: str) -> str:
    value = _INVALID.sub("_", name).strip().rstrip(". ")
    if not value:
        raise ValueError("项目名不能为空")
    if value.upper() in _RESERVED:
        value += " 项目"
    return value


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    root: Path

    @classmethod
    def current(cls) -> "ApplicationPaths":
        return cls(application_root())

    @property
    def projects(self) -> Path:
        return self.root / "projects"

    @property
    def work(self) -> Path:
        return self.root / ".work"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def settings(self) -> Path:
        return self.root / "settings.json"

    def ensure(self) -> None:
        for path in (self.projects, self.work, self.models, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        probe = self.root / ".audioalign-write-test"
        try:
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
        except OSError as exc:
            raise PermissionError(f"程序目录不可写：{self.root}。请把程序移动到可写目录。") from exc

    def project_dir(self, name: str) -> Path:
        return self.projects / sanitize_project_name(name)

    def load_settings(self) -> dict:
        if not self.settings.exists():
            return {}
        return json.loads(self.settings.read_text("utf-8"))

    def save_settings(self, data: dict) -> None:
        temporary = self.settings.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.settings)

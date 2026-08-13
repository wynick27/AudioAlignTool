from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from .text import decode_text
from .timecode import parse_srt_time


_TIMING = re.compile(
    r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})(?:\s+.*)?$"
)
_HTML_TAG = re.compile(r"</?(?:b|i|u|font|ruby|rt|c)(?:\s+[^>]*)?>", re.I)
_SSA_OVERRIDE = re.compile(r"\{\\[^}]+\}")


@dataclass(slots=True, frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    raw_text: str


def visible_subtitle_text(value: str) -> str:
    value = _SSA_OVERRIDE.sub("", value)
    value = _HTML_TAG.sub("", value)
    return html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")


def parse_srt_text(value: str) -> list[SubtitleCue]:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").split("\n")
    cues: list[SubtitleCue] = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        declared_index = None
        if lines[cursor].strip().isdigit():
            declared_index = int(lines[cursor].strip())
            cursor += 1
        if cursor >= len(lines):
            raise ValueError("SRT ended before a timing line")
        match = _TIMING.fullmatch(lines[cursor])
        if not match:
            raise ValueError(f"Invalid SRT timing at line {cursor + 1}: {lines[cursor]}")
        start_ms, end_ms = parse_srt_time(match.group(1)), parse_srt_time(match.group(2))
        if end_ms <= start_ms:
            raise ValueError(f"SRT cue {declared_index or len(cues) + 1} ends before it starts")
        cursor += 1
        text_lines: list[str] = []
        while cursor < len(lines) and lines[cursor].strip():
            text_lines.append(lines[cursor])
            cursor += 1
        if not text_lines:
            raise ValueError(f"SRT cue {declared_index or len(cues) + 1} has no text")
        raw = "\n".join(text_lines)
        cues.append(SubtitleCue(
            declared_index or len(cues) + 1, start_ms, end_ms,
            visible_subtitle_text(raw), raw,
        ))
    if not cues:
        raise ValueError("SRT contains no subtitle cues")
    return sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))


def import_srt(path: str | Path) -> list[SubtitleCue]:
    return parse_srt_text(decode_text(Path(path).read_bytes()))

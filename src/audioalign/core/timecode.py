from __future__ import annotations

import re


_SRT_TIME = re.compile(r"^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})$")
_DISPLAY_TIME = re.compile(r"^(?:(\d{1,3}):)?(\d{1,2}):(\d{2})(?:[,.](\d{1,3}))?$")


def format_time_ms(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    seconds, millis = divmod(milliseconds, 1000)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}.{millis:03d}"


def format_srt_time(milliseconds: int) -> str:
    return format_time_ms(milliseconds).replace(".", ",")


def parse_srt_time(value: str) -> int:
    match = _SRT_TIME.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timecode: {value}")
    hours, minutes, seconds, millis = (int(item) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid SRT timecode: {value}")
    millis *= 10 ** max(0, 3 - len(match.group(4)))
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_display_time(value: str) -> int:
    match = _DISPLAY_TIME.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid timecode: {value}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis_text = match.group(4) or "0"
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid timecode: {value}")
    millis = int(millis_text) * 10 ** max(0, 3 - len(millis_text))
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis

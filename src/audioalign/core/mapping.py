from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import AudioAsset, AudioChapterMarker, Chapter, ChapterAudioLink


_NON_BODY = re.compile(
    r"\b(cover|copyright|contents?|table of contents|title page|portada|copyright|indice|\u76ee\u5f55|\u7248\u6743|\u5c01\u9762)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?:chapter|cap[i\u00ed]tulo|chapitre|\u7b2c)?\s*(\d+|[ivxlcdm]+)", re.IGNORECASE)


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(character for character in value if not unicodedata.combining(character))
    return "".join(character for character in value if character.isalnum())


def is_probably_body(chapter: Chapter) -> bool:
    return not bool(_NON_BODY.search(chapter.title))


def _chapter_number(value: str) -> str:
    match = _NUMBER.search(value)
    if not match:
        return ""
    number = match.group(1).casefold()
    return str(int(number)) if number.isdigit() else number


@dataclass(frozen=True, slots=True)
class AudioSlice:
    audio_id: int
    title: str
    path_label: str
    start_ms: int
    end_ms: int
    order: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


def audio_slices(assets: list[AudioAsset], markers: dict[int, list[AudioChapterMarker]]) -> list[AudioSlice]:
    result: list[AudioSlice] = []
    order = 0
    for asset in assets:
        if asset.id is None:
            continue
        chapters = markers.get(asset.id, [])
        if chapters:
            for marker in chapters:
                result.append(AudioSlice(asset.id, marker.title, asset.path.name, marker.start_ms, marker.end_ms, order))
                order += 1
        else:
            result.append(AudioSlice(asset.id, asset.title or asset.path.stem, asset.path.name, 0, asset.duration_ms, order))
            order += 1
    return result


def match_score(chapter: Chapter, choice: AudioSlice, text_length: int = 0) -> float:
    left = normalize_title(chapter.title)
    right = normalize_title(choice.title + " " + choice.path_label)
    title_score = SequenceMatcher(None, left, right).ratio() if left and right else 0.0
    chapter_number, audio_number = _chapter_number(chapter.title), _chapter_number(choice.title + " " + choice.path_label)
    number_score = 1.0 if chapter_number and chapter_number == audio_number else 0.0
    duration_score = 0.0
    if text_length and choice.duration_ms:
        # Wide audiobook speech-rate prior: useful only as a tie breaker.
        expected = max(10_000, text_length * 260)
        duration_score = min(expected, choice.duration_ms) / max(expected, choice.duration_ms)
    return 0.55 * number_score + 0.35 * title_score + 0.10 * duration_score


def automatic_links(
    chapters: list[Chapter],
    choices: list[AudioSlice],
    text_lengths: dict[int, int] | None = None,
) -> list[ChapterAudioLink]:
    """Monotonic greedy assignment with title/number/duration scoring."""
    text_lengths = text_lengths or {}
    largest = max(text_lengths.values(), default=0)
    minimum_body_length = max(500, round(largest * 0.03)) if largest else 0
    candidates = [
        chapter
        for chapter in chapters
        if chapter.id is not None
        and is_probably_body(chapter)
        and (not text_lengths or text_lengths.get(chapter.id or 0, 0) >= minimum_body_length)
    ]
    links: list[ChapterAudioLink] = []
    cursor = 0
    for chapter in candidates:
        if cursor >= len(choices):
            break
        window = choices[cursor : min(len(choices), cursor + 5)]
        best = max(window, key=lambda item: match_score(chapter, item, text_lengths.get(chapter.id or 0, 0)))
        cursor = best.order + 1
        links.append(
            ChapterAudioLink(
                None,
                chapter.id or 0,
                best.audio_id,
                0,
                best.start_ms,
                best.end_ms,
                match_score(chapter, best, text_lengths.get(chapter.id or 0, 0)),
            )
        )
    return links

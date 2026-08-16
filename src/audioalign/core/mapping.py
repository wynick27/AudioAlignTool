from __future__ import annotations

import re
import math
import statistics
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import AudioAsset, AudioChapterMarker, Chapter, ChapterAudioLink


_NON_BODY = re.compile(
    r"\b(cover|copyright|contents?|table of contents|title page|portada|copyright|indice|\u76ee\u5f55|\u7248\u6743|\u5c01\u9762)\b",
    re.IGNORECASE,
)
_PREFIXED_NUMBER = re.compile(
    r"\b(?:chapter|cap[i\u00ed]tulo|chapitre)\s*(\d+|[ivxlcdm]+)\b|\u7b2c\s*(\d+)\s*\u7ae0",
    re.IGNORECASE,
)
_LEADING_NUMBER = re.compile(r"^\s*(\d+|[ivxlcdm]+)(?:\b|[._-])", re.IGNORECASE)
_GENERIC_AUDIO_CHAPTER = re.compile(
    r"^\s*chapter\s*0*(\d+)(?:\s|[-\u2013\u2014:]|$)", re.IGNORECASE,
)
_ENGLISH_NUMBERS = {
    name: str(number)
    for number, name in enumerate((
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo",
        "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven",
        "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree",
        "thirtyfour", "thirtyfive", "thirtysix", "thirtyseven", "thirtyeight",
        "thirtynine", "forty",
    ))
}


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(character for character in value if not unicodedata.combining(character))
    return "".join(character for character in value if character.isalnum())


def is_probably_body(chapter: Chapter) -> bool:
    visible = "".join(
        character for character in unicodedata.normalize("NFKC", chapter.title)
        if unicodedata.category(character) != "Cf"
    )
    return not bool(_NON_BODY.search(visible))


def _chapter_number(value: str) -> str:
    visible = "".join(
        character for character in unicodedata.normalize("NFKC", value)
        if unicodedata.category(character) != "Cf"
    )
    compact = "".join(character for character in visible.casefold() if character.isalnum())
    if compact in _ENGLISH_NUMBERS:
        return _ENGLISH_NUMBERS[compact]
    chapter_suffix = compact.split("chapter", 1)[1] if "chapter" in compact else ""
    if chapter_suffix:
        for word in sorted(_ENGLISH_NUMBERS, key=len, reverse=True):
            if chapter_suffix.startswith(word):
                return _ENGLISH_NUMBERS[word]
    match = _PREFIXED_NUMBER.search(visible)
    if match:
        number = next(group for group in match.groups() if group)
        return str(int(number)) if number.isdigit() else number.casefold()
    match = _LEADING_NUMBER.search(visible)
    if not match:
        return ""
    number = match.group(1).casefold()
    return str(int(number)) if number.isdigit() else number


def _generic_audio_chapter_number(choice: "AudioSlice") -> int | None:
    """Return the ordinal of an unlabelled embedded M4B chapter marker."""
    match = _GENERIC_AUDIO_CHAPTER.match(choice.title)
    return int(match.group(1)) if match else None


def _numbered_text_run(chapters: list[Chapter]) -> tuple[int, int] | None:
    """Find the longest consecutive semantic chapter-number run."""
    best: tuple[int, int] | None = None
    start = 0
    previous: int | None = None
    for index in range(len(chapters) + 1):
        number_text = _chapter_number(chapters[index].title) if index < len(chapters) else ""
        number = int(number_text) if number_text.isdigit() else None
        if number is not None and (previous is None or number == previous + 1):
            if previous is None:
                start = index
            previous = number
            continue
        if previous is not None:
            candidate = (start, index)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
        start = index
        previous = number
    return best if best and best[1] - best[0] >= 4 else None


def _generic_sequence_assignment(
    chapters: list[Chapter], choices: list["AudioSlice"], text_lengths: dict[int, int],
) -> tuple[dict[int, int], float]:
    """Align a numbered text run to generic M4B markers by duration shape.

    Embedded markers named ``Chapter 003`` are container ordinals, not
    necessarily book chapter numbers: title pages and front matter commonly
    introduce an offset.  Comparing an entire run avoids the former greedy
    jumps while remaining language/model independent.
    """
    run = _numbered_text_run(chapters)
    if run is None:
        return {}, 0.0
    text_start, text_end = run
    run_length = text_end - text_start
    lengths = [text_lengths.get(chapters[index].id or 0, 0) for index in range(text_start, text_end)]
    if not all(length > 0 for length in lengths):
        return {}, 0.0
    generic = [
        (index, _generic_audio_chapter_number(choice))
        for index, choice in enumerate(choices)
        if _generic_audio_chapter_number(choice) is not None
    ]
    best: tuple[float, list[int]] | None = None
    for offset in range(0, len(generic) - run_length + 1):
        window = generic[offset : offset + run_length]
        indices = [item[0] for item in window]
        ordinals = [item[1] for item in window]
        if any(indices[index] + 1 != indices[index + 1] for index in range(run_length - 1)):
            continue
        if any(ordinals[index] + 1 != ordinals[index + 1] for index in range(run_length - 1)):
            continue
        rates = [choices[choice_index].duration_ms / length for choice_index, length in zip(indices, lengths)]
        if any(rate <= 0 for rate in rates):
            continue
        median_rate = statistics.median(rates)
        dispersion = statistics.mean(abs(math.log(rate / median_rate)) for rate in rates)
        if best is None or dispersion < best[0]:
            best = dispersion, indices
    if best is None:
        return {}, 0.0
    dispersion, indices = best
    # A weak duration shape must not manufacture a confident offset.
    if dispersion > 0.32:
        return {}, 0.0
    confidence = max(0.55, min(0.98, math.exp(-3.0 * dispersion)))
    return {
        text_start + offset: choice_index
        for offset, choice_index in enumerate(indices)
    }, confidence


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
    sequence, sequence_confidence = _generic_sequence_assignment(candidates, choices, text_lengths)
    assignments: dict[int, int] = dict(sequence)
    if sequence:
        first_text, last_text = min(sequence), max(sequence)
        first_choice, last_choice = sequence[first_text], sequence[last_text]
        upper = first_choice
        for chapter_index in range(first_text - 1, -1, -1):
            if upper <= 0:
                break
            window_start = max(0, upper - 5)
            choice_index = max(
                range(window_start, upper),
                key=lambda index: match_score(
                    candidates[chapter_index], choices[index],
                    text_lengths.get(candidates[chapter_index].id or 0, 0),
                ),
            )
            assignments[chapter_index] = choice_index
            upper = choice_index
        cursor = last_choice + 1
        tail = list(range(last_text + 1, len(candidates)))
        available = max(0, len(choices) - cursor)
        if len(tail) > available:
            # EPUB spines often contain short adverts/colophons after the
            # narrated epilogue. Keep the largest remaining body sections,
            # in document order, when the M4B has fewer tail markers.
            tail = sorted(
                sorted(
                    tail,
                    key=lambda index: text_lengths.get(candidates[index].id or 0, 0),
                    reverse=True,
                )[:available]
            )
        for tail_position, chapter_index in enumerate(tail):
            if cursor >= len(choices):
                break
            remaining_text = len(tail) - tail_position
            # When the remaining counts match, preserve the obvious tail
            # sequence (Epilogue, appendices, etc.) instead of skipping it.
            if len(choices) - cursor == remaining_text:
                choice_index = cursor
            else:
                window = range(cursor, min(len(choices), cursor + 5))
                choice_index = max(
                    window,
                    key=lambda index: match_score(
                        candidates[chapter_index], choices[index],
                        text_lengths.get(candidates[chapter_index].id or 0, 0),
                    ),
                )
            assignments[chapter_index] = choice_index
            cursor = choice_index + 1
    else:
        cursor = 0
        for chapter_index, chapter in enumerate(candidates):
            if cursor >= len(choices):
                break
            window = range(cursor, min(len(choices), cursor + 5))
            choice_index = max(
                window,
                key=lambda index: match_score(
                    chapter, choices[index], text_lengths.get(chapter.id or 0, 0),
                ),
            )
            assignments[chapter_index] = choice_index
            cursor = choice_index + 1

    links: list[ChapterAudioLink] = []
    for chapter_index, chapter in enumerate(candidates):
        if chapter_index not in assignments:
            continue
        best = choices[assignments[chapter_index]]
        confidence = match_score(chapter, best, text_lengths.get(chapter.id or 0, 0))
        if chapter_index in sequence:
            confidence = max(confidence, sequence_confidence)
        links.append(
            ChapterAudioLink(
                None,
                chapter.id or 0,
                best.audio_id,
                0,
                best.start_ms,
                best.end_ms,
                confidence,
            )
        )
    return links

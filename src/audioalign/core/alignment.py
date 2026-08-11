from __future__ import annotations

from bisect import bisect_right
from difflib import SequenceMatcher
from statistics import mean
from dataclasses import dataclass
from typing import Sequence

from .models import (
    ASRToken,
    BoundaryCandidate,
    SegmentOrigin,
    SegmentStatus,
    SilenceAlignmentOptions,
    TextAudioAnchor,
    TextSegment,
)
from .text import normalize_for_match


@dataclass(slots=True)
class SilenceAlignmentResult:
    segments: list[TextSegment]
    anchors: list[TextAudioAnchor]
    unused_candidates: list[BoundaryCandidate]
    proposed_count: int = 0
    stopped_early: bool = False


def segments_from_asr_tokens(
    tokens: Sequence[ASRToken],
    *,
    chapter_id: int,
    max_duration_ms: int = 12_000,
    pause_boundary_ms: int = 700,
) -> list[TextSegment]:
    """Create editable sentence segments when a project has audio but no source text."""
    if not tokens:
        return []
    groups: list[list[ASRToken]] = []
    current: list[ASRToken] = []
    terminal = set("。！？!?；;")
    for index, token in enumerate(tokens):
        current.append(token)
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        pause = (next_token.start_ms - token.end_ms) if next_token else pause_boundary_ms
        duration = token.end_ms - current[0].start_ms
        visible_length = sum(len(item.text.strip()) for item in current)
        punctuation_end = token.text.rstrip()[-1:] in terminal
        too_long = duration >= max_duration_ms or visible_length >= 90
        if punctuation_end or pause >= pause_boundary_ms or too_long or next_token is None:
            groups.append(current)
            current = []
    result: list[TextSegment] = []
    for position, group in enumerate(groups):
        text = "".join(token.text for token in group).strip()
        probability = mean(token.probability for token in group)
        result.append(
            TextSegment(
                id=None,
                chapter_id=chapter_id,
                position=position,
                text=text,
                start_ms=group[0].start_ms,
                end_ms=group[-1].end_ms,
                confidence=max(0.0, min(1.0, probability)),
                status=SegmentStatus.AUTO if probability >= 0.58 else SegmentStatus.LOW_CONFIDENCE,
                locked=False,
                origin=SegmentOrigin.ASR,
            )
        )
    return result


def _token_character_map(tokens: Sequence[ASRToken], language: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    owner: list[int] = []
    for index, token in enumerate(tokens):
        normalized = normalize_for_match(token.text, language)
        characters.extend(normalized)
        owner.extend([index] * len(normalized))
    return "".join(characters), owner


def anchors_from_segments_tokens(
    segments: Sequence[TextSegment],
    tokens: Sequence[ASRToken],
    *,
    language: str = "auto",
) -> list[TextAudioAnchor]:
    """Map original source character ranges to matched ASR word times."""
    if not segments or not tokens:
        return []
    source_characters: list[str] = []
    source_owner: list[int] = []
    segment_ranges: list[tuple[int, int, TextSegment]] = []
    raw_offset = 0
    for segment in segments:
        start = raw_offset
        for local_index, character in enumerate(segment.text):
            normalized = normalize_for_match(character, language)
            source_characters.extend(normalized)
            source_owner.extend([raw_offset + local_index] * len(normalized))
        raw_offset += len(segment.text)
        segment_ranges.append((start, raw_offset, segment))
    transcript, transcript_owner = _token_character_map(tokens, language)
    matcher = SequenceMatcher(None, "".join(source_characters), transcript, autojunk=False)
    token_source_positions: dict[int, list[int]] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            source_index = block.a + offset
            transcript_index = block.b + offset
            if source_index >= len(source_owner) or transcript_index >= len(transcript_owner):
                continue
            token_source_positions.setdefault(transcript_owner[transcript_index], []).append(source_owner[source_index])
    anchors: list[TextAudioAnchor] = []
    chapter_id = segments[0].chapter_id
    for token_index, positions in sorted(token_source_positions.items()):
        token = tokens[token_index]
        start_char, end_char = min(positions), max(positions) + 1
        segment_id = next(
            (segment.id for start, end, segment in segment_ranges if start <= start_char < end),
            None,
        )
        anchors.append(
            TextAudioAnchor(
                None,
                chapter_id,
                segment_id,
                start_char,
                end_char,
                token.start_ms,
                token.end_ms,
                token.probability,
                "asr-word",
            )
        )
    return anchors


def align_segments_to_tokens(
    segments: Sequence[TextSegment],
    tokens: Sequence[ASRToken],
    *,
    language: str = "auto",
    low_confidence: float = 0.58,
) -> list[TextSegment]:
    """Monotonic fuzzy alignment using character-level matching blocks.

    Original text is never changed. Matching blocks map normalized source
    characters to normalized ASR characters, which then map back to word times.
    """
    if not tokens:
        return [
            TextSegment(
                id=s.id, chapter_id=s.chapter_id, position=s.position, text=s.text,
                start_ms=s.start_ms, end_ms=s.end_ms, confidence=0.0,
                status=s.status if s.locked else SegmentStatus.UNMATCHED, locked=s.locked,
                origin=s.origin, source_fragment_id=s.source_fragment_id,
                source_start_char=s.source_start_char, source_end_char=s.source_end_char,
            )
            for s in segments
        ]
    source_parts = [normalize_for_match(segment.text, language) for segment in segments]
    offsets: list[int] = [0]
    for part in source_parts:
        offsets.append(offsets[-1] + len(part))
    source_text = "".join(source_parts)
    transcript, transcript_owner = _token_character_map(tokens, language)
    matcher = SequenceMatcher(None, source_text, transcript, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset

    result: list[TextSegment] = []
    last_end = 0
    for index, segment in enumerate(segments):
        if segment.locked:
            result.append(segment)
            last_end = max(last_end, segment.end_ms)
            continue
        start_offset, end_offset = offsets[index], offsets[index + 1]
        matched_chars = [mapping[pos] for pos in range(start_offset, end_offset) if pos in mapping]
        coverage = len(matched_chars) / max(1, end_offset - start_offset)
        if not matched_chars or coverage < 0.18:
            result.append(
                TextSegment(
                    id=segment.id, chapter_id=segment.chapter_id, position=segment.position,
                    text=segment.text, start_ms=last_end, end_ms=last_end,
                    confidence=coverage, status=SegmentStatus.UNMATCHED, locked=False,
                    origin=segment.origin, source_fragment_id=segment.source_fragment_id,
                    source_start_char=segment.source_start_char, source_end_char=segment.source_end_char,
                )
            )
            continue
        first_char, last_char = min(matched_chars), max(matched_chars)
        first_token = transcript_owner[first_char]
        last_token = transcript_owner[last_char]
        start_ms = max(last_end, tokens[first_token].start_ms)
        end_ms = max(start_ms, tokens[last_token].end_ms)
        probabilities = [token.probability for token in tokens[first_token:last_token + 1]]
        probability = mean(probabilities) if probabilities else 0.0
        confidence = max(0.0, min(1.0, coverage * (0.7 + 0.3 * max(0.0, probability))))
        status = SegmentStatus.AUTO if confidence >= low_confidence else SegmentStatus.LOW_CONFIDENCE
        result.append(
            TextSegment(
                id=segment.id, chapter_id=segment.chapter_id, position=segment.position,
                text=segment.text, start_ms=start_ms, end_ms=end_ms,
                confidence=confidence, status=status, locked=False,
                origin=segment.origin, source_fragment_id=segment.source_fragment_id,
                source_start_char=segment.source_start_char, source_end_char=segment.source_end_char,
            )
        )
        last_end = end_ms
    return result


def snap_boundaries(
    segments: Sequence[TextSegment],
    candidates: Sequence[BoundaryCandidate],
    *,
    window_ms: int = 1000,
    padding_ms: int = 80,
) -> list[TextSegment]:
    times = sorted(candidate.time_ms for candidate in candidates)
    output = [
        TextSegment(
            id=s.id, chapter_id=s.chapter_id, position=s.position, text=s.text,
            start_ms=s.start_ms, end_ms=s.end_ms, confidence=s.confidence,
            status=s.status, locked=s.locked,
            origin=s.origin, source_fragment_id=s.source_fragment_id,
            source_start_char=s.source_start_char, source_end_char=s.source_end_char,
        )
        for s in segments
    ]
    for index in range(len(output) - 1):
        left, right = output[index], output[index + 1]
        if left.locked or right.locked or left.end_ms <= 0:
            continue
        target = (left.end_ms + right.start_ms) // 2 if right.start_ms else left.end_ms
        insertion = bisect_right(times, target)
        nearby = times[max(0, insertion - 3): insertion + 3]
        if not nearby:
            continue
        selected = min(nearby, key=lambda value: abs(value - target))
        if abs(selected - target) > window_ms:
            continue
        left.end_ms = max(left.start_ms, selected - padding_ms)
        right.start_ms = min(max(left.end_ms, selected + padding_ms), max(selected + padding_ms, right.end_ms))
        if left.status == SegmentStatus.AUTO:
            left.status = SegmentStatus.MANUAL
        if right.status == SegmentStatus.AUTO:
            right.status = SegmentStatus.MANUAL
    return output


def enforce_monotonic(segments: Sequence[TextSegment]) -> list[TextSegment]:
    result: list[TextSegment] = []
    cursor = 0
    for source in segments:
        start = max(cursor, source.start_ms)
        end = max(start, source.end_ms)
        result.append(
            TextSegment(
                id=source.id, chapter_id=source.chapter_id, position=source.position,
                text=source.text, start_ms=start, end_ms=end, confidence=source.confidence,
                status=source.status, locked=source.locked,
                origin=source.origin, source_fragment_id=source.source_fragment_id,
                source_start_char=source.source_start_char, source_end_char=source.source_end_char,
            )
        )
        cursor = end
    return result


def _copy_segment(source: TextSegment) -> TextSegment:
    return TextSegment(
        source.id, source.chapter_id, source.position, source.text,
        source.start_ms, source.end_ms, source.confidence, source.status, source.locked,
        source.origin, source.source_fragment_id, source.source_start_char, source.source_end_char,
    )


def _select_silence_boundaries(
    weights: list[int],
    candidates: list[BoundaryCandidate],
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, bool, float]]:
    """Choose monotonic silence boundaries nearest cumulative text-duration targets."""
    needed = max(0, len(weights) - 1)
    if not needed:
        return []
    total_weight = max(1, sum(weights))
    cumulative = 0
    targets: list[int] = []
    for weight in weights[:-1]:
        cumulative += weight
        targets.append(start_ms + round((end_ms - start_ms) * cumulative / total_weight))
    available = [item for item in candidates if start_ms < item.time_ms < end_ms]
    if len(available) >= needed:
        # Dynamic programming selects distinct candidates while allowing extra
        # pauses to be skipped. Candidate confidence mildly favours long pauses.
        infinity = float("inf")
        costs = [[infinity] * len(available) for _ in range(needed)]
        previous = [[-1] * len(available) for _ in range(needed)]
        duration = max(1, end_ms - start_ms)
        for boundary_index, target in enumerate(targets):
            for candidate_index, candidate in enumerate(available):
                if candidate_index < boundary_index:
                    continue
                local = abs(candidate.time_ms - target) / duration + (1.0 - candidate.score) * 0.04
                if boundary_index == 0:
                    costs[boundary_index][candidate_index] = local
                    continue
                best_index = min(
                    range(boundary_index - 1, candidate_index),
                    key=lambda index: costs[boundary_index - 1][index],
                    default=-1,
                )
                if best_index >= 0 and costs[boundary_index - 1][best_index] < infinity:
                    costs[boundary_index][candidate_index] = costs[boundary_index - 1][best_index] + local
                    previous[boundary_index][candidate_index] = best_index
        index = min(range(needed - 1, len(available)), key=lambda value: costs[-1][value])
        chosen: list[BoundaryCandidate] = []
        for row in range(needed - 1, -1, -1):
            chosen.append(available[index])
            index = previous[row][index]
        chosen.reverse()
        tolerance = max(2_000, round((end_ms - start_ms) / max(1, len(weights)) * 0.8))
        return [
            (item.time_ms, abs(item.time_ms - target) <= tolerance and item.score >= 0.25, item.score)
            for item, target in zip(chosen, targets)
        ]

    # With too few pauses, reserve each real candidate for its nearest target
    # and estimate the rest. A monotonic clamp prevents crossed boundaries.
    output: list[tuple[int, bool, float]] = [(target, False, 0.35) for target in targets]
    unused_targets = set(range(len(targets)))
    for candidate in sorted(available, key=lambda item: item.score, reverse=True):
        if not unused_targets:
            break
        target_index = min(unused_targets, key=lambda index: abs(targets[index] - candidate.time_ms))
        output[target_index] = (candidate.time_ms, True, candidate.score)
        unused_targets.remove(target_index)
    cursor = start_ms + 1
    normalized: list[tuple[int, bool, float]] = []
    for value, reliable, score in output:
        value = max(cursor, min(end_ms - (needed - len(normalized)), value))
        normalized.append((value, reliable, score))
        cursor = value + 1
    return normalized


def anchors_from_segment_timings(
    segments: Sequence[TextSegment],
    *,
    method: str = "silence-length-estimate",
) -> list[TextAudioAnchor]:
    anchors: list[TextAudioAnchor] = []
    source_offset = 0
    for segment in segments:
        end_offset = source_offset + len(segment.text)
        if segment.end_ms > segment.start_ms:
            anchors.append(TextAudioAnchor(
                None, segment.chapter_id, segment.id, source_offset, end_offset,
                segment.start_ms, segment.end_ms, segment.confidence, method,
            ))
        source_offset = end_offset
    return anchors


def align_segments_from_silence(
    segments: Sequence[TextSegment],
    candidates: Sequence[BoundaryCandidate],
    options: SilenceAlignmentOptions,
) -> SilenceAlignmentResult:
    """Assign timings from pauses and text length without using speech recognition."""
    output = [_copy_segment(segment) for segment in segments]
    if not output or options.start_segment_index >= len(output) or options.end_ms <= options.start_ms:
        return SilenceAlignmentResult(output, anchors_from_segment_timings(output), list(candidates), 0, False)
    start_index = max(0, options.start_segment_index)
    used_times: set[int] = set()
    cursor_index = start_index
    cursor_time = max(0, options.start_ms)
    proposed_count = 0
    stopped_early = False
    while cursor_index < len(output) and cursor_time < options.end_ms:
        if output[cursor_index].locked:
            cursor_time = max(cursor_time, output[cursor_index].end_ms)
            cursor_index += 1
            continue
        run_end_index = cursor_index
        while run_end_index < len(output) and not output[run_end_index].locked:
            run_end_index += 1
        next_locked = output[run_end_index] if run_end_index < len(output) else None
        run_end_ms = min(options.end_ms, next_locked.start_ms) if next_locked and next_locked.start_ms > cursor_time else options.end_ms
        run = output[cursor_index:run_end_index]
        if not run or run_end_ms <= cursor_time:
            break
        weights = []
        for item in run:
            base = max(1, len(normalize_for_match(item.text)))
            # Punctuation predicts a little speaking/pause time without being
            # treated as spoken characters.
            punctuation = sum(item.text.count(mark) for mark in "，、；：,.!?。！？")
            weights.append(base + punctuation * 2)
        boundaries = _select_silence_boundaries(weights, list(candidates), cursor_time, run_end_ms)
        for time_ms, reliable, _score in boundaries:
            if reliable:
                used_times.add(time_ms)
        unreliable_streak = 0
        for local_index, segment in enumerate(run):
            left = cursor_time if local_index == 0 else boundaries[local_index - 1][0] + options.padding_ms
            right = run_end_ms if local_index == len(run) - 1 else boundaries[local_index][0] - options.padding_ms
            segment.start_ms = max(cursor_time, min(run_end_ms, left))
            segment.end_ms = max(segment.start_ms, min(run_end_ms, right))
            neighbouring = []
            if local_index > 0:
                neighbouring.append(boundaries[local_index - 1])
            if local_index < len(boundaries):
                neighbouring.append(boundaries[local_index])
            reliable = all(item[1] for item in neighbouring) if neighbouring else True
            unreliable_streak = 0 if reliable else unreliable_streak + 1
            if unreliable_streak > 3:
                original = segments[cursor_index + local_index]
                segment.start_ms = original.start_ms
                segment.end_ms = original.end_ms
                segment.confidence = original.confidence
                segment.status = original.status
                stopped_early = True
                break
            score = mean(item[2] for item in neighbouring) if neighbouring else 0.7
            duration_weight = weights[local_index] / max(1, sum(weights))
            expected = max(1, (run_end_ms - cursor_time) * duration_weight)
            ratio = min(expected, max(1, segment.end_ms - segment.start_ms)) / max(expected, max(1, segment.end_ms - segment.start_ms))
            segment.confidence = max(0.0, min(1.0, score * 0.65 + ratio * 0.35))
            segment.status = SegmentStatus.AUTO if reliable and segment.confidence >= options.low_confidence else SegmentStatus.LOW_CONFIDENCE
            proposed_count += 1
        if stopped_early:
            break
        cursor_time = max(cursor_time, run_end_ms)
        cursor_index = run_end_index
    unused = [candidate for candidate in candidates if candidate.time_ms not in used_times]
    return SilenceAlignmentResult(
        output, anchors_from_segment_timings(output), unused, proposed_count, stopped_early,
    )

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean
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
from .text import is_cjk_text, normalize_for_match


@dataclass(slots=True)
class SilenceAlignmentResult:
    segments: list[TextSegment]
    anchors: list[TextAudioAnchor]
    unused_candidates: list[BoundaryCandidate]
    proposed_count: int = 0
    stopped_early: bool = False


@dataclass(slots=True, frozen=True)
class _SelectedSilenceBoundary:
    center_ms: int
    start_ms: int
    end_ms: int
    reliable: bool
    score: float


def silence_boundary_edges(
    candidate: BoundaryCandidate,
    padding_ms: int = 0,
) -> tuple[int, int]:
    """Return ``(previous_end, following_start)`` for one silence interval.

    A cue ending before a pause belongs at the pause's leading edge, while a
    cue beginning after it belongs at the trailing edge.  Padding is kept
    inside the detected non-speech interval, never on the speech side and
    never with the two directions reversed.  Legacy point candidates retain
    the old symmetric-gap behaviour.
    """
    center = max(0, int(candidate.time_ms))
    start = center if candidate.start_ms is None else max(0, int(candidate.start_ms))
    end = center if candidate.end_ms is None else max(0, int(candidate.end_ms))
    if end < start:
        start, end = end, start
    padding = max(0, int(padding_ms))
    if end > start:
        inside = min(padding, (end - start) // 2)
        return start + inside, end - inside
    return max(0, center - padding), center + padding


@dataclass(slots=True)
class ProgressiveAlignmentEvaluation:
    """Quality signals used to grow a VAD/text block after forced alignment."""

    score: float
    accepted_count: int
    timed_token_ratio: float
    used_window_ratio: float
    touches_window_end: bool
    needs_more_audio: bool
    may_contain_more_text: bool


@dataclass(slots=True, frozen=True)
class ForcedAlignmentPlannerOptions:
    minimum_segments: int = 4
    # Sentence count is derived from estimated spoken duration.  This is only a
    # pathological safety ceiling for books split into many one- or two-word
    # fragments, not the normal block size.
    # The upstream Qwen long-audio recipe caps one call at twenty sentences.
    # Duration still chooses the normal end inside this model-capacity guard.
    maximum_segments: int = 20
    target_duration_ms: int = 80_000
    maximum_window_ms: int = 90_000
    # The next block repeats two source sentences and their proven audio.  The
    # older block remains their sole writer; the repeat is validation only.
    overlap_segments: int = 2
    stable_score: float = 0.72
    review_score: float = 0.50
    beam_width: int = 3
    maximum_backtracks: int = 2
    recovery_search_ms: int = 90_000
    recovery_attempts: int = 8
    leading_text_search_segments: int = 6
    initial_resync_pairs: int = 18
    candidate_trial_budget: int = 6
    # The following overlapping block is the real validator.  Once a local
    # candidate is already stable, running more alternatives only multiplies
    # GPU work and native-memory pressure on long audiobook chapters.
    stable_candidates_required: int = 2


@dataclass(slots=True)
class ForcedAlignmentHypothesis:
    start_index: int
    end_index: int
    audio_start_ms: int
    audio_end_ms: int
    segments: list[TextSegment]
    tokens: list[ASRToken]
    score: float
    reasons: tuple[str, ...] = ()
    weak_boundary: bool = False


@dataclass(slots=True)
class ForcedAlignmentBlockResult:
    start_index: int
    end_index: int
    status: str
    hypotheses: list[ForcedAlignmentHypothesis]
    selected: ForcedAlignmentHypothesis | None = None
    message: str = ""


@dataclass(slots=True)
class ForcedAlignmentRunResult:
    blocks: list[ForcedAlignmentBlockResult]
    completed_count: int
    review_count: int
    stopped_reason: str = ""


def forced_alignment_hypothesis_priority(
    hypothesis: ForcedAlignmentHypothesis,
    *,
    stable_score: float = 0.72,
) -> tuple[bool, float, int, float]:
    """Prefer complete stable blocks over higher-scoring partial recovery."""
    content = [segment for segment in hypothesis.segments if segment.text.strip()]
    coverage = sum(
        segment.end_ms > segment.start_ms for segment in content
    ) / max(1, len(content))
    return (
        hypothesis.score >= stable_score,
        coverage,
        hypothesis.end_index - hypothesis.start_index
        if hypothesis.score >= stable_score else 0,
        hypothesis.score,
    )


def forced_alignment_text_group_ends(
    segments: Sequence[TextSegment],
    start_index: int,
    *,
    language: str = "auto",
    observed_ms_per_unit: float | None = None,
    options: ForcedAlignmentPlannerOptions | None = None,
) -> list[int]:
    """Return nearby multi-sentence block endings, never crossing a lock."""
    options = options or ForcedAlignmentPlannerOptions()
    if not 0 <= start_index < len(segments):
        return []
    available_end = start_index
    estimated = 0
    preferred = start_index
    for index in range(start_index, min(len(segments), start_index + options.maximum_segments)):
        if segments[index].locked:
            break
        available_end = index + 1
        if segments[index].text.strip():
            estimated += estimate_spoken_duration_ms(
                segments[index].text, language, observed_ms_per_unit,
            )
        if preferred == start_index and (
            available_end - start_index >= options.minimum_segments
            and estimated >= options.target_duration_ms
        ):
            preferred = available_end
    if available_end <= start_index:
        return []
    if preferred == start_index:
        preferred = available_end
    minimum_end = min(available_end, start_index + options.minimum_segments)
    candidates = {
        max(minimum_end, min(available_end, preferred + offset))
        for offset in (-1, 0, 1)
    }
    candidates.add(minimum_end)
    return sorted(candidates, key=lambda end: (abs(end - preferred), end))


def forced_alignment_window_ends(
    start_ms: int,
    limit_ms: int,
    text: str,
    candidates: Sequence[BoundaryCandidate],
    *,
    language: str = "auto",
    observed_ms_per_unit: float | None = None,
    expected_duration_ms: int | None = None,
    options: ForcedAlignmentPlannerOptions | None = None,
) -> list[tuple[int, bool]]:
    """Plan strong-VAD-first audio ends followed by bounded weak fallbacks."""
    options = options or ForcedAlignmentPlannerOptions()
    start_ms = max(0, int(start_ms))
    limit_ms = max(start_ms, int(limit_ms))
    if limit_ms <= start_ms:
        return []
    # ``estimate_spoken_duration_ms`` is deliberately capped for one unusually
    # long sentence. A multi-sentence block must pass the sum of its sentence
    # estimates, otherwise every block longer than 60 seconds is planned with
    # a prematurely clipped VAD end.
    expected = (
        estimate_spoken_duration_ms(text, language, observed_ms_per_unit)
        if expected_duration_ms is None
        else max(1, int(expected_duration_ms))
    )
    expected = min(options.maximum_window_ms, max(4_000, expected))
    target = min(limit_ms, start_ms + expected)
    latest = min(limit_ms, start_ms + options.maximum_window_ms)
    earliest = min(latest, start_ms + max(2_000, round(expected * 0.55)))
    strong: list[tuple[int, bool]] = []
    for candidate in candidates:
        duration = max(0, (candidate.end_ms or candidate.time_ms) - (candidate.start_ms or candidate.time_ms))
        if duration < 500 and candidate.score < 0.78:
            continue
        end = int(candidate.end_ms if candidate.end_ms is not None else candidate.time_ms)
        if earliest <= end <= latest:
            strong.append((end, False))
    strong.sort(key=lambda item: (abs(item[0] - target), item[0]))
    target_fallback = min(latest, max(earliest, target))
    expanded_fallback = min(
        latest, max(earliest, start_ms + round(expected * 1.35)),
    )
    result: list[tuple[int, bool]] = []
    # The second attempt must expand materially. Trying three nearby VAD
    # boundaries before expanding exhausted the per-block budget whenever the
    # default speaking-rate estimate was 15–30% too fast.
    ordered = [
        *strong[:1],
        (expanded_fallback, True),
        *strong[1:3],
        (target_fallback, True),
        (latest, True),
    ]
    for item in ordered:
        if item[0] <= start_ms or any(abs(existing[0] - item[0]) < 250 for existing in result):
            continue
        result.append(item)
    return result[:5]


def score_forced_alignment_hypothesis(
    segments: Sequence[TextSegment],
    tokens: Sequence[ASRToken],
    *,
    audio_start_ms: int,
    audio_end_ms: int,
    candidates: Sequence[BoundaryCandidate] = (),
    language: str = "auto",
    observed_ms_per_unit: float | None = None,
    weak_boundary: bool = False,
) -> tuple[float, tuple[str, ...], float | None]:
    """Score structural/acoustic plausibility without trusting model confidence."""
    duration = max(1, audio_end_ms - audio_start_ms)
    timed = [token for token in tokens if token.end_ms > token.start_ms]
    if not timed or any(
        token.start_ms < 0 or token.end_ms < token.start_ms or token.end_ms > duration + 250
        for token in tokens
    ):
        return 0.0, ("无效或越界时间戳",), observed_ms_per_unit
    monotonic = all(left.end_ms <= right.end_ms for left, right in zip(timed, timed[1:]))
    if not monotonic:
        return 0.0, ("时间戳不单调",), observed_ms_per_unit
    aligned = [segment for segment in segments if segment.end_ms > segment.start_ms]
    if not aligned:
        return 0.0, ("没有可用句段时间",), observed_ms_per_unit

    reasons: list[str] = []
    score = 0.35
    expected_count = sum(bool(segment.text.strip()) for segment in segments)
    coverage = len(aligned) / max(1, expected_count)
    if coverage < 1.0:
        reasons.append("块内有未朗读或未能映射的文字")
    last_end = max(token.end_ms for token in timed)
    touches_end = last_end >= duration - 250
    if touches_end:
        reasons.append("块末尾可能被截断")
    else:
        score += 0.15
    used_ratio = last_end / duration
    if 0.35 <= used_ratio <= 0.98:
        score += 0.10
    else:
        reasons.append("音频窗口利用率异常")

    units = sum(spoken_unit_count(segment.text, language) for segment in aligned)
    measured_rate = (aligned[-1].end_ms - aligned[0].start_ms) / max(1, units)
    reference_rate = observed_ms_per_unit
    if reference_rate is None:
        reference_rate = 175.0 if (language or "").casefold() in {
            "zh", "yue", "ja", "ko", "chinese", "japanese", "korean", "cantonese",
        } else 360.0
    rate_ratio = measured_rate / max(1.0, reference_rate)
    if 0.55 <= rate_ratio <= 1.80:
        score += 0.20 * (1.0 - min(1.0, abs(1.0 - rate_ratio) / 0.80))
    else:
        reasons.append("朗读速度明显异常")

    absolute_end = audio_start_ms + last_end
    nearest = min((abs(candidate.time_ms - absolute_end) for candidate in candidates), default=10_000)
    if nearest <= 800:
        score += 0.10 * (1.0 - nearest / 1_000)
    elif weak_boundary:
        reasons.append("未落在强静音边界")

    rates = [
        (segment.end_ms - segment.start_ms) / max(1, spoken_unit_count(segment.text, language))
        for segment in aligned
    ]
    if rates:
        ordered = sorted(rates)
        median_rate = ordered[len(ordered) // 2]
        outliers = sum(rate < median_rate * 0.35 or rate > median_rate * 2.8 for rate in rates)
        if outliers == 0:
            score += 0.10
        else:
            reasons.append("句段时长与文字量不匹配")
    # A missing visual heading is tolerable, but a hypothesis that only maps a
    # small fraction of its text must not outrank a complete neighbouring one.
    score *= 0.65 + 0.35 * coverage
    if touches_end:
        # A forced aligner can assign the final supplied word to the clip edge
        # even when the actual utterance continues. Such a candidate may be
        # shown for review, but must not become the cursor for the next block.
        score = min(score, 0.69)
    return max(0.0, min(1.0, score)), tuple(reasons), measured_rate


def forced_alignment_overlap_score(
    previous: ForcedAlignmentHypothesis,
    current: ForcedAlignmentHypothesis,
    *,
    overlap_segments: int = 2,
) -> float:
    """Compare absolute times of positions shared by two adjacent blocks."""
    previous_by_position = {segment.position: segment for segment in previous.segments}
    current_by_position = {segment.position: segment for segment in current.segments}
    shared = sorted(set(previous_by_position) & set(current_by_position))[-max(1, overlap_segments):]
    if not shared:
        return 0.0
    scores = []
    for position in shared:
        left = previous_by_position[position]
        right = current_by_position[position]
        if left.end_ms <= left.start_ms or right.end_ms <= right.start_ms:
            # An unspoken visual heading or a locally omitted source word is
            # not evidence of time drift. A neighbouring timed overlap row
            # must still validate the cursor.
            continue
        tolerance = max(250, round(max(left.end_ms - left.start_ms, right.end_ms - right.start_ms) * 0.05))
        error = max(abs(left.start_ms - right.start_ms), abs(left.end_ms - right.end_ms))
        if error > tolerance:
            return 0.0
        scores.append(max(0.0, 1.0 - error / max(1, tolerance)))
    return mean(scores) if scores else 0.0


def forced_alignment_has_usable_coverage(
    segments: Sequence[TextSegment],
    *,
    minimum_ratio: float = 0.70,
    maximum_unmatched_suffix: int = 2,
) -> bool:
    """Distinguish a local source difference from truncated model output."""
    content = [segment for segment in segments if segment.text.strip()]
    if not content:
        return False
    timed_indexes = [
        index for index, segment in enumerate(content)
        if segment.end_ms > segment.start_ms
    ]
    if not timed_indexes:
        return False
    coverage = len(timed_indexes) / len(content)
    trailing_unmatched = len(content) - timed_indexes[-1] - 1
    return (
        coverage >= max(0.0, min(1.0, minimum_ratio))
        and trailing_unmatched <= max(0, int(maximum_unmatched_suffix))
    )


def forced_alignment_candidate_can_end_search(
    segments: Sequence[TextSegment],
    score: float,
    *,
    stable_score: float = 0.72,
) -> bool:
    """Only a fully covered stable candidate may stop alternative trials."""
    return score >= stable_score and all(
        not segment.text.strip() or segment.end_ms > segment.start_ms
        for segment in segments
    )


def partition_monotonic_alignment_rows(
    rows: Sequence[tuple[int, TextSegment]],
    *,
    tolerance_ms: int = 250,
) -> tuple[set[int], set[int]]:
    """Separate safe rows from time-regressive forced-alignment output.

    A model block may fail locally, but it must never move the chapter cursor
    back into already confirmed audio. Rejected rows do not advance the cursor,
    which allows a later independently confirmed block to recover.
    """
    accepted: set[int] = set()
    rejected: set[int] = set()
    cursor_ms = 0
    for row_index, segment in sorted(rows, key=lambda item: item[0]):
        if segment.end_ms <= segment.start_ms:
            rejected.add(row_index)
            continue
        if segment.start_ms < cursor_ms - max(0, int(tolerance_ms)):
            rejected.add(row_index)
            continue
        accepted.add(row_index)
        cursor_ms = max(cursor_ms, segment.end_ms)
    return accepted, rejected


def locate_contiguous_audio_part(parts: Sequence[tuple], milliseconds: int):
    """Locate a chapter-local media slice, including an exact next-part edge."""
    part = next(
        (item for item in parts if item[3] <= milliseconds < item[4]), None,
    )
    if part is not None:
        return part, milliseconds
    following = next((item for item in parts if item[3] >= milliseconds), None)
    if following is None:
        return None, milliseconds
    return following, following[3]


def spoken_unit_count(text: str, language: str = "auto") -> int:
    """Return a stable speaking-length unit count for adaptive VAD windows."""
    code = (language or "auto").casefold()
    if code in {"zh", "yue", "ja", "ko", "chinese", "japanese", "korean", "cantonese"}:
        return max(1, len(re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)))
    words = re.findall(r"[^\W_]+(?:[’'\-][^\W_]+)*", text, flags=re.UNICODE)
    return max(1, len(words))


def estimate_spoken_duration_ms(
    text: str,
    language: str = "auto",
    observed_ms_per_unit: float | None = None,
) -> int:
    """Estimate one sentence's duration without performing speech recognition.

    The estimate only chooses the first VAD-bounded trial window.  A forced
    aligner's returned end time, rather than this estimate, advances the next
    sentence.
    """
    code = (language or "auto").casefold()
    cjk = code in {"zh", "yue", "ja", "ko", "chinese", "japanese", "korean", "cantonese"}
    default_rate = 175.0 if cjk else 360.0
    minimum_rate, maximum_rate = ((70.0, 500.0) if cjk else (140.0, 1_000.0))
    rate = default_rate if observed_ms_per_unit is None else max(
        minimum_rate, min(maximum_rate, float(observed_ms_per_unit))
    )
    punctuation_pause = sum(text.count(mark) for mark in "，、；：,.!?。！？…") * 70
    duration = round(spoken_unit_count(text, language) * rate + punctuation_pause)
    return max(900, min(60_000, duration))


def progressive_vad_window_ends(
    start_ms: int,
    limit_ms: int,
    text: str,
    candidates: Sequence[BoundaryCandidate],
    *,
    language: str = "auto",
    observed_ms_per_unit: float | None = None,
    max_attempts: int = 4,
    max_window_ms: int = 90_000,
) -> list[int]:
    """Plan expanding VAD-bounded windows for one known-text sentence.

    The first trial deliberately ends at or after the expected speech length;
    later trials only expand forward.  A bounded fallback is retained for
    speech with no detectable pause, and never exceeds the model's safe input.
    """
    start_ms = max(0, int(start_ms))
    limit_ms = max(start_ms, int(limit_ms))
    if limit_ms <= start_ms:
        return []
    expected = estimate_spoken_duration_ms(text, language, observed_ms_per_unit)
    target = start_ms + expected + 600
    minimum_end = start_ms + max(750, round(expected * 0.55))
    hard_end = min(limit_ms, start_ms + max(8_000, min(max_window_ms, expected * 3 + 4_000)))
    times = sorted({
        int(candidate.time_ms) for candidate in candidates
        if minimum_end <= candidate.time_ms <= hard_end
    })
    at_or_after = [value for value in times if value >= target]
    selected = at_or_after[:max(1, max_attempts)]
    if not selected and times:
        selected = [times[-1]]
    # A sentence may contain no pause at all.  The bounded fallback also gives
    # the aligner one final expansion when the first VAD boundary was premature.
    if hard_end > start_ms and (not selected or hard_end - selected[-1] >= 500):
        selected.append(hard_end)
    return selected[:max(1, max_attempts)]


def progressive_vad_next_start(
    aligned_end_ms: int,
    candidates: Sequence[BoundaryCandidate],
    *,
    padding_ms: int = 80,
    max_gap_ms: int = 2_000,
    limit_ms: int | None = None,
) -> int:
    """Advance through the pause following a successfully aligned sentence."""
    aligned_end_ms = max(0, int(aligned_end_ms))
    nearby: list[BoundaryCandidate] = []
    for candidate in candidates:
        silence_start = candidate.start_ms if candidate.start_ms is not None else candidate.time_ms
        silence_end = candidate.end_ms if candidate.end_ms is not None else candidate.time_ms
        if silence_end < aligned_end_ms - 250:
            continue
        if silence_start > aligned_end_ms + max_gap_ms:
            continue
        nearby.append(candidate)
    if nearby:
        chosen = min(nearby, key=lambda item: (abs(item.time_ms - aligned_end_ms), -item.score))
        pause_end = chosen.end_ms if chosen.end_ms is not None else chosen.time_ms
        result = max(aligned_end_ms, pause_end + max(0, padding_ms))
    else:
        result = aligned_end_ms
    return min(result, limit_ms) if limit_ms is not None else result


def progressive_text_group_end(
    segments: Sequence[TextSegment],
    start_index: int,
    *,
    language: str = "auto",
    observed_ms_per_unit: float | None = None,
    minimum_segments: int = 2,
    maximum_segments: int = 6,
    target_duration_ms: int = 12_000,
) -> int:
    """Choose an initial multi-sentence text block for one VAD-bounded trial.

    A pause is not assumed to be a sentence boundary.  The block grows until
    it contains useful acoustic context, while locked segments remain hard
    boundaries.  The returned index is exclusive.
    """
    start_index = max(0, int(start_index))
    if start_index >= len(segments):
        return start_index
    minimum_segments = max(1, int(minimum_segments))
    maximum_segments = max(minimum_segments, int(maximum_segments))
    duration = 0
    count = 0
    end_index = start_index
    for index in range(start_index, min(len(segments), start_index + maximum_segments)):
        segment = segments[index]
        if segment.locked:
            break
        end_index = index + 1
        if segment.text.strip():
            count += 1
            duration += estimate_spoken_duration_ms(
                segment.text, language, observed_ms_per_unit,
            )
        if count >= minimum_segments and duration >= max(1_000, int(target_duration_ms)):
            break
    return end_index


def evaluate_progressive_alignment(
    segments: Sequence[TextSegment],
    tokens: Sequence[ASRToken],
    window_duration_ms: int,
) -> ProgressiveAlignmentEvaluation:
    """Evaluate a multi-sentence forced-alignment result before committing it."""
    window_duration_ms = max(1, int(window_duration_ms))
    accepted_count = 0
    confidences: list[float] = []
    for segment in segments:
        if (
            segment.status == SegmentStatus.UNMATCHED
            or segment.end_ms <= segment.start_ms
        ):
            break
        accepted_count += 1
        confidences.append(max(0.0, min(1.0, segment.confidence)))
    timed_tokens = [token for token in tokens if token.end_ms > token.start_ms]
    timed_ratio = len(timed_tokens) / max(1, len(tokens))
    last_end = max((token.end_ms for token in tokens), default=0)
    used_ratio = max(0.0, min(1.0, last_end / window_duration_ms))
    touches_end = bool(tokens) and last_end >= window_duration_ms - 220
    accepted_ratio = accepted_count / max(1, len(segments))
    mean_confidence = mean(confidences) if confidences else 0.0
    score = (
        accepted_ratio * 0.55
        + mean_confidence * 0.25
        + timed_ratio * 0.15
        + min(1.0, used_ratio / 0.72) * 0.05
    )
    if touches_end:
        score -= 0.08
    needs_more_audio = accepted_count < len(segments) or touches_end
    unused_ms = max(0, window_duration_ms - last_end)
    may_contain_more_text = (
        accepted_count == len(segments)
        and unused_ms >= max(2_000, round(window_duration_ms * 0.28))
    )
    return ProgressiveAlignmentEvaluation(
        max(0.0, min(1.0, score)),
        accepted_count,
        timed_ratio,
        used_ratio,
        touches_end,
        needs_more_audio,
        may_contain_more_text,
    )


def segments_from_asr_tokens(
    tokens: Sequence[ASRToken],
    *,
    chapter_id: int,
    language: str = "auto",
    max_duration_ms: int = 12_000,
    pause_boundary_ms: int = 700,
    restore_punctuation: bool = True,
) -> list[TextSegment]:
    """Create editable, punctuated sentences when audio has no source text."""
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
        text = _join_asr_token_text(group, language, pause_commas=restore_punctuation)
        if restore_punctuation:
            text = _restore_asr_punctuation(text, group, language)
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


def _join_asr_token_text(
    tokens: Sequence[ASRToken],
    language: str = "auto",
    *,
    pause_commas: bool = False,
) -> str:
    """Join word timestamps without destroying spaces from word-based ASR backends."""
    if not tokens:
        return ""
    pieces = [str(token.text) for token in tokens]
    if pause_commas and len(tokens) > 1:
        sample = "".join(pieces)
        comma = "，" if _asr_uses_cjk_punctuation(sample, language) else ","
        for index, (left, right) in enumerate(zip(tokens, tokens[1:])):
            gap = right.start_ms - left.end_ms
            stripped = pieces[index].rstrip()
            next_text = pieces[index + 1].lstrip()
            if (
                420 <= gap < 700 and stripped and next_text
                and stripped[-1] not in ",，.!?。！？;；:：…"
                and next_text[0] not in ",，.!?。！？;；:：…"
            ):
                trailing = pieces[index][len(stripped):]
                pieces[index] = stripped + comma + trailing
    # Whisper word tokens normally carry a leading space. Preserve that output
    # exactly; Qwen word timestamps commonly do not, so infer word boundaries.
    if any(piece[:1].isspace() or piece[-1:].isspace() for piece in pieces):
        return "".join(pieces).strip()
    compact_languages = {"zh", "zho", "cmn", "yue", "ja", "jpn", "ko", "kor"}
    language_code = (language or "auto").casefold().split("-", 1)[0]
    contains_cjk = any("\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff" for piece in pieces for char in piece)
    if language_code in compact_languages or (language_code == "auto" and contains_cjk):
        return "".join(pieces).strip()
    no_space_before = set(",.!?;:%)]}»”’…")
    no_space_after = set("([{«“‘¿¡")
    result = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not result or piece[0] in no_space_before or result[-1] in no_space_after:
            result += piece
        elif piece.startswith(("'s", "'re", "'ve", "'ll", "'d", "'m", "n't")):
            result += piece
        else:
            result += " " + piece
    return result.strip()


def _asr_uses_cjk_punctuation(text: str, language: str) -> bool:
    language_code = (language or "auto").casefold().split("-", 1)[0]
    if language_code in {"zh", "zho", "cmn", "yue", "ja", "jpn", "ko", "kor"}:
        return True
    return language_code == "auto" and any(
        "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
        for char in text
    )


def _restore_asr_punctuation(
    text: str,
    tokens: Sequence[ASRToken],
    language: str,
) -> str:
    """Add conservative pause commas and a sentence terminator to raw ASR text."""
    if not text:
        return text
    cjk = _asr_uses_cjk_punctuation(text, language)
    comma = "，" if cjk else ","
    terminal = "。" if cjk else "."
    stripped = text.rstrip()
    closers = "”’\"'」』】》）)]}»"
    core = stripped.rstrip(closers).rstrip()
    closing = stripped[len(core):]
    if core and core[-1] not in ".。!?！？…;；":
        stripped = core + terminal + closing
    return stripped


def _token_character_map(tokens: Sequence[ASRToken], language: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    owner: list[int] = []
    for index, token in enumerate(tokens):
        normalized = normalize_for_match(token.text, language)
        characters.extend(normalized)
        owner.extend([index] * len(normalized))
    return "".join(characters), owner


def _uses_character_alignment(language: str, sample: str) -> bool:
    code = (language or "auto").casefold().split("-", 1)[0]
    if code in {"zh", "yue", "ja", "ko"}:
        return True
    if code not in {"", "auto"}:
        return False
    return is_cjk_text(sample)


def _alignment_units(text: str, language: str, *, character_mode: bool) -> list[str]:
    """Create matching units without erasing word or segment boundaries.

    ``normalize_for_match`` deliberately removes spaces. Concatenating its
    output for a whole chapter used to turn adjacent English source segments
    such as ``II`` + ``I`` into ``III`` and let one omitted heading shift every
    later sentence. Space languages therefore match whole lexical units; CJK
    languages retain their useful character-level behaviour.
    """
    if character_mode:
        return list(normalize_for_match(text, language))
    words = re.findall(r"[^\W_]+(?:[’'\-][^\W_]+)*", text, flags=re.UNICODE)
    return [unit for word in words if (unit := normalize_for_match(word, language))]


def _alignment_units_with_spans(
    text: str,
    language: str,
    *,
    character_mode: bool,
) -> list[tuple[str, int, int]]:
    if character_mode:
        result: list[tuple[str, int, int]] = []
        for index, character in enumerate(text):
            result.extend(
                (unit, index, index + 1)
                for unit in normalize_for_match(character, language)
            )
        return result
    result = []
    for match in re.finditer(r"[^\W_]+(?:[’'\-][^\W_]+)*", text, flags=re.UNICODE):
        unit = normalize_for_match(match.group(), language)
        if unit:
            result.append((unit, match.start(), match.end()))
    return result


def _transcript_units_with_owners(
    tokens: Sequence[ASRToken],
    source_unit_groups: Sequence[Sequence[str]],
    language: str,
    *,
    character_mode: bool,
) -> tuple[list[str], list[int]]:
    """Normalize model tokens while undoing punctuation-fused Latin words.

    Qwen removes em dashes from its returned token text and can consequently
    emit ``Owenunfortunately`` for source ``Owen—unfortunately``.  Matching
    that as one lexical unit drops both source words and moves the sentence
    start to the next token.  Split only when the fused value exactly equals a
    short *contiguous* sequence of source units, so this remains monotonic and
    language-independent rather than becoming an English abbreviation rule.
    """
    source_units = [unit for group in source_unit_groups for unit in group]
    fused_splits: dict[str, tuple[str, ...]] = {}
    source_vocabulary = set(source_units)
    if not character_mode:
        # Never join across a source segment boundary. Otherwise a transcript
        # heading ``III`` could be falsely decomposed into adjacent source rows
        # ``II`` + ``I`` and shift the rest of the chapter.
        for group in source_unit_groups:
            for start in range(len(group)):
                combined = group[start]
                for end in range(start + 1, min(len(group), start + 6)):
                    combined += group[end]
                    if combined not in source_vocabulary:
                        fused_splits.setdefault(combined, tuple(group[start:end + 1]))

    transcript_units: list[str] = []
    owners: list[int] = []
    for token_index, token in enumerate(tokens):
        units = _alignment_units(token.text, language, character_mode=character_mode)
        if len(units) == 1 and units[0] in fused_splits:
            units = list(fused_splits[units[0]])
        transcript_units.extend(units)
        owners.extend([token_index] * len(units))
    return transcript_units, owners


def _ordered_unit_matches(
    source_units: Sequence[str], transcript_units: Sequence[str],
) -> dict[int, int]:
    """Return conservative monotonic matches that tolerate local insertions.

    Long matching runs are reliable anchors. A one-unit run is accepted only
    when that unit is unique on both sides (or sufficiently distinctive), so a
    stray common word cannot connect two unrelated passages. An unmatched
    source sentence consumes no transcript units, leaving later anchors intact.
    """
    matcher = SequenceMatcher(None, source_units, transcript_units, autojunk=False)
    source_counts: dict[str, int] = {}
    transcript_counts: dict[str, int] = {}
    for unit in source_units:
        source_counts[unit] = source_counts.get(unit, 0) + 1
    for unit in transcript_units:
        transcript_counts[unit] = transcript_counts.get(unit, 0) + 1
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        if block.size <= 0:
            continue
        if block.size == 1:
            unit = source_units[block.a]
            unique = source_counts.get(unit, 0) == transcript_counts.get(unit, 0) == 1
            if not unique and len(unit) < 5:
                continue
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping


def anchors_from_segments_tokens(
    segments: Sequence[TextSegment],
    tokens: Sequence[ASRToken],
    *,
    language: str = "auto",
) -> list[TextAudioAnchor]:
    """Map original source character ranges to matched ASR word times."""
    if not segments or not tokens:
        return []
    sample = "".join(segment.text for segment in segments) + "".join(token.text for token in tokens)
    character_mode = _uses_character_alignment(language, sample)
    source_units: list[str] = []
    source_unit_groups: list[list[str]] = []
    source_spans: list[tuple[int, int]] = []
    segment_ranges: list[tuple[int, int, TextSegment]] = []
    raw_offset = 0
    for segment in segments:
        start = raw_offset
        segment_units: list[str] = []
        for unit, local_start, local_end in _alignment_units_with_spans(
            segment.text, language, character_mode=character_mode,
        ):
            source_units.append(unit)
            segment_units.append(unit)
            source_spans.append((raw_offset + local_start, raw_offset + local_end))
        source_unit_groups.append(segment_units)
        raw_offset += len(segment.text)
        segment_ranges.append((start, raw_offset, segment))
    transcript_units, transcript_owner = _transcript_units_with_owners(
        tokens, source_unit_groups, language, character_mode=character_mode,
    )
    mapping = _ordered_unit_matches(source_units, transcript_units)
    token_source_positions: dict[int, list[int]] = {}
    for source_index, transcript_index in mapping.items():
        if source_index >= len(source_spans) or transcript_index >= len(transcript_owner):
            continue
        span_start, span_end = source_spans[source_index]
        positions = token_source_positions.setdefault(transcript_owner[transcript_index], [])
        positions.extend((span_start, max(span_start, span_end - 1)))
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
    """Monotonically align source segments while containing local differences.

    Space languages use lexical units, so an extra word or source-only sentence
    cannot be concatenated across segment boundaries and shift the remainder of
    a chapter. CJK languages use normalized characters. Original text is never
    changed.
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
    sample = "".join(segment.text for segment in segments) + "".join(token.text for token in tokens)
    character_mode = _uses_character_alignment(language, sample)
    source_parts = [
        _alignment_units(segment.text, language, character_mode=character_mode)
        for segment in segments
    ]
    offsets: list[int] = [0]
    for part in source_parts:
        offsets.append(offsets[-1] + len(part))
    source_units = [unit for part in source_parts for unit in part]
    transcript_units, transcript_owner = _transcript_units_with_owners(
        tokens, source_parts, language, character_mode=character_mode,
    )
    mapping = _ordered_unit_matches(source_units, transcript_units)

    result: list[TextSegment] = []
    last_end = 0
    for index, segment in enumerate(segments):
        if segment.locked:
            result.append(segment)
            last_end = max(last_end, segment.end_ms)
            continue
        start_offset, end_offset = offsets[index], offsets[index + 1]
        matched_units = [mapping[pos] for pos in range(start_offset, end_offset) if pos in mapping]
        coverage = len(matched_units) / max(1, end_offset - start_offset)
        if not matched_units or coverage < 0.18:
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
        first_unit, last_unit = min(matched_units), max(matched_units)
        first_token = transcript_owner[first_unit]
        last_token = transcript_owner[last_unit]
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
    ordered = sorted(candidates, key=lambda candidate: candidate.time_ms)
    times = [candidate.time_ms for candidate in ordered]
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
        if (
            left.locked or right.locked
            or left.end_ms <= left.start_ms or right.end_ms <= right.start_ms
        ):
            continue
        target = (left.end_ms + right.start_ms) // 2 if right.start_ms else left.end_ms
        insertion = bisect_right(times, target)
        nearby = ordered[max(0, insertion - 3): insertion + 3]
        if not nearby:
            continue
        def distance(candidate: BoundaryCandidate) -> int:
            start = candidate.time_ms if candidate.start_ms is None else candidate.start_ms
            end = candidate.time_ms if candidate.end_ms is None else candidate.end_ms
            if end < start:
                start, end = end, start
            if target < start:
                return start - target
            if target > end:
                return target - end
            return 0

        selected = min(nearby, key=lambda candidate: (distance(candidate), -candidate.score))
        if distance(selected) > window_ms:
            continue
        previous_end, following_start = silence_boundary_edges(selected, padding_ms)
        left.end_ms = max(left.start_ms, min(previous_end, right.end_ms))
        right.start_ms = min(right.end_ms, max(left.end_ms, following_start))
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
) -> list[_SelectedSilenceBoundary]:
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
    available = sorted(
        (item for item in candidates if start_ms < item.time_ms < end_ms),
        key=lambda item: item.time_ms,
    )
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
            _SelectedSilenceBoundary(
                item.time_ms,
                item.time_ms if item.start_ms is None else item.start_ms,
                item.time_ms if item.end_ms is None else item.end_ms,
                abs(item.time_ms - target) <= tolerance and item.score >= 0.25,
                item.score,
            )
            for item, target in zip(chosen, targets)
        ]

    # With too few pauses, reserve each real candidate for its nearest target
    # and estimate the rest. A monotonic clamp prevents crossed boundaries.
    output: list[_SelectedSilenceBoundary | None] = [None] * needed
    # Place real pauses in chronological order.  Restrict each assignment to
    # the range that still leaves one target slot for every later pause.
    previous_target = -1
    for candidate_index, candidate in enumerate(available):
        first_target = previous_target + 1
        last_target = needed - (len(available) - candidate_index)
        target_index = min(
            range(first_target, last_target + 1),
            key=lambda index: abs(targets[index] - candidate.time_ms),
        )
        output[target_index] = _SelectedSilenceBoundary(
            candidate.time_ms,
            candidate.time_ms if candidate.start_ms is None else candidate.start_ms,
            candidate.time_ms if candidate.end_ms is None else candidate.end_ms,
            True,
            candidate.score,
        )
        previous_target = target_index

    # Fill only the missing slots.  Estimated points are constrained between
    # the neighbouring real VAD centres; the real intervals are never moved.
    anchors = [(-1, start_ms)] + [
        (index, boundary.center_ms)
        for index, boundary in enumerate(output) if boundary is not None
    ] + [(needed, end_ms)]
    for (left_index, left_time), (right_index, right_time) in zip(anchors, anchors[1:]):
        cursor = left_time
        for index in range(left_index + 1, right_index):
            remaining = right_index - index - 1
            value = max(cursor + 1, min(right_time - remaining - 1, targets[index]))
            output[index] = _SelectedSilenceBoundary(value, value, value, False, 0.35)
            cursor = value
    return [boundary for boundary in output if boundary is not None]


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
        for boundary in boundaries:
            if boundary.reliable:
                used_times.add(boundary.center_ms)
        unreliable_streak = 0
        for local_index, segment in enumerate(run):
            if local_index == 0:
                left = cursor_time
            else:
                previous = boundaries[local_index - 1]
                _previous_end, left = silence_boundary_edges(
                    BoundaryCandidate(
                        previous.center_ms, previous.score,
                        start_ms=previous.start_ms, end_ms=previous.end_ms,
                    ),
                    options.padding_ms,
                )
            if local_index == len(run) - 1:
                right = run_end_ms
            else:
                following = boundaries[local_index]
                right, _following_start = silence_boundary_edges(
                    BoundaryCandidate(
                        following.center_ms, following.score,
                        start_ms=following.start_ms, end_ms=following.end_ms,
                    ),
                    options.padding_ms,
                )
            segment.start_ms = max(cursor_time, min(run_end_ms, left))
            segment.end_ms = max(segment.start_ms, min(run_end_ms, right))
            neighbouring = []
            if local_index > 0:
                neighbouring.append(boundaries[local_index - 1])
            if local_index < len(boundaries):
                neighbouring.append(boundaries[local_index])
            reliable = all(item.reliable for item in neighbouring) if neighbouring else True
            unreliable_streak = 0 if reliable else unreliable_streak + 1
            if unreliable_streak > 3:
                original = segments[cursor_index + local_index]
                segment.start_ms = original.start_ms
                segment.end_ms = original.end_ms
                segment.confidence = original.confidence
                segment.status = original.status
                stopped_early = True
                break
            score = mean(item.score for item in neighbouring) if neighbouring else 0.7
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

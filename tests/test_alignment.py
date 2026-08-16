from __future__ import annotations

import unittest

from audioalign.core.alignment import (
    align_segments_from_silence,
    anchors_from_segments_tokens,
    align_segments_to_tokens,
    evaluate_progressive_alignment,
    ForcedAlignmentHypothesis,
    ForcedAlignmentPlannerOptions,
    forced_alignment_overlap_score,
    forced_alignment_text_group_ends,
    forced_alignment_window_ends,
    locate_contiguous_audio_part,
    progressive_vad_next_start,
    progressive_vad_window_ends,
    progressive_text_group_end,
    score_forced_alignment_hypothesis,
    segments_from_asr_tokens,
    snap_boundaries,
)
from audioalign.core.models import ASRToken, BoundaryCandidate, SegmentStatus, SilenceAlignmentOptions, TextSegment


class AlignmentTests(unittest.TestCase):
    def test_align_source_text_to_asr_words(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "你好世界。"),
            TextSegment(None, 1, 1, "这是测试。"),
        ]
        tokens = [
            ASRToken(None, 1, 0, "你好", 100, 500, 0.95),
            ASRToken(None, 1, 1, "世界", 520, 900, 0.95),
            ASRToken(None, 1, 2, "这是", 1200, 1500, 0.9),
            ASRToken(None, 1, 3, "测试", 1520, 1900, 0.9),
        ]
        result = align_segments_to_tokens(segments, tokens, language="zh")
        self.assertEqual((100, 900), (result[0].start_ms, result[0].end_ms))
        self.assertEqual((1200, 1900), (result[1].start_ms, result[1].end_ms))
        self.assertNotEqual(SegmentStatus.UNMATCHED, result[0].status)

    def test_audio_only_tokens_create_sentences(self) -> None:
        tokens = [
            ASRToken(None, 4, 0, "Hola ", 0, 300, 0.9),
            ASRToken(None, 4, 1, "mundo.", 320, 700, 0.9),
            ASRToken(None, 4, 2, "Otra ", 1600, 1900, 0.8),
            ASRToken(None, 4, 3, "frase.", 1910, 2300, 0.8),
        ]
        result = segments_from_asr_tokens(tokens, chapter_id=4)
        self.assertEqual(2, len(result))
        self.assertEqual("Hola mundo.", result[0].text)
        self.assertEqual(1600, result[1].start_ms)

    def test_audio_only_qwen_word_tokens_restore_english_spaces(self) -> None:
        tokens = [
            ASRToken(None, 4, 0, "Harry", 0, 200, 0.9),
            ASRToken(None, 4, 1, "Potter", 210, 450, 0.9),
            ASRToken(None, 4, 2, "was", 460, 590, 0.9),
            ASRToken(None, 4, 3, "here", 600, 800, 0.9),
            ASRToken(None, 4, 4, ".", 800, 820, 0.9),
        ]
        result = segments_from_asr_tokens(tokens, chapter_id=4, language="en")
        self.assertEqual("Harry Potter was here.", result[0].text)

    def test_audio_only_cjk_tokens_remain_compact(self) -> None:
        tokens = [
            ASRToken(None, 4, 0, "吾輩", 0, 200, 0.9),
            ASRToken(None, 4, 1, "は", 210, 260, 0.9),
            ASRToken(None, 4, 2, "猫", 270, 400, 0.9),
            ASRToken(None, 4, 3, "。", 400, 420, 0.9),
        ]
        result = segments_from_asr_tokens(tokens, chapter_id=4, language="ja")
        self.assertEqual("吾輩は猫。", result[0].text)

    def test_audio_only_transcript_gets_conservative_punctuation(self) -> None:
        tokens = [
            ASRToken(None, 4, 0, "This", 0, 180, 0.9),
            ASRToken(None, 4, 1, "is", 200, 300, 0.9),
            ASRToken(None, 4, 2, "a", 760, 800, 0.9),
            ASRToken(None, 4, 3, "test", 820, 1100, 0.9),
        ]
        result = segments_from_asr_tokens(tokens, chapter_id=4, language="en")
        self.assertEqual("This is, a test.", result[0].text)

    def test_audio_only_punctuation_can_be_disabled(self) -> None:
        tokens = [ASRToken(None, 4, 0, "unpunctuated", 0, 500, 0.9)]
        result = segments_from_asr_tokens(
            tokens, chapter_id=4, language="en", restore_punctuation=False,
        )
        self.assertEqual("unpunctuated", result[0].text)

    def test_snap_preserves_locked_boundary(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "a", 0, 1000, 1, SegmentStatus.LOCKED, True),
            TextSegment(None, 1, 1, "b", 1000, 2000, 1, SegmentStatus.AUTO),
        ]
        result = snap_boundaries(segments, [BoundaryCandidate(850, 1.0)], window_ms=500)
        self.assertEqual(1000, result[0].end_ms)
        self.assertEqual(1000, result[1].start_ms)

    def test_word_anchors_point_back_to_original_characters(self) -> None:
        segments = [TextSegment(10, 1, 0, "Hola, mundo.", 100, 900)]
        tokens = [
            ASRToken(None, 1, 0, "Hola", 100, 400, 0.95),
            ASRToken(None, 1, 1, "mundo", 500, 900, 0.9),
        ]
        anchors = anchors_from_segments_tokens(segments, tokens, language="es")
        self.assertEqual(2, len(anchors))
        self.assertEqual((0, 4, 100, 400), (anchors[0].source_start_char, anchors[0].source_end_char, anchors[0].start_ms, anchors[0].end_ms))
        self.assertEqual("mundo", segments[0].text[anchors[1].source_start_char:anchors[1].source_end_char])

    def test_silence_only_alignment_uses_anchor_and_preserves_lock(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "短句。"),
            TextSegment(None, 1, 1, "这是明显更长的一句话。"),
            TextSegment(None, 1, 2, "锁定", 5000, 6000, 1.0, SegmentStatus.LOCKED, True),
        ]
        candidates = [
            BoundaryCandidate(1200, 0.9, start_ms=1050, end_ms=1350),
            BoundaryCandidate(2400, 0.8, start_ms=2250, end_ms=2550),
        ]
        result = align_segments_from_silence(
            segments, candidates, SilenceAlignmentOptions(0, 200, 6000, padding_ms=50)
        )
        self.assertLess(result.segments[0].end_ms, result.segments[1].start_ms)
        self.assertEqual((5000, 6000), (result.segments[2].start_ms, result.segments[2].end_ms))
        self.assertTrue(result.segments[2].locked)
        self.assertGreaterEqual(len(result.anchors), 3)

    def test_progressive_vad_windows_expand_only_forward(self) -> None:
        candidates = [
            BoundaryCandidate(1_200, 0.8, start_ms=1_050, end_ms=1_350),
            BoundaryCandidate(2_600, 0.9, start_ms=2_450, end_ms=2_750),
            BoundaryCandidate(4_400, 0.7, start_ms=4_250, end_ms=4_550),
            BoundaryCandidate(7_000, 0.9, start_ms=6_850, end_ms=7_150),
        ]
        ends = progressive_vad_window_ends(
            1_000, 10_000, "This is one sentence.", candidates, language="en",
        )
        self.assertEqual(sorted(ends), ends)
        self.assertGreaterEqual(ends[0], 4_400)
        self.assertLessEqual(ends[-1] - 1_000, 90_000)

    def test_progressive_vad_cursor_advances_past_following_pause(self) -> None:
        candidates = [
            BoundaryCandidate(3_100, 0.95, start_ms=3_000, end_ms=3_300),
            BoundaryCandidate(8_000, 0.8, start_ms=7_800, end_ms=8_200),
        ]
        self.assertEqual(
            3_380,
            progressive_vad_next_start(2_950, candidates, padding_ms=80),
        )

    def test_progressive_alignment_starts_with_multiple_sentences(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "Chapter one."),
            TextSegment(None, 1, 1, "A short title."),
            TextSegment(None, 1, 2, "This longer sentence supplies enough spoken context for alignment."),
            TextSegment(None, 1, 3, "The next sentence should remain available for a retry."),
        ]
        end = progressive_text_group_end(
            segments, 0, language="en", target_duration_ms=5_000,
        )
        self.assertGreaterEqual(end, 2)
        self.assertLessEqual(end, len(segments))

    def test_progressive_alignment_group_stops_before_locked_segment(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "First sentence."),
            TextSegment(None, 1, 1, "Locked sentence.", 1000, 2000, locked=True),
            TextSegment(None, 1, 2, "Later sentence."),
        ]
        self.assertEqual(1, progressive_text_group_end(segments, 0, language="en"))

    def test_progressive_evaluation_requests_more_text_for_unused_audio(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "First.", 200, 1000, 0.9, SegmentStatus.AUTO),
            TextSegment(None, 1, 1, "Second.", 1100, 2000, 0.9, SegmentStatus.AUTO),
        ]
        tokens = [
            ASRToken(None, 1, 0, "First", 200, 1000, 0.9),
            ASRToken(None, 1, 1, "Second", 1100, 2000, 0.9),
        ]
        evaluation = evaluate_progressive_alignment(segments, tokens, 8_000)
        self.assertEqual(2, evaluation.accepted_count)
        self.assertTrue(evaluation.may_contain_more_text)
        self.assertFalse(evaluation.needs_more_audio)

    def test_progressive_evaluation_expands_clipped_audio(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "First.", 100, 1500, 0.9, SegmentStatus.AUTO),
            TextSegment(None, 1, 1, "Second.", 1500, 3980, 0.9, SegmentStatus.AUTO),
        ]
        tokens = [
            ASRToken(None, 1, 0, "First", 100, 1500, 0.9),
            ASRToken(None, 1, 1, "Second", 1500, 3980, 0.9),
        ]
        evaluation = evaluate_progressive_alignment(segments, tokens, 4_000)
        self.assertTrue(evaluation.touches_window_end)
        self.assertTrue(evaluation.needs_more_audio)

    def test_forced_block_planner_uses_multiple_sentences_and_stops_at_lock(self) -> None:
        segments = [
            TextSegment(None, 1, index, f"Sentence number {index} has enough words for context.")
            for index in range(8)
        ]
        segments[6].locked = True
        ends = forced_alignment_text_group_ends(
            segments, 0, language="en",
            options=ForcedAlignmentPlannerOptions(target_duration_ms=4_000),
        )
        self.assertTrue(ends)
        self.assertTrue(all(4 <= end <= 6 for end in ends))

    def test_forced_audio_windows_prefer_strong_vad_and_are_bounded(self) -> None:
        candidates = [
            BoundaryCandidate(28_000, 0.9, start_ms=27_300, end_ms=28_200),
            BoundaryCandidate(31_000, 0.4, start_ms=30_800, end_ms=31_100),
        ]
        windows = forced_alignment_window_ends(
            0, 180_000, "one two three four five six seven eight nine ten",
            candidates, language="en", observed_ms_per_unit=2_700,
        )
        self.assertTrue(windows)
        self.assertEqual((28_200, False), windows[0])
        self.assertTrue(all(end <= 75_000 for end, _weak in windows))

    def test_forced_score_uses_timing_plausibility_not_token_probability(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "one two three four", 200, 1600),
            TextSegment(None, 1, 1, "five six seven eight", 1700, 3200),
        ]
        tokens = [
            ASRToken(None, 1, 0, "one", 200, 800, 0.0),
            ASRToken(None, 1, 1, "eight", 2600, 3200, 0.0),
        ]
        score, reasons, measured = score_forced_alignment_hypothesis(
            segments, tokens, audio_start_ms=0, audio_end_ms=4_000, language="en",
            observed_ms_per_unit=400,
        )
        self.assertGreaterEqual(score, 0.50)
        self.assertIsNotNone(measured)
        self.assertNotIn("无效或越界时间戳", reasons)

    def test_forced_overlap_rejects_one_sentence_drift(self) -> None:
        def hypothesis(start: int, shift: int) -> ForcedAlignmentHypothesis:
            block = [
                TextSegment(
                    None, 1, start + offset, str(start + offset),
                    1_000 * (start + offset) + shift,
                    1_000 * (start + offset) + 800 + shift,
                )
                for offset in range(3)
            ]
            return ForcedAlignmentHypothesis(start, start + 3, 0, 10_000, block, [], 0.8)

        self.assertGreater(
            forced_alignment_overlap_score(hypothesis(0, 0), hypothesis(1, 100)), 0,
        )
        self.assertEqual(
            0.0, forced_alignment_overlap_score(hypothesis(0, 0), hypothesis(1, 1_000)),
        )

    def test_contiguous_media_boundary_selects_following_part(self) -> None:
        first = ("first", None, "a", 0, 10_000)
        second = ("second", None, "b", 10_000, 20_000)
        part, cursor = locate_contiguous_audio_part([first, second], 10_000)
        self.assertIs(second, part)
        self.assertEqual(10_000, cursor)


if __name__ == "__main__":
    unittest.main()

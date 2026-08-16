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
    forced_alignment_has_usable_coverage,
    forced_alignment_candidate_can_end_search,
    forced_alignment_hypothesis_priority,
    forced_alignment_text_group_ends,
    forced_alignment_window_ends,
    locate_contiguous_audio_part,
    partition_monotonic_alignment_rows,
    progressive_vad_next_start,
    progressive_vad_window_ends,
    progressive_text_group_end,
    score_forced_alignment_hypothesis,
    segments_from_asr_tokens,
    snap_boundaries,
)
from audioalign.core.models import ASRToken, BoundaryCandidate, SegmentStatus, SilenceAlignmentOptions, TextSegment


class AlignmentTests(unittest.TestCase):
    def test_forced_planner_uses_duration_blocks_with_model_capacity_guard(self) -> None:
        options = ForcedAlignmentPlannerOptions()
        self.assertEqual(2, options.stable_candidates_required)
        self.assertLessEqual(options.candidate_trial_budget, 6)
        self.assertEqual(2, options.overlap_segments)
        self.assertEqual(20, options.maximum_segments)
        self.assertEqual(80_000, options.target_duration_ms)
        self.assertEqual(90_000, options.maximum_window_ms)

    def test_forced_text_group_size_is_derived_from_estimated_duration(self) -> None:
        short = [TextSegment(None, 1, index, "Yes.") for index in range(40)]
        long = [
            TextSegment(None, 1, index, "This sentence contains enough words to take much longer to read aloud.")
            for index in range(40)
        ]
        options = ForcedAlignmentPlannerOptions(target_duration_ms=12_000)

        short_end = forced_alignment_text_group_ends(
            short, 0, language="en", options=options,
        )[0]
        long_end = forced_alignment_text_group_ends(
            long, 0, language="en", options=options,
        )[0]

        self.assertGreater(short_end, long_end)

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

    def test_extra_source_heading_does_not_shift_following_english_sentences(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "Chapter two"),
            TextSegment(None, 1, 1, "III"),
            TextSegment(None, 1, 2, "It was a cold and rainy night."),
            TextSegment(None, 1, 3, "They went home before midnight."),
        ]
        tokens = [
            ASRToken(None, 1, 0, "Chapter", 100, 300, 0.9),
            ASRToken(None, 1, 1, "two", 310, 500, 0.9),
            ASRToken(None, 1, 2, "It", 700, 780, 0.9),
            ASRToken(None, 1, 3, "was", 790, 900, 0.9),
            ASRToken(None, 1, 4, "a", 910, 950, 0.9),
            ASRToken(None, 1, 5, "cold", 960, 1100, 0.9),
            ASRToken(None, 1, 6, "and", 1110, 1200, 0.9),
            ASRToken(None, 1, 7, "rainy", 1210, 1400, 0.9),
            ASRToken(None, 1, 8, "night", 1410, 1600, 0.9),
            ASRToken(None, 1, 9, "They", 1900, 2050, 0.9),
            ASRToken(None, 1, 10, "went", 2060, 2200, 0.9),
            ASRToken(None, 1, 11, "home", 2210, 2360, 0.9),
            ASRToken(None, 1, 12, "before", 2370, 2550, 0.9),
            ASRToken(None, 1, 13, "midnight", 2560, 2800, 0.9),
        ]

        result = align_segments_to_tokens(segments, tokens, language="en")

        self.assertEqual(SegmentStatus.UNMATCHED, result[1].status)
        self.assertEqual((700, 1600), (result[2].start_ms, result[2].end_ms))
        self.assertEqual((1900, 2800), (result[3].start_ms, result[3].end_ms))

    def test_adjacent_roman_numeral_segments_are_not_concatenated(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "II"),
            TextSegment(None, 1, 1, "I"),
            TextSegment(None, 1, 2, "A later sentence remains aligned."),
        ]
        tokens = [
            ASRToken(None, 1, 0, "III", 100, 300, 0.9),
            ASRToken(None, 1, 1, "A", 600, 650, 0.9),
            ASRToken(None, 1, 2, "later", 660, 800, 0.9),
            ASRToken(None, 1, 3, "sentence", 810, 1050, 0.9),
            ASRToken(None, 1, 4, "remains", 1060, 1230, 0.9),
            ASRToken(None, 1, 5, "aligned", 1240, 1450, 0.9),
        ]

        result = align_segments_to_tokens(segments, tokens, language="en")

        self.assertEqual(SegmentStatus.UNMATCHED, result[0].status)
        self.assertEqual(SegmentStatus.UNMATCHED, result[1].status)
        self.assertEqual((600, 1450), (result[2].start_ms, result[2].end_ms))

    def test_qwen_punctuation_fused_tokens_keep_sentence_starts(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "Mr."),
            TextSegment(None, 1, 1, "Owen—unfortunately delayed—unable to get here."),
            TextSegment(None, 1, 2, "Instructions—everything they wanted—if they asked."),
        ]
        tokens = [
            ASRToken(None, 1, 0, "Mr", 100, 300, 0.9),
            ASRToken(None, 1, 1, "Owenunfortunately", 300, 1_200, 0.9),
            ASRToken(None, 1, 2, "delayedunable", 1_500, 2_100, 0.9),
            ASRToken(None, 1, 3, "to", 2_100, 2_200, 0.9),
            ASRToken(None, 1, 4, "get", 2_200, 2_350, 0.9),
            ASRToken(None, 1, 5, "here", 2_350, 2_600, 0.9),
            ASRToken(None, 1, 6, "Instructionseverything", 3_000, 4_000, 0.9),
            ASRToken(None, 1, 7, "they", 4_000, 4_200, 0.9),
            ASRToken(None, 1, 8, "wantedif", 4_200, 4_800, 0.9),
            ASRToken(None, 1, 9, "they", 4_800, 5_000, 0.9),
            ASRToken(None, 1, 10, "asked", 5_000, 5_300, 0.9),
        ]

        result = align_segments_to_tokens(segments, tokens, language="en")

        self.assertEqual((300, 2_600), (result[1].start_ms, result[1].end_ms))
        self.assertEqual((3_000, 5_300), (result[2].start_ms, result[2].end_ms))

    def test_anchor_mapping_skips_extra_roman_heading_and_resumes_after_it(self) -> None:
        segments = [
            TextSegment(10, 1, 0, "III"),
            TextSegment(11, 1, 1, "The later sentence is still aligned."),
        ]
        tokens = [
            ASRToken(None, 1, 0, "The", 500, 600, 0.9),
            ASRToken(None, 1, 1, "later", 610, 750, 0.9),
            ASRToken(None, 1, 2, "sentence", 760, 950, 0.9),
            ASRToken(None, 1, 3, "is", 960, 1020, 0.9),
            ASRToken(None, 1, 4, "still", 1030, 1150, 0.9),
            ASRToken(None, 1, 5, "aligned", 1160, 1340, 0.9),
        ]

        anchors = anchors_from_segments_tokens(segments, tokens, language="en")

        self.assertTrue(anchors)
        self.assertTrue(all(anchor.segment_id == 11 for anchor in anchors))
        self.assertEqual(3, min(anchor.source_start_char for anchor in anchors))

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

    def test_snap_uses_opposite_edges_of_the_silence_interval(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "before", 100, 900, status=SegmentStatus.AUTO),
            TextSegment(None, 1, 1, "after", 1_100, 2_000, status=SegmentStatus.AUTO),
        ]
        silence = BoundaryCandidate(
            1_000, 0.95, start_ms=800, end_ms=1_200,
        )

        result = snap_boundaries(
            segments, [silence], window_ms=500, padding_ms=80,
        )

        # The preceding cue ends at the leading edge of the pause; the next
        # cue begins at its trailing edge. Padding stays inside the silence.
        self.assertEqual(880, result[0].end_ms)
        self.assertEqual(1_120, result[1].start_ms)
        self.assertNotEqual(silence.end_ms, result[0].end_ms)
        self.assertNotEqual(silence.start_ms, result[1].start_ms)

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

    def test_silence_only_alignment_keeps_cue_edges_on_their_vad_sides(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "first sentence"),
            TextSegment(None, 1, 1, "second sentence"),
            TextSegment(None, 1, 2, "third sentence"),
        ]
        silence = BoundaryCandidate(
            1_500, 0.95, start_ms=1_200, end_ms=1_800,
        )

        result = align_segments_from_silence(
            segments, [silence],
            SilenceAlignmentOptions(0, 0, 4_500, padding_ms=100),
        )

        self.assertEqual(1_300, result.segments[0].end_ms)
        self.assertEqual(1_700, result.segments[1].start_ms)

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
        self.assertEqual((13_500, True), windows[1])
        self.assertTrue(all(end <= 90_000 for end, _weak in windows))

    def test_forced_block_window_uses_uncapped_sentence_estimate_sum(self) -> None:
        candidates = [
            BoundaryCandidate(62_000, 0.9, start_ms=61_500, end_ms=62_400),
            BoundaryCandidate(78_000, 0.9, start_ms=77_500, end_ms=78_400),
        ]

        windows = forced_alignment_window_ends(
            0, 120_000, "many sentences whose combined estimate exceeds one minute",
            candidates, language="en", expected_duration_ms=78_000,
        )

        self.assertEqual((78_400, False), windows[0])

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

    def test_forced_score_never_stabilizes_a_clip_edge_timestamp(self) -> None:
        segments = [TextSegment(None, 1, 0, "one two three four", 200, 3_900)]
        tokens = [
            ASRToken(None, 1, 0, "one", 200, 800, 0.9),
            ASRToken(None, 1, 1, "four", 3_400, 3_900, 0.9),
        ]

        score, reasons, _measured = score_forced_alignment_hypothesis(
            segments, tokens, audio_start_ms=0, audio_end_ms=4_000,
            language="en", observed_ms_per_unit=900,
        )

        self.assertLess(score, ForcedAlignmentPlannerOptions().stable_score)
        self.assertIn("块末尾可能被截断", reasons)

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

    def test_forced_coverage_accepts_local_hole_but_rejects_truncated_suffix(self) -> None:
        local_hole = [
            TextSegment(None, 1, 0, "VII", 100, 200),
            TextSegment(None, 1, 1, "Mr."),
            TextSegment(None, 1, 2, "Justice Wargrave spoke.", 300, 900),
            TextSegment(None, 1, 3, "He sat down.", 1_000, 1_500),
        ]
        truncated = [
            *[TextSegment(None, 1, index, f"timed {index}", index * 100, index * 100 + 80)
              for index in range(5)],
            *[TextSegment(None, 1, index, f"missing {index}") for index in range(5, 20)],
        ]

        self.assertTrue(forced_alignment_has_usable_coverage(local_hole))
        self.assertFalse(forced_alignment_has_usable_coverage(truncated))

    def test_complete_stable_hypothesis_outranks_partial_higher_score(self) -> None:
        complete = ForcedAlignmentHypothesis(
            0, 2, 0, 2_000,
            [
                TextSegment(None, 1, 0, "Two", 100, 300),
                TextSegment(None, 1, 1, "I", 600, 800),
            ], [], 0.76,
        )
        partial = ForcedAlignmentHypothesis(
            0, 2, 0, 2_000,
            [
                TextSegment(None, 1, 0, "Two"),
                TextSegment(None, 1, 1, "I", 100, 300),
            ], [], 0.90,
        )

        self.assertGreater(
            forced_alignment_hypothesis_priority(complete),
            forced_alignment_hypothesis_priority(partial),
        )
        self.assertTrue(
            forced_alignment_candidate_can_end_search(complete.segments, complete.score)
        )
        self.assertFalse(
            forced_alignment_candidate_can_end_search(partial.segments, partial.score)
        )

    def test_forced_overlap_ignores_shared_unmatched_heading(self) -> None:
        previous = ForcedAlignmentHypothesis(
            0, 3, 0, 3_000,
            [
                TextSegment(None, 1, 0, "before", 100, 800),
                TextSegment(None, 1, 1, "VII"),
                TextSegment(None, 1, 2, "after", 1_000, 1_800),
            ], [], 0.8,
        )
        current = ForcedAlignmentHypothesis(
            1, 4, 0, 3_000,
            [
                TextSegment(None, 1, 1, "VII"),
                TextSegment(None, 1, 2, "after", 1_080, 1_880),
                TextSegment(None, 1, 3, "later", 2_000, 2_700),
            ], [], 0.8,
        )

        self.assertGreater(forced_alignment_overlap_score(previous, current), 0)

    def test_contiguous_media_boundary_selects_following_part(self) -> None:
        first = ("first", None, "a", 0, 10_000)
        second = ("second", None, "b", 10_000, 20_000)
        part, cursor = locate_contiguous_audio_part([first, second], 10_000)
        self.assertIs(second, part)
        self.assertEqual(10_000, cursor)

    def test_regressive_forced_block_is_rejected_without_poisoning_later_recovery(self) -> None:
        rows = [
            (10, TextSegment(None, 1, 10, "stable", 700_000, 710_000)),
            (11, TextSegment(None, 1, 11, "bad reset", 77_000, 80_000)),
            (12, TextSegment(None, 1, 12, "still bad", 81_000, 85_000)),
            (13, TextSegment(None, 1, 13, "recovered", 711_000, 716_000)),
        ]

        accepted, rejected = partition_monotonic_alignment_rows(rows)

        self.assertEqual({10, 13}, accepted)
        self.assertEqual({11, 12}, rejected)


if __name__ == "__main__":
    unittest.main()

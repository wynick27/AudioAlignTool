from __future__ import annotations

import unittest

from audioalign.core.alignment import align_segments_from_silence, anchors_from_segments_tokens, align_segments_to_tokens, segments_from_asr_tokens, snap_boundaries
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


if __name__ == "__main__":
    unittest.main()

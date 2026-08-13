from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication, QMessageBox
    from audioalign.core.models import ASRToken, AlignmentMode, AudioAsset, AudioVisualizationMode, Chapter, ChapterAudioLink, InferenceDeviceInfo, PlaybackFollowState, SegmentOverlapPolicy, SegmentStatus, TaskLane, TextAudioAnchor, TextSegment
    from audioalign.core.paths import ApplicationPaths
    from audioalign.core.spectrogram import build_spectrogram_cache_from_slices
    from audioalign.core.storage import ProjectSession
    from audioalign.gui.asr_comparison import ASRComparisonView
    from audioalign.gui.main_window import (
        MainWindow,
        TaskManager,
        WORKFLOW_FASTER_WHISPER,
        WORKFLOW_QWEN_ASR,
        WORKFLOW_QWEN_FORCED,
    )
    from audioalign.gui.mapping_dialog import ChapterAudioMappingDialog
    from audioalign.gui.segment_model import SegmentTableModel
    from audioalign.gui.spectrogram_editor import (
        AudioVisualizerEditor,
        AudioVisualizerOverview,
        StableTimeAxis,
    )
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class GuiSmokeTests(unittest.TestCase):
    def test_segment_loop_uses_live_timing_but_manual_selection_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("live-loop", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(
                chapter_id,
                [TextSegment(None, chapter_id, 0, "sentence", 1_000, 4_000)],
            )
            window = MainWindow()
            window._set_session(session)
            window._set_segment_play_range(0)
            self.assertEqual((1_000, 4_000), window._active_play_bounds())

            window.segment_model.segments[0].end_ms = 2_500
            window._refresh_segment_play_range()
            self.assertEqual((1_000, 2_500), window._active_play_bounds())
            self.assertEqual(
                (1_000, 2_500),
                (window.spectrogram.selection.start_ms, window.spectrogram.selection.end_ms),
            )

            # A selection/play range originating from this cue follows a live
            # boundary edit, so replay never uses the stale old endpoint.
            window._clear_play_range()
            window.spectrogram.set_segment_selection(1_000, 2_500)
            window._play_range_start = 1_000
            window._play_range_end = 2_500
            window._boundary_moved(0, "end", 2_200, False)
            self.assertEqual(
                (1_000, 2_200),
                (window.spectrogram.selection.start_ms, window.spectrogram.selection.end_ms),
            )
            self.assertEqual((1_000, 2_200), window._active_play_bounds())
            self.assertTrue(window.spectrogram.selection_tracks_segment)
            self.assertTrue(all(not pane.selection.isVisible() for pane in window.spectrogram._panes))

            # An independently drawn range is deliberately not attached to a
            # cue and therefore remains fixed across later cue edits.
            window._clear_play_range()
            window.spectrogram.set_selection(5_000, 7_000)
            window._play_range_start = 5_000
            window._play_range_end = 7_000
            window.segment_model.segments[0].end_ms = 2_000
            self.assertEqual((5_000, 7_000), window._active_play_bounds())
            self.assertFalse(window.spectrogram.selection_tracks_segment)
            self.assertTrue(any(pane.selection.isVisible() for pane in window.spectrogram._panes))
            window.close()

    def test_unmatched_zero_time_row_does_not_mark_timed_neighbours_as_conflicts(self) -> None:
        model = SegmentTableModel()
        model.set_segments([
            TextSegment(None, 1, 0, "前句", 1_000, 2_000, status=SegmentStatus.MANUAL),
            TextSegment(None, 1, 1, "已删除时间的原文", 0, 0, status=SegmentStatus.UNMATCHED),
            TextSegment(None, 1, 2, "后句", 2_000, 3_000, status=SegmentStatus.MANUAL),
            TextSegment(None, 1, 3, "再后一句", 3_000, 4_000, status=SegmentStatus.MANUAL),
        ])

        conflict = QColor(230, 55, 70, 65)
        unmatched = QColor(220, 60, 90, 40)
        self.assertNotEqual(conflict, model.data(model.index(0, 0), Qt.ItemDataRole.BackgroundRole))
        self.assertEqual(unmatched, model.data(model.index(1, 0), Qt.ItemDataRole.BackgroundRole))
        self.assertNotEqual(conflict, model.data(model.index(2, 0), Qt.ItemDataRole.BackgroundRole))
        self.assertNotEqual(conflict, model.data(model.index(3, 0), Qt.ItemDataRole.BackgroundRole))

        editor = AudioVisualizerEditor()
        editor.set_segments(model.segments)
        self.assertNotEqual((230, 55, 70, 105), editor._cue_colour(0))
        self.assertEqual((220, 65, 90, 82), editor._cue_colour(1))
        self.assertNotEqual((230, 55, 70, 105), editor._cue_colour(2))

    def test_selecting_a_segment_does_not_recenter_audio_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("selection-no-jump", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(
                chapter_id,
                [
                    TextSegment(None, chapter_id, 0, "first", 1_000, 2_000),
                    TextSegment(None, chapter_id, 1, "second", 20_000, 21_000),
                ],
            )
            window = MainWindow()
            window._set_session(session)
            original_range = (window.spectrogram.view_start, window.spectrogram.view_end)
            with patch.object(window.spectrogram, "focus_time") as focus:
                window._select_segment_row(1)
                self.app.processEvents()
                focus.assert_not_called()
            self.assertEqual(
                original_range,
                (window.spectrogram.view_start, window.spectrogram.view_end),
            )
            window.close()

    def test_sentence_double_click_locates_only_when_start_is_offscreen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("double-click-locate", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(
                chapter_id,
                [
                    TextSegment(None, chapter_id, 0, "visible", 2_000, 3_000),
                    TextSegment(None, chapter_id, 1, "offscreen", 20_000, 21_000),
                ],
            )
            window = MainWindow()
            window._set_session(session)
            window.spectrogram.set_cache(None, 30_000)
            window.spectrogram.set_time_range(0, 10_000)

            with patch.object(window.spectrogram, "focus_time") as focus:
                window._segment_double_activated(0)
                focus.assert_not_called()
                window._segment_double_activated(1)
                focus.assert_called_once_with(20_000, 10_000)

            # A single click in the lower audio editor seeks/selects but must
            # never invoke sentence-style centring.
            with patch.object(window.spectrogram, "focus_time") as focus:
                window._audio_time_activated(20_500, 1)
                focus.assert_not_called()
            window.close()

    def test_audio_double_click_preserves_pre_seek_playing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("double-click", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(
                chapter_id,
                [TextSegment(None, chapter_id, 0, "sentence", 1_000, 4_000)],
            )
            window = MainWindow()
            window._set_session(session)
            # Simulate the backend reporting Paused after the first click seek,
            # while the click latch records that playback had been active.
            window._recent_audio_click_at = time.monotonic()
            window._recent_audio_click_row = 0
            window._recent_audio_click_was_playing = True
            window._segment_double_activated(0, 2_000)
            self.assertEqual((1_000, 4_000), window._active_play_bounds())
            window.close()

    def test_fixed_text_editor_splits_at_cursor_as_one_undo_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("cursor-split", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            original = "First clause, second clause"
            session.repository.replace_segments(
                chapter_id,
                [TextSegment(None, chapter_id, 0, original, 100, 2100)],
            )
            window = MainWindow()
            window._set_session(session)
            window._select_segment_row(0)
            cursor = window.editor_text.textCursor()
            cursor.setPosition(len("First clause"))
            window.editor_text.setTextCursor(cursor)
            window.split_segment_at_text_cursor()

            self.assertEqual(2, window.segment_model.rowCount())
            self.assertEqual("First clause,", window.segment_model.segments[0].text)
            self.assertEqual("second clause", window.segment_model.segments[1].text)
            self.assertEqual(
                window.segment_model.segments[0].end_ms,
                window.segment_model.segments[1].start_ms,
            )
            window.undo()
            self.assertEqual(1, window.segment_model.rowCount())
            self.assertEqual(original, window.segment_model.segments[0].text)
            window.close()

    def test_asr_comparison_separates_blocks_and_restores_word_spaces(self) -> None:
        view = ASRComparisonView()
        view.set_content(
            [
                TextSegment(None, 1, 0, "Hello, world."),
                TextSegment(None, 1, 1, "日本語です。"),
            ],
            [
                ASRToken(None, 1, 0, "Hello", 0, 300, 0.9),
                ASRToken(None, 1, 1, ",", 300, 320, 0.9),
                ASRToken(None, 1, 2, "world", 320, 600, 0.9),
                ASRToken(None, 1, 3, ".", 600, 650, 0.9),
                ASRToken(None, 1, 4, "日本", 1800, 2100, 0.9),
                ASRToken(None, 1, 5, "語です。", 2100, 2500, 0.9),
            ],
        )
        source_text = view.source.toPlainText()
        transcript_text = view.transcript.toPlainText()
        self.assertIn("Hello, world.", source_text)
        self.assertIn("日本語です。", source_text)
        self.assertIn("Hello, world.", transcript_text)
        self.assertIn("日本語です。", transcript_text)
        self.assertNotIn("日本 語", transcript_text)
        self.assertGreaterEqual(transcript_text.count("\n"), 1)

    def test_asr_word_click_seeks_and_centres_audio_view(self) -> None:
        window = MainWindow()
        window.spectrogram.view_start = 10_000
        window.spectrogram.view_end = 30_000
        with patch.object(window, "_seek_local") as seek, patch.object(
            window.spectrogram, "focus_time"
        ) as focus:
            window._asr_word_seek(42_500)
            seek.assert_called_once_with(42_500)
            focus.assert_called_once_with(42_500, 20_000)
        window.close()

    def test_punctuation_split_uses_asr_character_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("anchor-split", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            text = "First sentence. Second sentence."
            session.repository.replace_segments(chapter_id, [
                TextSegment(
                    None, chapter_id, 0, text, 0, 2_000,
                    source_start_char=0, source_end_char=len(text),
                )
            ])
            segment = session.repository.segments(chapter_id)[0]
            session.repository.replace_anchors(chapter_id, [
                TextAudioAnchor(None, chapter_id, segment.id, 0, 14, 100, 500, 1.0, "asr-word"),
                TextAudioAnchor(None, chapter_id, segment.id, 16, 31, 900, 1_500, 1.0, "asr-word"),
            ])
            window = MainWindow()
            window._set_session(session)
            window._select_segment_row(0)
            window.split_segment_by_punctuation()
            self.assertEqual(2, window.segment_model.rowCount())
            self.assertEqual(900, window.segment_model.segments[0].end_ms)
            self.assertEqual(900, window.segment_model.segments[1].start_ms)
            window.close()

    def test_binding_updates_existing_row_without_rebuilding_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("fast-bind", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(
                chapter_id, [TextSegment(None, chapter_id, 0, "sentence", 100, 500)],
            )
            window = MainWindow()
            window._set_session(session)
            window._select_segment_row(0)
            window.spectrogram.set_selection(700, 1_100)
            with patch.object(
                session.repository, "replace_chapter_edit_state",
                wraps=session.repository.replace_chapter_edit_state,
            ) as replace, patch.object(
                session.repository, "update_segments",
                wraps=session.repository.update_segments,
            ) as update:
                window.bind_selection()
                replace.assert_not_called()
                update.assert_called_once()
            self.assertEqual((700, 1_100), (
                session.repository.segments(chapter_id)[0].start_ms,
                session.repository.segments(chapter_id)[0].end_ms,
            ))
            window.close()

    def test_inference_and_media_lanes_run_concurrently(self) -> None:
        manager = TaskManager()
        barrier = threading.Barrier(2)
        completed: list[str] = []

        def make_job(label: str):
            def job(progress):
                progress(0.1, label)
                barrier.wait(timeout=3)
                progress(1.0, label)
                return label
            return job

        manager.submit("inference", make_job("inference"), completed.append)
        manager.submit("media", make_job("media"), completed.append, lane=TaskLane.MEDIA)
        deadline = time.monotonic() + 5
        while len(completed) < 2 and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertCountEqual(["inference", "media"], completed)

    def test_three_overlap_policies(self) -> None:
        window = MainWindow()
        try:
            def reset():
                window.segment_model.set_segments([
                    TextSegment(None, 1, 0, "a", 0, 1000),
                    TextSegment(None, 1, 1, "b", 1000, 2000),
                ])
                window.spectrogram.set_segments(window.segment_model.segments)

            reset()
            window.overlap_policy_combo.setCurrentIndex(
                window.overlap_policy_combo.findData(SegmentOverlapPolicy.CLAMP_CURRENT)
            )
            window._boundary_moved(0, "end", 1500, False)
            self.assertEqual((1000, 1000), (
                window.segment_model.segments[0].end_ms,
                window.segment_model.segments[1].start_ms,
            ))

            reset()
            window.overlap_policy_combo.setCurrentIndex(
                window.overlap_policy_combo.findData(SegmentOverlapPolicy.TRIM_NEIGHBORS)
            )
            window._boundary_moved(0, "end", 1500, False)
            self.assertEqual((1500, 1500), (
                window.segment_model.segments[0].end_ms,
                window.segment_model.segments[1].start_ms,
            ))

            reset()
            window.overlap_policy_combo.setCurrentIndex(
                window.overlap_policy_combo.findData(SegmentOverlapPolicy.ALLOW_OVERLAP)
            )
            window._boundary_moved(0, "end", 1500, False)
            self.assertEqual((1500, 1000), (
                window.segment_model.segments[0].end_ms,
                window.segment_model.segments[1].start_ms,
            ))
        finally:
            window.close()

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._settings_path = Path(__file__).resolve().parents[1] / "settings.json"
        self._settings_before = self._settings_path.read_bytes() if self._settings_path.exists() else None

    def tearDown(self) -> None:
        if self._settings_before is None:
            self._settings_path.unlink(missing_ok=True)
        else:
            self._settings_path.write_bytes(self._settings_before)

    def test_time_axis_spacing_and_follow_cursor_are_stable(self) -> None:
        axis = StableTimeAxis()
        axis.set_step(5_000)
        ticks = axis.tickValues(9.2, 15.8, 800)[0][1]
        self.assertEqual([10.0, 15.0], ticks)

        editor = AudioVisualizerEditor()
        editor.resize(1000, 300)
        editor.set_cache(None, 60_000)
        editor.set_time_range(5_000, 35_000)
        editor.follow_playhead(20_000, force=True)
        centre = (editor.view_start + editor.view_end) / 2
        self.assertEqual(20_000, centre)
        self.assertAlmostEqual(20.0, editor.wave_pane.play_line.value())

        # A sub-pixel update is deliberately coalesced.  Both the view and
        # visible line stay at the old centre instead of producing a sawtooth.
        editor.follow_playhead(20_001)
        centre = (editor.view_start + editor.view_end) / 2
        self.assertAlmostEqual(centre / 1000, editor.wave_pane.play_line.value())

    def test_shared_boundary_uses_pointer_side_and_press_ignores_stale_hover(self) -> None:
        editor = AudioVisualizerEditor()
        editor.resize(1000, 300)
        editor.set_cache(None, 10_000)
        editor.set_time_range(0, 10_000)
        editor.set_segments([
            TextSegment(None, 1, 0, "前一句", 0, 5_000),
            TextSegment(None, 1, 1, "后一句", 5_000, 10_000),
        ])

        left = editor._nearest_target(4_990)
        self.assertEqual(("end", 0), (left.kind, left.segment_index))
        right = editor._nearest_target(5_010)
        self.assertEqual(("start", 1), (right.kind, right.segment_index))
        exact = editor._nearest_target(5_000)
        self.assertEqual(("start", 1), (exact.kind, exact.segment_index))

        # A delayed hover event from the left side must not override the real
        # press position after the pointer has crossed to the following cue.
        editor._hover_target = left
        editor._hover_position = 4_990
        editor._drag_start(5.010, 0)
        self.assertEqual(("start", 1), (editor._drag.kind, editor._drag.segment_index))

    def test_empty_audio_timeline_keeps_one_fixed_scale(self) -> None:
        editor = AudioVisualizerEditor()
        editor.resize(1000, 300)
        self.assertFalse(editor.has_audio)
        self.assertEqual((0, 30_000), (editor.view_start, editor.view_end))
        initial_ticks = editor.wave_pane.time_axis.tickValues(0, 30, 900)
        editor.follow_playhead(12_345, force=True)
        editor.focus_time(999_999, 500)
        editor._range_changed("waveform", [[0.123, 0.124], [-1, 1]])
        editor.set_mode(AudioVisualizationMode.WAVEFORM)
        editor.set_mode(AudioVisualizationMode.COMBINED)
        self.assertEqual((0, 30_000), (editor.view_start, editor.view_end))
        self.assertEqual(0, editor.playhead)
        self.assertEqual(initial_ticks, editor.wave_pane.time_axis.tickValues(0, 30, 900))

        overview = AudioVisualizerOverview()
        self.assertFalse(overview.has_audio)
        self.assertEqual(30_000, overview.duration_ms)

        # A known audio duration is usable before its FFT cache has finished.
        editor.set_cache(None, 60_000)
        self.assertTrue(editor.has_audio)
        self.assertEqual(60_000, editor.duration_ms)
        overview.set_cache(None, 60_000)
        overview.set_playhead(12_345)
        self.assertAlmostEqual(12.345, overview.play_line.value())
        self.assertIn("黄色=低置信度句段", overview.toolTip())
        self.assertIn("蓝色边框及淡蓝填充=主视图窗口", overview.toolTip())
        self.assertTrue(all(line.pen.color().name() == "#4da3ff" for line in overview.window.lines))

    def test_window_loads_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"LOCALAPPDATA": temporary}):
            session = ProjectSession.create("GUI 测试", Path(temporary) / "project", internal=False)
            chapter_id = session.repository.add_chapter(Chapter(None, "纯音频章节", 0))
            session.repository.replace_segments(
                chapter_id,
                [TextSegment(None, chapter_id, 0, "自动生成的文字", 0, 1000)],
            )
            window = MainWindow()
            window._set_session(session)
            self.assertEqual(1, window.chapter_list.count())
            self.assertEqual(1, window.segment_model.rowCount())
            self.assertEqual("纯音频章节", window.chapter_list.item(0).text()[2:])
            window.close()

    def test_spectrogram_views_load_real_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            rate = 16000
            values = (np.sin(np.linspace(0, 300, rate * 2)) * 18000).astype(np.int16)
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(values.tobytes())
            session = ProjectSession.create("频谱 GUI 测试", root / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "第一章", 0))
            session.repository.replace_segments(
                chapter_id,
                [
                    TextSegment(None, chapter_id, 0, "第一句", 0, 900),
                    TextSegment(None, chapter_id, 1, "第二句", 900, 1800),
                ],
            )
            audio_id = session.repository.add_audio(AudioAsset(None, str(audio), duration_ms=2000, sample_rate=rate, channels=1, format="wav"))
            session.repository.set_chapter_links(chapter_id, [ChapterAudioLink(None, chapter_id, audio_id, 0, 0, 2000, 1.0)])
            import hashlib
            token = hashlib.sha1(f"{audio_id}:0:2000".encode("ascii")).hexdigest()[:16]
            cache_dir = session.root / "cache" / f"spectrogram-chapter-{chapter_id}-{token}"
            build_spectrogram_cache_from_slices([(audio, 0, 2000)], cache_dir)
            window = MainWindow()
            window._set_session(session)
            self.app.processEvents()
            self.assertIsNotNone(window.current_cache)
            self.assertGreater(window.article_view.line_model.rowCount(), 0)
            window.spectrogram.set_selection(250, 750)
            self.assertEqual((250, 750), (window.spectrogram.selection.start_ms, window.spectrogram.selection.end_ms))
            window.close()

    def test_mapping_apply_button_commits_and_cancel_is_chinese(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("配对测试", root / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "CAPÍTULO 1", 0))
            session.repository.replace_segments(chapter_id, [TextSegment(None, chapter_id, 0, "Texto " * 200, 0, 0)])
            audio_id = session.repository.add_audio(
                AudioAsset(None, str(root / "01.opus"), duration_ms=60_000, format="opus", title="CAPÍTULO 1")
            )
            dialog = ChapterAudioMappingDialog(session)
            dialog.auto_match()
            self.assertEqual("取消", dialog.cancel_button.text())
            dialog.apply_button.click()
            links = session.repository.chapter_links(chapter_id)
            self.assertEqual(1, len(links))
            self.assertEqual(audio_id, links[0].audio_id)
            session.close()

    def test_modes_follow_speed_and_current_segment_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("编辑器测试", root / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "章节", 0))
            session.repository.replace_segments(
                chapter_id, [TextSegment(None, chapter_id, 0, "原文", 100, 900)]
            )
            window = MainWindow()
            window._set_session(session)
            window.set_visualization_mode(AudioVisualizationMode.WAVEFORM)
            self.assertEqual(AudioVisualizationMode.WAVEFORM, window.spectrogram.mode)
            self.assertTrue(window.waveform_button.isChecked())
            self.assertFalse(window.spectrum_button.isChecked())
            self.assertFalse(window.combined_button.isChecked())
            window.waveform_button.click()
            self.assertTrue(window.waveform_button.isChecked())
            waveform_glyph = window.waveform_button.text()
            window.combined_button.click()
            self.assertFalse(window.waveform_button.isChecked())
            self.assertTrue(window.combined_button.isChecked())
            self.assertEqual(waveform_glyph, window.waveform_button.text())
            self.assertNotEqual("...", window.waveform_button.text())
            window.resize(1200, 800)
            window.show()
            self.app.processEvents()
            self.assertLessEqual(
                abs(window.spectrogram.wave_pane.plot.height() - window.spectrogram.spectrum_pane.plot.height()), 2
            )
            self.assertEqual(
                window.spectrogram.wave_pane.plot.getViewBox().width(),
                window.spectrogram.spectrum_pane.plot.getViewBox().width(),
            )
            window.set_playback_rate(1.75)
            self.assertAlmostEqual(1.75, window.player.playbackRate())
            qwen_mode = window.mode_combo.findData(WORKFLOW_QWEN_FORCED)
            window.mode_combo.setCurrentIndex(qwen_mode)
            self.assertIn("Qwen", window.recognize_button.text())
            qwen_actions = [action.text() for action in window.qwen_align_menu.actions() if not action.isSeparator()]
            self.assertEqual(
                ["对齐当前句", "对齐所选句子 ↔ 音频选区", "从当前句/时间向后对齐"],
                qwen_actions,
            )
            self.assertTrue(window.refresh_recognition_button.isHidden())
            self.assertFalse(window.model_combo.isEnabled())
            window.current_parts = [object()]
            with patch.object(window, "qwen_align_chapter") as align_chapter:
                window.recognize_current()
                align_chapter.assert_called_once()
            window.mode_combo.setCurrentIndex(window.mode_combo.findData(WORKFLOW_FASTER_WHISPER))
            self.assertTrue(window.model_combo.isEnabled())
            window.mode_combo.setCurrentIndex(window.mode_combo.findData(WORKFLOW_QWEN_ASR))
            self.assertEqual(31, window.language_combo.count())
            self.assertGreaterEqual(window.language_combo.findData("mk"), 0)
            window.mode_combo.setCurrentIndex(window.mode_combo.findData(WORKFLOW_QWEN_FORCED))
            self.assertEqual(11, window.language_combo.count())
            self.assertEqual(-1, window.language_combo.findData("auto"))
            window.spectrogram.set_follow_enabled(True)
            window.spectrogram.follow_playhead(500, force=True)
            self.assertEqual(PlaybackFollowState.FOLLOWING, window.spectrogram.follow_state)
            window.spectrogram.suspend_follow("test")
            self.assertEqual(PlaybackFollowState.SUSPENDED, window.spectrogram.follow_state)
            window.follow_button.click()
            self.assertEqual(PlaybackFollowState.FOLLOWING, window.spectrogram.follow_state)
            window.follow_button.click()
            self.assertEqual(PlaybackFollowState.DISABLED, window.spectrogram.follow_state)
            window.follow_button.click()
            self.assertEqual(PlaybackFollowState.FOLLOWING, window.spectrogram.follow_state)
            window.spectrogram.set_cache(None, 10_000)
            window.spectrogram.set_time_range(0, 10_000)
            self.assertGreater(window.spectrogram.spectrum_pane.play_line.zValue(), window.spectrogram.image.zValue())
            self.assertTrue(any(line.isVisible() for line in window.spectrogram.spectrum_pane.grid_lines))
            window.spectrogram._wheel(
                "waveform", 5.0, 120,
                int(Qt.KeyboardModifier.ControlModifier.value),
            )
            self.assertLess(window.spectrogram.view_end - window.spectrogram.view_start, 10_000)
            self.assertEqual(PlaybackFollowState.SUSPENDED, window.spectrogram.follow_state)
            window._select_segment_row(0)
            window.editor_text.setPlainText("修改后\n第二行")
            window._commit_segment_editor()
            self.assertEqual("修改后\n第二行", session.repository.segments(chapter_id)[0].text)
            self.assertEqual("▶", window.play_button.text())
            window.close()

    def test_new_project_replaces_current_session_and_creates_library_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = ApplicationPaths(root)
            paths.ensure()
            old = ProjectSession.create("old-project", root / "old-project")
            window = MainWindow()
            window.paths = paths
            window._set_session(old)
            with patch("audioalign.core.storage.ApplicationPaths.current", return_value=paths):
                self.assertTrue(window._ensure_project("new-project", force_new=True))
            project = paths.projects / "new-project"
            self.assertEqual(project, window.session.root)
            self.assertTrue((project / "manifest.json").is_file())
            self.assertTrue((project / "project.sqlite3").is_file())
            for name in ("source", "media", "cache"):
                self.assertTrue((project / name).is_dir())
            window.close()

    def test_book_batch_recognition_processes_paired_chapters_with_one_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "book.wav"
            rate = 16000
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(np.zeros(rate * 2, dtype=np.int16).tobytes())
            session = ProjectSession.create("batch", root / "project")
            audio_id = session.repository.add_audio(
                AudioAsset(None, str(audio), duration_ms=1000, sample_rate=rate, channels=1, format="wav")
            )
            chapter_ids = []
            for position in range(2):
                chapter_id = session.repository.add_chapter(Chapter(None, f"chapter {position + 1}", position))
                chapter_ids.append(chapter_id)
                session.repository.replace_segments(
                    chapter_id, [TextSegment(None, chapter_id, 0, "hello")]
                )
                session.repository.set_chapter_links(
                    chapter_id,
                    [ChapterAudioLink(None, chapter_id, audio_id, 0, position * 1000, (position + 1) * 1000, 1.0)],
                )

            class FakeTranscriber:
                def __init__(self):
                    self.calls = 0
                    self.last_device_info = InferenceDeviceInfo(
                        "fake", "small", actual_device="cpu", compute_type="int8"
                    )

                def transcribe(self, _path, chapter_id, _options, progress):
                    self.calls += 1
                    progress(1.0, "fake recognition")
                    return [ASRToken(None, chapter_id, 0, "hello", 100, 900, 0.95)]

            fake = FakeTranscriber()
            ready = SimpleNamespace(
                runtime_available=True,
                model_available=True,
                whisperx_available=True,
                cuda_available=True,
                message="ready",
            )
            window = MainWindow()
            window.session = session
            window.mode_combo.setCurrentIndex(window.mode_combo.findData(WORKFLOW_FASTER_WHISPER))
            with (
                patch("audioalign.gui.main_window.runtime_status", return_value=ready),
                patch("audioalign.gui.main_window.transcriber_for_options", return_value=fake),
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes),
            ):
                window.recognize_book()
                deadline = time.monotonic() + 10
                while (window.tasks.current or window.tasks.queue) and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()

                # A second book run must skip both inference and text matching;
                # the completed alignment is a separate cache layer.
                window.recognize_book()
                deadline = time.monotonic() + 10
                while (window.tasks.current or window.tasks.queue) and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.app.processEvents()

            self.assertFalse(window.tasks.current)
            self.assertEqual(2, fake.calls)
            for chapter_id in chapter_ids:
                segment = session.repository.segments(chapter_id)[0]
                self.assertEqual((100, 900), (segment.start_ms, segment.end_ms))
                self.assertEqual("hello", session.repository.asr_tokens(chapter_id)[0].text)
            window.close()

    def test_inline_audio_visibility_and_article_text_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("article-select", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(
                chapter_id, [TextSegment(None, chapter_id, 0, "Selectable article text", 0, 1000)],
            )
            window = MainWindow()
            window._set_session(session)
            window._set_list_visualization_mode(AudioVisualizationMode.NONE)
            self.assertTrue(window.segment_table.isColumnHidden(2))
            self.assertTrue(window.list_none_action.isChecked())
            window._set_list_visualization_mode(AudioVisualizationMode.WAVEFORM)
            self.assertFalse(window.segment_table.isColumnHidden(2))
            self.assertEqual(AudioVisualizationMode.WAVEFORM, window.mini_delegate.mode)
            self.assertTrue(window.list_waveform_action.isChecked())
            window._set_list_visualization_mode(AudioVisualizationMode.SPECTROGRAM)
            self.assertEqual(AudioVisualizationMode.SPECTROGRAM, window.mini_delegate.mode)

            window._set_article_visualization_mode(AudioVisualizationMode.NONE)
            self.assertFalse(window.article_view.canvas.audio_visible)
            self.assertTrue(window.article_view.none_button.isChecked())
            window._set_article_visualization_mode(AudioVisualizationMode.WAVEFORM)
            self.assertTrue(window.article_view.canvas.audio_visible)
            self.assertEqual(AudioVisualizationMode.WAVEFORM, window.article_view.canvas.mode)
            window._set_article_visualization_mode(AudioVisualizationMode.SPECTROGRAM)
            self.assertEqual(AudioVisualizationMode.SPECTROGRAM, window.article_view.canvas.mode)
            window.silence_markers_action.trigger()
            self.assertFalse(window.spectrogram.silences_visible)
            self.assertFalse(window.overview.silences_visible)
            canvas = window.article_view.canvas
            canvas._selection_anchor = 0
            canvas._selection_cursor = 10
            self.assertEqual("Selectable", canvas.selected_text())

            # PySide6's QTextLayout.FormatRange is a value type; painting a
            # current/selected span must not use the incompatible PyQt-style
            # three-argument constructor.
            canvas.focus_segment(0, ensure_visible=False)
            image = QImage(500, 60, QImage.Format.Format_ARGB32_Premultiplied)
            painter = QPainter(image)
            try:
                canvas._draw_text(painter, canvas.lines[0], QRect(0, 0, 480, 40))
            finally:
                painter.end()

            # Advancing through matched sentences must only recolour the old
            # and new regions, never rebuild all subtitle graphics.
            with patch.object(window.spectrogram, "_render_cues") as rebuild:
                window.spectrogram.select_segment(-1)
                window.spectrogram.select_segment(0)
                rebuild.assert_not_called()

            # Boundary motion is a lightweight preview.  SQLite and the
            # article/overview refresh happen once, on mouse release.
            with patch.object(window, "_persist_current_segments") as full_persist:
                window._boundary_moved(0, "end", 900, False)
                full_persist.assert_not_called()
            with patch.object(session.repository, "update_segment", wraps=session.repository.update_segment) as update:
                window._commit_dragged_segment(0)
                update.assert_called_once()

            original_text = window.segment_model.segments[0].text
            window.delete_segments()
            self.assertEqual(1, window.segment_model.rowCount())
            self.assertEqual(original_text, window.segment_model.segments[0].text)
            self.assertEqual(SegmentStatus.UNMATCHED, window.segment_model.segments[0].status)
            self.assertEqual((0, 0), (
                window.segment_model.segments[0].start_ms,
                window.segment_model.segments[0].end_ms,
            ))
            stored = session.repository.segments(chapter_id)[0]
            self.assertEqual(original_text, stored.text)
            self.assertEqual(SegmentStatus.UNMATCHED, stored.status)

            source_scroll = window.asr_comparison.source.verticalScrollBar()
            transcript_scroll = window.asr_comparison.transcript.verticalScrollBar()
            source_scroll.setRange(0, 100)
            transcript_scroll.setRange(0, 200)
            source_scroll.setValue(50)
            self.assertEqual(100, transcript_scroll.value())
            window.close()

    def test_undo_never_crosses_project_or_chapter_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = ProjectSession.create("first", root / "first")
            first_chapter = first.repository.add_chapter(Chapter(None, "first chapter", 0))
            first.repository.replace_segments(
                first_chapter, [TextSegment(None, first_chapter, 0, "first original")],
            )
            second = ProjectSession.create("second", root / "second")
            second_chapter = second.repository.add_chapter(Chapter(None, "second chapter", 0))
            second.repository.replace_segments(
                second_chapter, [TextSegment(None, second_chapter, 0, "second original")],
            )
            other_chapter = second.repository.add_chapter(Chapter(None, "other chapter", 1))
            second.repository.replace_segments(
                other_chapter, [TextSegment(None, other_chapter, 0, "other original")],
            )

            window = MainWindow()
            window._set_session(first)
            window._push_history()
            window.segment_model.segments[0].text = "first changed"
            window._persist_current_segments()

            window._set_session(second)
            window.undo()
            self.assertEqual("second original", second.repository.segments(second_chapter)[0].text)

            window._push_history()
            window.segment_model.segments[0].text = "second changed"
            window._persist_current_segments()
            window._load_segment_editor(0)
            window.chapter_list.setCurrentRow(1)
            window.undo()
            self.assertEqual("other original", second.repository.segments(other_chapter)[0].text)
            window.chapter_list.setCurrentRow(0)
            window.undo()
            self.assertEqual("second original", second.repository.segments(second_chapter)[0].text)
            window.close()


if __name__ == "__main__":
    unittest.main()

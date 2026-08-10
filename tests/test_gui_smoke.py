from __future__ import annotations

import os
import tempfile
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
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication, QMessageBox
    from audioalign.core.models import ASRToken, AlignmentMode, AudioAsset, AudioVisualizationMode, Chapter, ChapterAudioLink, InferenceDeviceInfo, PlaybackFollowState, SegmentStatus, TextSegment
    from audioalign.core.paths import ApplicationPaths
    from audioalign.core.spectrogram import build_spectrogram_cache_from_slices
    from audioalign.core.storage import ProjectSession
    from audioalign.gui.main_window import (
        MainWindow,
        WORKFLOW_FASTER_WHISPER,
        WORKFLOW_QWEN_ASR,
        WORKFLOW_QWEN_FORCED,
    )
    from audioalign.gui.mapping_dialog import ChapterAudioMappingDialog
    from audioalign.gui.spectrogram_editor import AudioVisualizerEditor, StableTimeAxis
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class GuiSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

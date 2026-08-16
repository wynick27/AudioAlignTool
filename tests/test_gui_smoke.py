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
    from PySide6.QtCore import QItemSelectionModel, QPoint, QRect, Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication, QMessageBox
    from audioalign.core.models import ASRBackendId, ASRToken, AlignmentMode, AudioAsset, AudioVisualizationMode, BoundaryCandidate, Chapter, ChapterAudioLink, InferenceDeviceInfo, PlaybackFollowState, SegmentOverlapPolicy, SegmentStatus, SilenceDisplayMode, TaskLane, TextAudioAnchor, TextSegment
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
        _asr_uses_silence_snap,
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
    def test_qwen_asr_keeps_model_timestamps_and_manual_snap_is_directional(self) -> None:
        self.assertFalse(_asr_uses_silence_snap(ASRBackendId.QWEN3_ASR))
        self.assertTrue(_asr_uses_silence_snap(ASRBackendId.FASTER_WHISPER))
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("qwen-no-vad-shift", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "first", 300, 900),
                TextSegment(None, chapter_id, 1, "second", 1_300, 1_900),
            ])
            window = MainWindow()
            window._set_session(session)
            window.silence_candidates = [
                BoundaryCandidate(1_100, 0.95, start_ms=950, end_ms=1_250),
            ]
            tokens = [
                ASRToken(None, chapter_id, 0, "first", 300, 900, 0.95),
                ASRToken(None, chapter_id, 1, "second", 1_300, 1_900, 0.95),
            ]

            window._apply_recognition_tokens(
                chapter_id, tokens,
                SimpleNamespace(
                    backend=ASRBackendId.QWEN3_ASR.value,
                    model="Qwen3-ASR-0.6B",
                    actual_device="cpu",
                    compute_type="float32",
                    device_name="",
                    fallback_reason="",
                ),
            )

            saved = session.repository.segments(chapter_id)
            self.assertEqual((300, 900), (saved[0].start_ms, saved[0].end_ms))
            self.assertEqual((1_300, 1_900), (saved[1].start_ms, saved[1].end_ms))
            self.assertEqual(950, window._nearest_silence(1_000, "end"))
            self.assertEqual(1_250, window._nearest_silence(1_200, "start"))
            window.close()

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

    def test_segment_switch_updates_only_the_visible_content_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("visible-view", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "first", 1_000, 2_000),
                TextSegment(None, chapter_id, 1, "second", 2_000, 3_000),
            ])
            window = MainWindow()
            window._set_session(session)
            window.content_tabs.setCurrentWidget(window.segment_table)
            with (
                patch.object(window.article_view, "focus_segment") as article_focus,
                patch.object(window.original_book_view, "focus_segment") as book_focus,
                patch.object(window.asr_comparison, "focus_segment") as asr_focus,
            ):
                window._select_segment_row(1)
                article_focus.assert_not_called()
                book_focus.assert_not_called()
                asr_focus.assert_not_called()

                window.content_tabs.setCurrentWidget(window.article_view)
                article_focus.assert_called_once_with(1, ensure_visible=True)
                book_focus.assert_not_called()
                asr_focus.assert_not_called()

                article_focus.reset_mock()
                window.content_tabs.setCurrentWidget(window.asr_comparison)
                asr_focus.assert_called_once_with(1, ensure_visible=True)
                article_focus.assert_not_called()
                book_focus.assert_not_called()
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

    def test_m4b_slice_seek_waits_for_load_and_reuses_same_asset_source(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtMultimedia import QMediaPlayer

        class FakePlayer:
            def __init__(self) -> None:
                self.url = QUrl()
                self.status = QMediaPlayer.MediaStatus.LoadingMedia
                self.positions: list[int] = []
                self.sources: list[str] = []
                self.current_position = 0
                self.state = QMediaPlayer.PlaybackState.PausedState
                self.events: list[tuple[str, int | None]] = []

            def source(self):
                return self.url

            def setSource(self, url):
                self.url = url
                self.sources.append(url.toLocalFile())

            def mediaStatus(self):
                return self.status

            def setPosition(self, value):
                self.current_position = value
                self.positions.append(value)
                self.events.append(("position", value))

            def position(self):
                return self.current_position

            def playbackState(self):
                return self.state

            def play(self):
                self.state = QMediaPlayer.PlaybackState.PlayingState
                self.events.append(("play", None))

            def pause(self):
                self.state = QMediaPlayer.PlaybackState.PausedState
                self.events.append(("pause", None))

        window = MainWindow()
        real_player = window.player
        fake = FakePlayer()
        window.player = fake
        asset = AudioAsset(7, "book.m4b", duration_ms=10_000)
        first = ChapterAudioLink(None, 1, 7, 0, 1_000, 2_000, 1.0)
        second = ChapterAudioLink(None, 1, 7, 1, 5_000, 6_000, 1.0)
        path = Path("book.m4b")
        window.current_parts = [
            (first, asset, path, 0, 1_000),
            (second, asset, path, 1_000, 2_000),
        ]
        window.current_link = first
        window.current_asset = asset

        window._request_media_position(asset, path, 1_000, autoplay=False)
        self.assertEqual([], fake.positions)
        fake.status = QMediaPlayer.MediaStatus.LoadedMedia
        window._media_status_changed(fake.status)
        self.assertEqual([1_000], fake.positions)

        window._pending_media_position = None
        fake.sources.clear()
        fake.state = QMediaPlayer.PlaybackState.PlayingState
        fake.events.clear()
        window._seek_local(1_500)
        self.assertIs(second, window.current_link)
        self.assertEqual(5_500, fake.positions[-1])
        self.assertEqual([], fake.sources)
        self.assertEqual(
            [("pause", None), ("position", 5_500), ("play", None)],
            fake.events,
        )
        window._on_position(0)
        self.assertEqual(5_500, window._pending_media_position)
        window._on_position(5_500)
        self.assertIsNone(window._pending_media_position)

        # A silent/failed deep seek may make Qt report resource position zero.
        # The editor must retain the explicit chapter-local cursor for a
        # subsequent alignment command instead of reporting "no start".
        fake.current_position = 0
        self.assertEqual(1_500, window._local_position())

        fake.state = QMediaPlayer.PlaybackState.PlayingState
        window._audio_watch_started_at = time.monotonic() - 4.0
        window._last_audio_buffer_at = 0.0
        with patch.object(window, "_ensure_m4b_proxy", return_value=True) as ensure_proxy:
            window._check_m4b_audio_output()
            ensure_proxy.assert_called_once()

        # A proxy/load task that finishes after the user cancelled playback
        # must not be allowed to start audio tens of seconds later.
        window._playback_requested = False
        fake.state = QMediaPlayer.PlaybackState.PlayingState
        window._play_state_changed(QMediaPlayer.PlaybackState.PlayingState)
        self.app.processEvents()
        self.assertEqual(QMediaPlayer.PlaybackState.PausedState, fake.state)

        window.player = real_player
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

    def test_cursor_split_uses_chapter_anchor_coordinates_not_fragment_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("cursor-anchor", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "Alpha.", 0, 1_000, source_start_char=0, source_end_char=6),
                TextSegment(None, chapter_id, 1, "Second half.", 5_000, 8_000, source_start_char=0, source_end_char=12),
            ])
            stored = session.repository.segments(chapter_id)
            session.repository.replace_anchors(chapter_id, [
                TextAudioAnchor(None, chapter_id, stored[0].id, 0, 5, 0, 900, 0.9, "asr-word"),
                TextAudioAnchor(None, chapter_id, stored[1].id, 6, 12, 5_000, 6_000, 0.9, "asr-word"),
                TextAudioAnchor(None, chapter_id, stored[1].id, 13, 17, 6_500, 7_500, 0.9, "asr-word"),
            ])
            window = MainWindow()
            window._set_session(session)
            window._select_segment_row(1)
            cursor = window.editor_text.textCursor()
            cursor.setPosition(len("Second"))
            window.editor_text.setTextCursor(cursor)
            window.split_segment_at_text_cursor()
            self.assertEqual(3, window.segment_model.rowCount())
            self.assertEqual(6_000, window.segment_model.segments[1].end_ms)
            self.assertEqual(6_000, window.segment_model.segments[2].start_ms)
            window.close()

    def test_find_window_supports_regular_expressions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("find-regex", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "Nothing here."),
                TextSegment(None, chapter_id, 1, "Order AB-2048 is ready."),
            ])
            window = MainWindow()
            window._set_session(session)
            window.show_find_dialog()
            window._perform_find(r"AB-\d+", True, True, False, "chapter", False)
            self.assertEqual(1, window.segment_table.currentIndex().row())
            self.assertEqual("AB-2048", window.editor_text.textCursor().selectedText())
            window._perform_find("[", True, False, False, "chapter", False)
            self.assertIn("正则表达式错误", window._find_dialog.status_label.text())
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

    def test_asr_comparison_uses_word_anchors_for_paired_rows(self) -> None:
        view = ASRComparisonView()
        segments = [
            TextSegment(10, 1, 0, "first source", 0, 2_000),
            TextSegment(11, 1, 1, "second source", 0, 2_000),
        ]
        tokens = [
            ASRToken(None, 1, 0, "alpha", 100, 200, 0.9),
            ASRToken(None, 1, 1, "beta", 300, 400, 0.9),
        ]
        anchors = [
            TextAudioAnchor(None, 1, 10, 0, 5, 100, 200, 1.0, "asr-word"),
            TextAudioAnchor(None, 1, 11, 6, 10, 300, 400, 1.0, "asr-word"),
        ]
        view.set_content(segments, tokens, anchors)
        rendered = view.comparison.toPlainText()
        self.assertLess(rendered.index("first source"), rendered.index("alpha"))
        self.assertLess(rendered.index("alpha"), rendered.index("second source"))
        self.assertLess(rendered.index("second source"), rendered.index("beta"))

    def test_asr_word_click_only_seeks_and_double_click_centres(self) -> None:
        window = MainWindow()
        window.spectrogram.view_start = 10_000
        window.spectrogram.view_end = 30_000
        with patch.object(window, "_seek_local") as seek, patch.object(
            window.spectrogram, "focus_time"
        ) as focus:
            window._asr_word_seek(42_500)
            seek.assert_called_once_with(42_500)
            focus.assert_not_called()
            window._asr_word_jump(42_500)
            self.assertEqual(2, seek.call_count)
            focus.assert_called_once_with(42_500, 20_000)
        window.close()

    def test_original_book_click_locates_and_double_click_centres(self) -> None:
        window = MainWindow()
        window._session_generation = 4
        window.segment_model.set_segments([
            TextSegment(77, 1, 0, "source", 12_000, 14_000),
        ])
        window.spectrogram.view_start = 0
        window.spectrogram.view_end = 10_000
        with patch.object(window, "_audio_time_activated") as locate, patch.object(
            window.spectrogram, "focus_time"
        ) as focus:
            window._original_book_segment_activated(77, 4, False)
            locate.assert_called_once_with(12_000, 0)
            focus.assert_not_called()
            window._original_book_segment_activated(77, 4, True)
            self.assertEqual(2, locate.call_count)
            focus.assert_called_once_with(12_000, 10_000)
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

    def test_punctuation_split_uses_unsaved_fixed_editor_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("live-punctuation-split", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "Old text without a boundary", 0, 2_000)
            ])
            window = MainWindow()
            window._set_session(session)
            window._select_segment_row(0)
            window.editor_text.setPlainText("First sentence. Second sentence.")
            window.split_segment_by_punctuation()
            self.assertEqual(2, window.segment_model.rowCount())
            self.assertEqual("First sentence.", window.segment_model.segments[0].text)
            self.assertEqual("Second sentence.", window.segment_model.segments[1].text)
            window.close()

    def test_contiguous_multi_selection_merges_in_one_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("multi-merge", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "First", 100, 500, 0.9, SegmentStatus.AUTO),
                TextSegment(None, chapter_id, 1, "second", 500, 900, 0.8, SegmentStatus.AUTO),
                TextSegment(None, chapter_id, 2, "third.", 900, 1_300, 0.7, SegmentStatus.AUTO),
                TextSegment(None, chapter_id, 3, "Keep me", 1_400, 1_800, 0.9, SegmentStatus.AUTO),
            ])
            window = MainWindow()
            window._set_session(session)
            selection = window.segment_table.selectionModel()
            selection.clearSelection()
            for row in range(3):
                selection.select(
                    window.segment_model.index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            window.merge_selected_segments()
            stored = session.repository.segments(chapter_id)
            self.assertEqual(2, len(stored))
            self.assertEqual("First second third.", stored[0].text)
            self.assertEqual((100, 1_300), (stored[0].start_ms, stored[0].end_ms))
            self.assertEqual("Keep me", stored[1].text)
            window.undo()
            self.assertEqual(4, window.segment_model.rowCount())
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

            # An unmatched text row is not a timing boundary. The current cue
            # must still be clamped by the next cue that actually has timing.
            window.segment_model.set_segments([
                TextSegment(None, 1, 0, "a", 0, 1000),
                TextSegment(None, 1, 1, "unmatched", 0, 0, status=SegmentStatus.UNMATCHED),
                TextSegment(None, 1, 2, "c", 2000, 3000),
            ])
            window.spectrogram.set_segments(window.segment_model.segments)
            window.overlap_policy_combo.setCurrentIndex(
                window.overlap_policy_combo.findData(SegmentOverlapPolicy.CLAMP_CURRENT)
            )
            window._boundary_moved(0, "end", 2500, False)
            self.assertEqual(2000, window.segment_model.segments[0].end_ms)
            self.assertEqual(0, window.segment_model.segments[1].end_ms)
            self.assertEqual(2000, window.segment_model.segments[2].start_ms)
        finally:
            window.close()

    def test_multi_selection_context_and_selection_edge_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("multi-context", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "first", 0, 1000),
                TextSegment(None, chapter_id, 1, "second", 1500, 2500),
                TextSegment(None, chapter_id, 2, "third", 3000, 4000),
            ])
            window = MainWindow()
            window._set_session(session)
            window.resize(1000, 700)
            window.show()
            self.app.processEvents()
            window._select_segment_rows([1, 2], current_row=1)
            window.spectrogram.set_selection(1200, 4500)

            point = window.segment_table.visualRect(window.segment_model.index(2, 0)).center()
            menu = window._segment_table_context_menu(point, execute=False)
            captured = [action.text() for action in menu.actions()]

            self.assertEqual([1, 2], window._selected_rows())
            self.assertEqual({1, 2}, window.spectrogram.selected_segments)
            self.assertIn("Qwen 强制对齐所选 2 句 ↔ 音频选区", captured)
            self.assertIn("将首句开始设为选区开始", captured)
            self.assertNotIn("编辑文本", captured)

            window.set_boundary_from_selection("start")
            window.set_boundary_from_selection("end")
            self.assertEqual(1200, window.segment_model.segments[1].start_ms)
            self.assertEqual(4500, window.segment_model.segments[2].end_ms)
            stored = session.repository.segments(chapter_id)
            self.assertEqual((1200, 4500), (stored[1].start_ms, stored[2].end_ms))
            window.close()

    def test_qwen_range_alignment_persists_anchor_method(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("qwen-range", root / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "hello", 0, 900),
                TextSegment(None, chapter_id, 1, "world", 900, 1800),
            ])
            window = MainWindow()
            window._set_session(session)
            asset = AudioAsset(1, str(root / "audio.wav"), duration_ms=3_000)
            link = ChapterAudioLink(None, chapter_id, 1, 0, 0, 3_000, 1.0)
            window.current_parts = [(link, asset, root / "audio.wav", 0, 3_000)]
            window._select_segment_rows([0, 1], current_row=0)
            window.spectrogram.set_selection(100, 2_100)

            class FakeAligner:
                def __init__(self):
                    self.last_device_info = InferenceDeviceInfo(
                        "qwen", "forced", actual_device="cpu", compute_type="float32"
                    )

                def align(self, _path, _text, _language, chapter, _options, progress):
                    progress(1.0, "done")
                    return [
                        ASRToken(None, chapter, 0, "hello", 0, 800, 0.95),
                        ASRToken(None, chapter, 1, "world", 900, 1_800, 0.95),
                    ]

            ready = SimpleNamespace(
                runtime_available=True,
                cuda_available=True,
                message="ready",
            )

            def run_now(_name, function, finished, **_kwargs):
                finished(function(lambda *_args: None))

            with (
                patch("audioalign.gui.main_window.runtime_status", return_value=ready),
                patch("audioalign.gui.main_window.Qwen3ForcedAligner", FakeAligner),
                patch.object(window, "_selected_language_code", return_value="en"),
                patch.object(window.tasks, "submit", side_effect=run_now),
            ):
                window.qwen_align_selected_range()

            anchors = session.repository.anchors(chapter_id)
            self.assertTrue(anchors)
            self.assertTrue(all(
                anchor.method == "qwen-forced-aligner-range" for anchor in anchors
            ))
            window.close()

    def test_qwen_anchor_alignment_uses_shared_multi_sentence_block_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("qwen-vad-progressive", root / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "first sentence", 0, 0),
                TextSegment(None, chapter_id, 1, "second sentence", 0, 0),
            ])
            window = MainWindow()
            window._set_session(session)
            asset = AudioAsset(1, str(root / "audio.wav"), duration_ms=8_000)
            link = ChapterAudioLink(None, chapter_id, 1, 0, 0, 8_000, 1.0)
            window.current_parts = [(link, asset, root / "audio.wav", 0, 8_000)]
            window.silence_candidates = [
                BoundaryCandidate(1_500, 0.95, start_ms=1_350, end_ms=1_650),
                BoundaryCandidate(3_200, 0.9, start_ms=3_050, end_ms=3_350),
                BoundaryCandidate(5_000, 0.8, start_ms=4_850, end_ms=5_150),
            ]
            window._select_segment_row(0)

            class FakeAligner:
                def __init__(self):
                    self.last_device_info = InferenceDeviceInfo(
                        "qwen", "forced", actual_device="cpu", compute_type="float32"
                    )

                def align(self, _path, text, _language, chapter, _options, progress):
                    progress(1.0, "done")
                    return [
                        ASRToken(None, chapter, index, line, 100 + index * 1_200,
                                 1_100 + index * 1_200, 0.95)
                        for index, line in enumerate(text.splitlines())
                    ]

            ready = SimpleNamespace(runtime_available=True, cuda_available=True, message="ready")

            def run_now(_name, function, finished, **_kwargs):
                finished(function(lambda *_args: None))

            with (
                patch("audioalign.gui.main_window.runtime_status", return_value=ready),
                patch("audioalign.gui.main_window.Qwen3ForcedAligner", FakeAligner),
                patch.object(window, "_selected_language_code", return_value="en"),
                patch.object(window, "_local_position", return_value=100),
                patch.object(window.tasks, "submit", side_effect=run_now),
            ):
                window.qwen_align_from_current_anchor()

            aligned = session.repository.segments(chapter_id)
            self.assertEqual((200, 1_200), (aligned[0].start_ms, aligned[0].end_ms))
            self.assertEqual((1_400, 2_400), (aligned[1].start_ms, aligned[1].end_ms))
            anchors = session.repository.anchors(chapter_id)
            self.assertTrue(anchors)
            self.assertTrue(all(
                anchor.method == "qwen-forced-aligner-block" for anchor in anchors
            ))
            window.close()

    def test_qwen_block_failure_preserves_timing_and_marks_review_without_asr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("qwen-block-failure", root / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "first sentence", 100, 900, 0.9, SegmentStatus.AUTO),
                TextSegment(None, chapter_id, 1, "second sentence", 1_000, 1_800, 0.9, SegmentStatus.AUTO),
            ])
            window = MainWindow()
            window._set_session(session)
            asset = AudioAsset(1, str(root / "audio.wav"), duration_ms=8_000)
            link = ChapterAudioLink(None, chapter_id, 1, 0, 0, 8_000, 1.0)
            window.current_parts = [(link, asset, root / "audio.wav", 0, 8_000)]
            window.silence_candidates = [
                BoundaryCandidate(3_200, 0.9, start_ms=3_000, end_ms=3_400),
            ]
            window._select_segment_row(0)

            class EmptyAligner:
                def __init__(self):
                    self.last_device_info = InferenceDeviceInfo(
                        "qwen", "forced", actual_device="cpu", compute_type="float32"
                    )

                def align(self, *_args, **_kwargs):
                    return []

            ready = SimpleNamespace(runtime_available=True, cuda_available=True, message="ready")

            def run_now(_name, function, finished, **_kwargs):
                finished(function(lambda *_args: None))

            with (
                patch("audioalign.gui.main_window.runtime_status", return_value=ready),
                patch("audioalign.gui.main_window.Qwen3ForcedAligner", EmptyAligner),
                patch("audioalign.gui.main_window.transcriber_for_options") as asr_factory,
                patch.object(window, "_selected_language_code", return_value="en"),
                patch.object(window, "_local_position", return_value=0),
                patch.object(window.tasks, "submit", side_effect=run_now),
            ):
                window.qwen_align_from_current_anchor()
                asr_factory.assert_not_called()

            aligned = session.repository.segments(chapter_id)
            self.assertEqual((100, 900), (aligned[0].start_ms, aligned[0].end_ms))
            self.assertEqual(SegmentStatus.LOW_CONFIDENCE, aligned[0].status)
            self.assertIn("部分完成", window.status_stage.text())
            window.close()

    def test_qwen_chapter_start_resynchronizes_without_language_special_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("qwen-chapter-prefix", root / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "Two", 0))
            session.repository.replace_segments(chapter_id, [
                TextSegment(None, chapter_id, 0, "Two"),
                TextSegment(None, chapter_id, 1, "I"),
                TextSegment(None, chapter_id, 2, "Outside Oakbridge station a little group stood waiting."),
                TextSegment(None, chapter_id, 3, "Behind them stood porters with suitcases."),
            ])
            window = MainWindow()
            window._set_session(session)
            asset = AudioAsset(1, str(root / "audio.m4b"), duration_ms=20_000)
            link = ChapterAudioLink(None, chapter_id, 1, 0, 0, 20_000, 1.0)
            window.current_parts = [(link, asset, root / "audio.m4b", 0, 20_000)]
            window.silence_candidates = [
                BoundaryCandidate(8_000, 0.95, start_ms=7_500, end_ms=8_400),
                BoundaryCandidate(14_000, 0.9, start_ms=13_500, end_ms=14_400),
            ]

            class ResynchronizingAligner:
                def __init__(self):
                    self.last_device_info = InferenceDeviceInfo(
                        "qwen", "forced", actual_device="cpu", compute_type="float32"
                    )

                def align(self, _path, text, _language, chapter, _options, progress):
                    progress(1.0, "done")
                    if not text.startswith("Outside Oakbridge"):
                        return []
                    return [
                        ASRToken(None, chapter, index, line, 300 + index * 2_000,
                                 1_700 + index * 2_000, 0.9)
                        for index, line in enumerate(text.splitlines())
                    ]

            ready = SimpleNamespace(runtime_available=True, cuda_available=True, message="ready")

            def run_now(_name, function, finished, **_kwargs):
                finished(function(lambda *_args: None))

            with (
                patch("audioalign.gui.main_window.runtime_status", return_value=ready),
                patch("audioalign.gui.main_window.Qwen3ForcedAligner", ResynchronizingAligner),
                patch.object(window, "_selected_language_code", return_value="en"),
                patch.object(window.tasks, "submit", side_effect=run_now),
                patch.object(window, "_show_error") as show_error,
            ):
                window.qwen_align_chapter()

            aligned = session.repository.segments(chapter_id)
            self.assertEqual(SegmentStatus.UNMATCHED, aligned[0].status)
            self.assertEqual(SegmentStatus.UNMATCHED, aligned[1].status)
            self.assertGreater(aligned[2].end_ms, aligned[2].start_ms)
            show_error.assert_not_called()
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

        editor.set_selection(1_000, 4_000)
        editor.set_selection_edge("start", 2_000)
        editor.set_selection_edge("end", 5_000)
        self.assertEqual(
            (2_000, 5_000),
            (editor.selection.start_ms, editor.selection.end_ms),
        )
        jumps: list[int] = []
        editor.seekRequested.connect(jumps.append)
        editor.jump_to_selection_edge("start")
        editor.jump_to_selection_edge("end")
        self.assertEqual([2_000, 5_000], jumps)
        menu = editor._context_menu(editor.wave_pane.plot, QPoint(10, 10), execute=False)
        menu_labels = [action.text() for action in menu.actions()]
        self.assertIn("将此处设为音频选区开始", menu_labels)
        self.assertIn("跳到音频选区结束", menu_labels)

        editor.wave_pane.plot.setYRange(-4, 4, padding=0)
        editor.spectrum_pane.plot.setYRange(500, 2_000, padding=0)
        editor.reset_vertical_scale()
        wave_range = editor.wave_pane.plot.viewRange()[1]
        spectrum_range = editor.spectrum_pane.plot.viewRange()[1]
        self.assertAlmostEqual(-1.05, wave_range[0], places=2)
        self.assertAlmostEqual(1.05, wave_range[1], places=2)
        self.assertAlmostEqual(50, spectrum_range[0], places=1)
        self.assertAlmostEqual(8_000, spectrum_range[1], places=1)

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
            window.content_tabs.setCurrentWidget(window.article_view)
            self.assertGreater(window.article_view.line_model.rowCount(), 0)
            window.spectrogram.set_selection(250, 750)
            self.assertEqual((250, 750), (window.spectrogram.selection.start_ms, window.spectrogram.selection.end_ms))
            window.silence_candidates = [
                BoundaryCandidate(400, 0.5, start_ms=300, end_ms=500),
                BoundaryCandidate(1200, 0.9, start_ms=900, end_ms=1500),
            ]
            window._set_silence_display_mode(SilenceDisplayMode.KEY)
            self.assertEqual(1, len(window.spectrogram.wave_pane.silences))
            self.assertEqual(1, len(window.spectrogram.spectrum_pane.silences))
            self.assertEqual(1, len(window.overview._silences))

            # Whole-chapter and anchored automatic alignment must share the
            # same block engine; the old proportional paths are unreachable.
            window._select_segment_row(0)
            with patch.object(window, "_start_qwen_block_alignment") as shared:
                window._qwen_align_chapter(0, 0, anchored=False)
                shared.assert_called_once()
            with patch.object(window, "_start_qwen_block_alignment") as shared:
                window.qwen_align_from_current_anchor()
                shared.assert_called_once()
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

    def test_audio_only_mapping_creates_one_empty_chapter_per_media_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("纯音频配对", root / "project")
            first_id = session.repository.add_audio(
                AudioAsset(None, str(root / "01.opus"), duration_ms=60_000, title="第一章")
            )
            second_id = session.repository.add_audio(
                AudioAsset(None, str(root / "02.opus"), duration_ms=70_000, title="第二章")
            )
            dialog = ChapterAudioMappingDialog(session)
            self.assertTrue(dialog.audio_only_hint.isVisibleTo(dialog))
            dialog.create_chapters_for_unpaired()
            self.assertEqual(2, dialog.created_audio_only_chapters)
            dialog.apply_button.click()
            chapters = session.repository.chapters()
            self.assertEqual(["第一章", "第二章"], [chapter.title for chapter in chapters])
            self.assertEqual(
                [first_id, second_id],
                [session.repository.chapter_links(chapter.id or 0)[0].audio_id for chapter in chapters],
            )
            self.assertTrue(all(not session.repository.segments(chapter.id or 0) for chapter in chapters))
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
                ["对齐当前句", "对齐所选句子 ↔ 音频选区", "块级对齐（从当前句/时间向后）"],
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

            window.content_tabs.setCurrentWidget(window.article_view)
            window._set_article_visualization_mode(AudioVisualizationMode.NONE)
            self.assertFalse(window.article_view.canvas.audio_visible)
            self.assertTrue(window.article_view.none_button.isChecked())
            window._set_article_visualization_mode(AudioVisualizationMode.WAVEFORM)
            self.assertTrue(window.article_view.canvas.audio_visible)
            self.assertEqual(AudioVisualizationMode.WAVEFORM, window.article_view.canvas.mode)
            window._set_article_visualization_mode(AudioVisualizationMode.SPECTROGRAM)
            self.assertEqual(AudioVisualizationMode.SPECTROGRAM, window.article_view.canvas.mode)
            window._set_silence_display_mode(SilenceDisplayMode.HIDDEN)
            self.assertFalse(window.spectrogram.silences_visible)
            self.assertFalse(window.overview.silences_visible)
            window._set_silence_display_mode(SilenceDisplayMode.KEY)
            self.assertTrue(window.silence_key_action.isChecked())
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

            window.content_tabs.setCurrentWidget(window.asr_comparison)
            source_scroll = window.asr_comparison.source.verticalScrollBar()
            transcript_scroll = window.asr_comparison.transcript.verticalScrollBar()
            self.assertIs(source_scroll, transcript_scroll)
            source_scroll.setRange(0, 100)
            source_scroll.setValue(50)
            self.assertEqual(50, transcript_scroll.value())

            window.asr_comparison.focus_segment(0)
            highlighted = window.asr_comparison._table.cellAt(1, 0).format().background().color()
            self.assertEqual(QColor("#fff0aa"), highlighted)
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

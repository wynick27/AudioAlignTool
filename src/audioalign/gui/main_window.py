from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QFont, QKeySequence, QTextCursor
from PySide6.QtMultimedia import QAudioBufferOutput, QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableView,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from audioalign.core.alignment import (
    align_segments_from_silence,
    anchors_from_segments_tokens,
    align_segments_to_tokens,
    ForcedAlignmentBlockResult,
    ForcedAlignmentHypothesis,
    ForcedAlignmentPlannerOptions,
    ForcedAlignmentRunResult,
    forced_alignment_overlap_score,
    forced_alignment_has_usable_coverage,
    forced_alignment_candidate_can_end_search,
    forced_alignment_hypothesis_priority,
    partition_monotonic_alignment_rows,
    estimate_spoken_duration_ms,
    forced_alignment_text_group_ends,
    forced_alignment_window_ends,
    locate_contiguous_audio_part,
    progressive_vad_next_start,
    segments_from_asr_tokens,
    silence_boundary_edges,
    snap_boundaries,
    score_forced_alignment_hypothesis,
    spoken_unit_count,
)
from audioalign.core.asr import (
    ASROptions,
    InferenceMemoryPressureError,
    QWEN_ASR_LANGUAGE_NAMES,
    QWEN_FORCED_LANGUAGE_CODES,
    Qwen3ForcedAligner,
    ensure_inference_memory_headroom,
    is_inference_out_of_memory,
    plan_recognition_chunks,
    qwen_cuda_disabled_reason,
    recognition_cache_key,
    release_inference_memory,
    RuntimeStatus,
    runtime_status,
    transcriber_for_options,
)
from audioalign.core.audio import (
    create_m4a_proxy,
    decode_audio_mono,
    detect_silence_candidates,
    key_silence_candidates,
)
from audioalign.core.epub_media_overlay import (
    EpubMediaOverlayOptions, EpubTextPolicy, export_epub_media_overlay,
)
from audioalign.core.exporters import export_html, export_json, export_subtitles
from audioalign.core.models import (
    AlignmentMode,
    ASRBackendId,
    AudioConversionPolicy,
    AudioVisualizationMode,
    AudioAsset,
    BookWorkPlan,
    Chapter,
    ChapterAudioLink,
    ChapterEditSnapshot,
    PlaybackFollowState,
    ProjectContextKey,
    RecognitionChunk,
    SegmentOrigin,
    SegmentOverlapPolicy,
    SegmentStatus,
    SilenceDisplayMode,
    SilenceAlignmentOptions,
    SilenceSettings,
    SourceFragment,
    SourceDocument,
    SourceDocumentKind,
    TaskHandle,
    TaskLane,
    TextAudioAnchor,
    TextSegment,
)
from audioalign.core.paths import ApplicationPaths, sanitize_project_name
from audioalign.core.runtime import subprocess_runtime_environment
from audioalign.gui.runtime_manager import RuntimeManagerDialog
from audioalign.core.spectrogram import (
    AudioVisualizationCache,
    build_audio_visualization_cache_from_slices,
    is_visualization_cache,
)
from audioalign.core.storage import (
    ProjectExistsError,
    ProjectRepository,
    ProjectSession,
    fingerprint_file,
    write_recognition_chunk,
)
from audioalign.core.timecode import format_time_ms
from audioalign.core.text import (
    collect_local_html_resources,
    collect_local_markdown_resources,
    cursor_split_offset,
    import_book,
    join_segment_text,
    normalize_for_match,
    preferred_split_offset,
    split_sentences_with_offsets,
)
from audioalign.core.subtitles import import_srt

from .article_spectrogram import ArticleSpectrogramView
from .asr_comparison import ASRComparisonView
from .find_dialog import FindDialog
from .mapping_dialog import ChapterAudioMappingDialog
from .original_book_view import OriginalBookView
from .segment_model import MiniSpectrogramDelegate, SegmentTableModel
from .spectrogram_editor import AudioVisualizerEditor, AudioVisualizerOverview


def _are_unedited_asr_segments(segments: list[TextSegment]) -> bool:
    """Allow cached ASR to refresh generated punctuation, never manual text."""
    return bool(segments) and all(
        segment.origin == SegmentOrigin.ASR
        and not segment.locked
        and segment.status in {SegmentStatus.AUTO, SegmentStatus.LOW_CONFIDENCE}
        for segment in segments
    )


WORKFLOW_FASTER_WHISPER = "faster-whisper-asr-align"
WORKFLOW_WHISPERX = "whisperx-asr-align"
WORKFLOW_QWEN_ASR = "qwen3-asr-align"
WORKFLOW_QWEN_FORCED = "qwen3-forced-align"


def _asr_uses_silence_snap(backend: ASRBackendId | str | None) -> bool:
    """Qwen3-ASR timestamps are model output and must not be VAD-shifted."""
    if isinstance(backend, ASRBackendId):
        backend = backend.value
    return backend != ASRBackendId.QWEN3_ASR.value

WHISPER_LANGUAGE_CODES = (
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy",
    "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn",
    "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si",
    "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue",
)


class JobCancelled(Exception):
    pass


class WorkerSignals(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class RuntimeProbeSignals(QObject):
    finished = Signal(str, object)


class TaskWorker(QRunnable):
    def __init__(self, function: Callable, paused: threading.Event) -> None:
        super().__init__()
        self.function = function
        self.paused = paused
        self.cancel_requested = False
        self.signals = WorkerSignals()

    def cancel(self) -> None:
        self.cancel_requested = True
        self.paused.set()

    def _progress(self, value: float, message: str) -> None:
        while not self.paused.wait(0.1):
            if self.cancel_requested:
                raise JobCancelled()
        if self.cancel_requested:
            raise JobCancelled()
        self.signals.progress.emit(value, message)

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(self._progress)
        except JobCancelled:
            self.signals.cancelled.emit()
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.finished.emit(result)


class TaskManager(QObject):
    started = Signal(str, int)
    progress = Signal(float, str)
    completed = Signal(str)
    failed = Signal(str, str)
    cancelled = Signal(str)
    queueChanged = Signal(int)
    laneStarted = Signal(str, str, int)
    laneProgress = Signal(str, float, str)
    laneCompleted = Signal(str, str)
    laneFailed = Signal(str, str, str)
    laneCancelled = Signal(str, str)
    laneQueueChanged = Signal(str, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pools = {lane: QThreadPool(self) for lane in TaskLane}
        for pool in self.pools.values():
            pool.setMaxThreadCount(1)
        self.queues: dict[TaskLane, list[tuple[int, int, int, str, Callable, Callable, int | None]]] = {
            lane: [] for lane in TaskLane
        }
        self.currents: dict[TaskLane, tuple[int, str, TaskWorker, int | None] | None] = {
            lane: None for lane in TaskLane
        }
        self.pause_events = {lane: threading.Event() for lane in TaskLane}
        for event in self.pause_events.values():
            event.set()
        self._sequence = 0
        self._active_generation = 0

    @property
    def current(self) -> TaskWorker | None:
        entry = self.currents[TaskLane.INFERENCE]
        return entry[2] if entry else None

    @property
    def current_name(self) -> str:
        entry = self.currents[TaskLane.INFERENCE]
        return entry[1] if entry else ""

    @property
    def queue(self):
        return self.queues[TaskLane.INFERENCE]

    def set_session_generation(self, generation: int) -> None:
        self._active_generation = generation
        for lane in TaskLane:
            self.queues[lane] = [item for item in self.queues[lane] if item[6] in {None, generation}]
            current = self.currents[lane]
            if current and current[3] not in {None, generation}:
                current[2].cancel()
            self._emit_queue_counts(lane)

    def submit(
        self,
        name: str,
        function: Callable,
        finished: Callable,
        *,
        lane: TaskLane = TaskLane.INFERENCE,
        priority: int = 0,
        session_generation: int | None = None,
    ) -> TaskHandle:
        lane = TaskLane(lane)
        self._sequence += 1
        task_id = self._sequence
        self.queues[lane].append(
            (-int(priority), self._sequence, task_id, name, function, finished, session_generation)
        )
        self.queues[lane].sort(key=lambda item: (item[0], item[1]))
        self._emit_queue_counts(lane)
        self._start_next(lane)
        return TaskHandle(task_id, lane, name)

    def _start_next(self, lane: TaskLane) -> None:
        if self.currents[lane] or not self.queues[lane]:
            return
        _priority, _sequence, task_id, name, function, callback, generation = self.queues[lane].pop(0)
        worker = TaskWorker(function, self.pause_events[lane])
        self.currents[lane] = (task_id, name, worker, generation)
        worker.signals.progress.connect(
            lambda fraction, message, source=lane: self._progress(source, fraction, message)
        )
        worker.signals.finished.connect(
            lambda result, source=lane: self._finish(source, task_id, name, callback, generation, result)
        )
        worker.signals.failed.connect(
            lambda details, source=lane: self._fail(source, task_id, name, details)
        )
        worker.signals.cancelled.connect(
            lambda source=lane: self._cancelled(source, task_id, name)
        )
        count = len(self.queues[lane]) + 1
        self.laneStarted.emit(lane.value, name, count)
        if lane == TaskLane.INFERENCE:
            self.started.emit(name, count)
        self._emit_queue_counts(lane)
        self.pools[lane].start(worker)

    def _progress(self, lane: TaskLane, fraction: float, message: str) -> None:
        self.laneProgress.emit(lane.value, fraction, message)
        if lane == TaskLane.INFERENCE:
            self.progress.emit(fraction, message)

    def _clear_current(self, lane: TaskLane, task_id: int) -> None:
        current = self.currents[lane]
        if not current or current[0] != task_id:
            return
        self.currents[lane] = None
        self._emit_queue_counts(lane)
        QTimer.singleShot(0, lambda source=lane: self._start_next(source))

    def _finish(
        self, lane: TaskLane, task_id: int, name: str, callback: Callable,
        generation: int | None, result,
    ) -> None:
        active = generation is None or generation == self._active_generation
        try:
            if active:
                callback(result)
        except Exception:
            details = traceback.format_exc()
            self.laneFailed.emit(lane.value, name, details)
            if lane == TaskLane.INFERENCE:
                self.failed.emit(name, details)
        else:
            if active:
                self.laneCompleted.emit(lane.value, name)
                if lane == TaskLane.INFERENCE:
                    self.completed.emit(name)
        self._clear_current(lane, task_id)

    def _fail(self, lane: TaskLane, task_id: int, name: str, details: str) -> None:
        current = self.currents[lane]
        active = bool(current and current[3] in {None, self._active_generation})
        if active:
            self.laneFailed.emit(lane.value, name, details)
            if lane == TaskLane.INFERENCE:
                self.failed.emit(name, details)
        self._clear_current(lane, task_id)

    def _cancelled(self, lane: TaskLane, task_id: int, name: str) -> None:
        current = self.currents[lane]
        active = bool(current and current[3] in {None, self._active_generation})
        if active:
            self.laneCancelled.emit(lane.value, name)
            if lane == TaskLane.INFERENCE:
                self.cancelled.emit(name)
        self._clear_current(lane, task_id)

    def _emit_queue_counts(self, lane: TaskLane) -> None:
        count = len(self.queues[lane]) + int(self.currents[lane] is not None)
        self.laneQueueChanged.emit(lane.value, count)
        self.queueChanged.emit(sum(
            len(self.queues[source]) + int(self.currents[source] is not None)
            for source in TaskLane
        ))

    def pause(self, paused: bool, lane: TaskLane = TaskLane.INFERENCE) -> None:
        event = self.pause_events[TaskLane(lane)]
        event.clear() if paused else event.set()

    def cancel_lane(self, lane: TaskLane) -> None:
        lane = TaskLane(lane)
        self.queues[lane].clear()
        current = self.currents[lane]
        if current:
            current[2].cancel()
        self._emit_queue_counts(lane)

    def cancel_all(self) -> None:
        for lane in TaskLane:
            self.cancel_lane(lane)


def _time(milliseconds: int) -> str:
    return format_time_ms(milliseconds)


class SegmentTextEdit(QPlainTextEdit):
    commitRequested = Signal()
    splitRequested = Signal()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride:
            # The text editor owns all keystrokes while focused. This prevents
            # single-key timing shortcuts from stealing spaces, arrows or edits.
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if (
            event.modifiers()
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
            == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
            and event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}
        ):
            self.splitRequested.emit()
            event.accept()
            return
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}):
            self.commitRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        self.commitRequested.emit()
        super().focusOutEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        action = menu.addAction("在当前光标处拆分（Ctrl+Shift+Enter）")
        action.setEnabled(not self.isReadOnly() and 0 < self.textCursor().position() < len(self.toPlainText()))
        action.triggered.connect(self.splitRequested.emit)
        menu.exec(event.globalPos())


class ShortcutSafeLineEdit(QLineEdit):
    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        return super().event(event)


class MainWindow(QMainWindow):
    AUDIO_FILTER = (
        "媒体 (*.mp3 *.m4a *.m4b *.aac *.wav *.flac *.ogg *.opus "
        "*.mp4 *.mkv *.mov *.webm);;所有文件 (*.*)"
    )

    def __init__(self) -> None:
        super().__init__()
        self.paths = ApplicationPaths.current()
        self.paths.ensure()
        self.preferences = self.paths.load_settings()
        self.setWindowTitle("AudioAlignTool")
        self.resize(1510, 940)
        geometry = self.preferences.get("main_window_geometry")
        if geometry:
            self.restoreGeometry(base64.b64decode(geometry))

        self.session: ProjectSession | None = None
        self._session_generation = 0
        self.current_chapter_id: int | None = None
        self.current_asset: AudioAsset | None = None
        self.current_link: ChapterAudioLink | None = None
        self.current_cache: AudioVisualizationCache | None = None
        self.current_parts: list[tuple[ChapterAudioLink, AudioAsset, Path, int, int]] = []
        self.silence_candidates = []
        self._history: dict[int, list[ChapterEditSnapshot]] = {}
        self._future: dict[int, list[ChapterEditSnapshot]] = {}
        self._media_revision = 0
        self._article_content_dirty = False
        self._asr_content_dirty = False
        self._original_book_content_dirty = False
        self._task_started_at = 0.0
        self._task_fraction = 0.0
        self._media_task_fraction = 0.0
        self._loop_selection = False
        self._play_range_start: int | None = None
        self._play_range_end: int | None = None
        # A sentence playback range is identified by the segment, not copied
        # timestamps. Its live boundaries may change while the user drags or
        # edits the timing fields. `_play_range_end` is reserved for a manual
        # time selection, whose range is intentionally fixed.
        self._play_range_segment_id: int | None = None
        self._play_range_segment_row: int | None = None
        self._tasks_paused = False
        self._proxy_attempted: set[int] = set()
        self._editor_loading = False
        self._editor_history_pushed = False
        self._editing_row = -1
        self._editor_cursor_row = -1
        self._editor_cursor_position = 0
        self._speed_warning_shown = False
        self._playback_row_update = False
        self._manual_selection_until = 0.0
        self._recent_audio_click_at = 0.0
        self._recent_audio_click_row = -1
        self._recent_audio_click_was_playing = False
        self._seek_generation = 0
        self._pending_seek_target: int | None = None
        self._pending_seek_deadline = 0.0
        # QMediaPlayer loads sources asynchronously. Keep the resource-absolute
        # seek until LoadedMedia instead of letting an M4B slice start at zero.
        self._pending_media_position: int | None = None
        self._pending_media_autoplay = False
        self._pending_media_deadline = 0.0
        self._player_asset_id: int | None = None
        self._logical_playhead_ms = 0
        # User intent is independent from the backend's transient state. A
        # late proxy/load callback may only play while this remains True.
        self._playback_requested = False
        self._audio_watch_started_at = 0.0
        self._last_audio_buffer_at = 0.0
        self._drag_changed_rows: set[int] = set()
        self._runtime_probe_pending: set[str] = set()
        self._runtime_probe_cache: dict[str, RuntimeStatus] = {}
        self._find_dialog: FindDialog | None = None
        self._find_state: tuple[tuple, int, int, int, int] | None = None
        self._runtime_probe_signals = RuntimeProbeSignals(self)
        self._runtime_probe_signals.finished.connect(self._runtime_probe_finished)

        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.9)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_buffer_output = QAudioBufferOutput(self)
        self.player.setAudioBufferOutput(self.audio_buffer_output)
        self.audio_buffer_output.audioBufferReceived.connect(self._audio_buffer_received)
        stored_rate = (
            1.0
            if self.preferences.get("always_start_1x", False)
            else self.preferences.get("playback_rate", 1.0)
        )
        self.playback_rate = max(0.25, min(3.0, float(stored_rate)))
        self.player.setPlaybackRate(self.playback_rate)
        if hasattr(self.player, "setPitchCompensation"):
            try:
                self.player.setPitchCompensation(True)
            except Exception:
                pass
        self.player.positionChanged.connect(self._on_position)
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.playbackStateChanged.connect(self._play_state_changed)
        self.player.errorOccurred.connect(self._player_error)
        self.audio_watchdog = QTimer(self)
        self.audio_watchdog.setInterval(400)
        self.audio_watchdog.timeout.connect(self._check_m4b_audio_output)
        self.audio_watchdog.start()

        self.tasks = TaskManager(self)
        self.tasks.laneStarted.connect(self._lane_task_started)
        self.tasks.laneProgress.connect(self._lane_task_progress)
        self.tasks.laneCompleted.connect(self._lane_task_completed)
        self.tasks.laneFailed.connect(self._lane_task_failed)
        self.tasks.laneCancelled.connect(self._lane_task_cancelled)
        self.tasks.laneQueueChanged.connect(self._lane_queue_changed)

        self._build_actions()
        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()

        self.editor_commit_timer = QTimer(self)
        self.editor_commit_timer.setSingleShot(True)
        self.editor_commit_timer.setInterval(350)
        self.editor_commit_timer.timeout.connect(self._commit_segment_editor)

        self.original_book_refresh_timer = QTimer(self)
        self.original_book_refresh_timer.setSingleShot(True)
        self.original_book_refresh_timer.setInterval(180)
        self.original_book_refresh_timer.timeout.connect(self._refresh_original_book)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(1500)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()
        self.task_clock = QTimer(self)
        self.task_clock.setInterval(500)
        self.task_clock.timeout.connect(self._update_task_clock)
        self.task_clock.start()
        self.status_stage.setText("就绪：新建项目、导入电子书，或直接导入音频")

    def _build_actions(self) -> None:
        self.new_action = QAction("新建项目", self, shortcut=QKeySequence.StandardKey.New, triggered=self.new_project)
        self.open_action = QAction("打开 .aatproj…", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_archive)
        self.open_folder_action = QAction("打开项目文件夹…", self, triggered=self.open_folder)
        self.library_action = QAction("打开程序项目库…", self, triggered=self.open_library)
        self.save_action = QAction("保存", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.save_project)
        self.save_archive_action = QAction("另存为 .aatproj…", self, triggered=self.save_as_archive)
        self.save_folder_action = QAction("另存为文件夹…", self, triggered=self.save_as_folder)
        self.import_book_action = QAction("导入 TXT/MD/HTML/EPUB…", self, triggered=self.import_text)
        self.import_srt_action = QAction("导入 SRT…", self, triggered=self.import_srt)
        self.import_audio_action = QAction("导入音频/视频/M4B…", self, triggered=self.import_audio)
        self.mapping_action = QAction("章节与音频配对…", self, triggered=self.edit_mappings)
        self.recognize_action = QAction("识别并对齐当前章节", self, triggered=self.recognize_current)
        self.recognize_book_action = QAction(
            "按当前识别方式处理全书（使用缓存）", self,
            triggered=lambda: self.recognize_book(force=False),
        )
        self.refresh_book_action = QAction(
            "按当前识别方式重新处理全书…", self,
            triggered=lambda: self.recognize_book(force=True),
        )
        self.detect_book_silence_action = QAction(
            "检测全书静音区", self, triggered=self.detect_book_silence,
        )
        self.clear_chapter_timing_action = QAction(
            "清除本章全部时间对应（保留文本）", self,
            triggered=self.clear_chapter_timing,
        )
        self.reset_chapter_alignment_action = QAction(
            "重置本章字幕并重新匹配…", self,
            triggered=self.reset_chapter_and_rematch,
        )
        self.export_html_action = QAction("导出 HTML…", self, triggered=lambda: self.export_output("html"))
        self.export_srt_action = QAction("导出 SRT…", self, triggered=lambda: self.export_output("srt"))
        self.export_vtt_action = QAction("导出 VTT…", self, triggered=lambda: self.export_output("vtt"))
        self.export_json_action = QAction("导出 JSON…", self, triggered=lambda: self.export_output("json"))
        self.export_epub_action = QAction(
            "导出 EPUB 3 Media Overlays…", self, triggered=self.export_epub_media_overlays,
        )
        self.undo_action = QAction("撤销", self, shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo)
        self.redo_action = QAction("重做", self, shortcut=QKeySequence.StandardKey.Redo, triggered=self.redo)
        self.find_action = QAction(
            "查找…", self, shortcut=QKeySequence.StandardKey.Find, triggered=self.show_find_dialog,
        )
        self.find_next_action = QAction(
            "查找下一个", self, shortcut=QKeySequence("F3"), triggered=lambda: self.find_again(False),
        )
        self.find_previous_action = QAction(
            "查找上一个", self, shortcut=QKeySequence("Shift+F3"), triggered=lambda: self.find_again(True),
        )
        defaults = {
            "play_pause": "Space", "play_current": "F5", "loop_current": "Shift+F5",
            "set_start": "F11", "set_end": "F12", "set_end_next": "F10",
            "previous": "Up", "next": "Down", "split": "Ctrl+Alt+V",
            "seek_back": "Left", "seek_forward": "Right", "fine_back": "Alt+Left", "fine_forward": "Alt+Right",
            "new_segment": "Insert", "delete_segment": "Delete",
            "edit_text": "F2",
            "split_cursor": "Ctrl+Shift+Return",
            "merge": "Ctrl+Shift+M", "lock": "Ctrl+L", "detect_silence": "Ctrl+Alt+D",
            "auto_silence": "Ctrl+Alt+G", "waveform": "Ctrl+1", "spectrogram": "Ctrl+2",
            "combined": "Ctrl+3", "speed_down": "Ctrl+[", "speed_up": "Ctrl+]",
            "speed_reset": "Ctrl+\\", "follow": "Ctrl+Alt+C", "return_playhead": "Ctrl+Alt+Home",
        }
        configured = self.preferences.get("shortcuts", {})
        self.command_actions: dict[str, QAction] = {}

        def command(command_id: str, text: str, callback, *, checkable: bool = False) -> QAction:
            action = QAction(text, self)
            action.setCheckable(checkable)
            shortcut = configured.get(command_id, defaults.get(command_id, ""))
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)
            self.command_actions[command_id] = action
            return action

        self.play_pause_action = command("play_pause", "播放/暂停", self.toggle_play)
        self.play_current_action = command("play_current", "播放当前句", self.play_current_segment)
        self.loop_current_action = command("loop_current", "循环当前句", self.toggle_current_loop, checkable=True)
        self.set_start_action = command("set_start", "设置开始", lambda: self.set_boundary("start"))
        self.set_end_action = command("set_end", "设置结束", lambda: self.set_boundary("end"))
        self.set_end_next_action = command("set_end_next", "设置结束并前往下一句", self.set_end_and_next)
        self.previous_action = command("previous", "上一句", lambda: self._move_row(-1))
        self.next_action = command("next", "下一句", lambda: self._move_row(1))
        self.seek_back_action = command("seek_back", "后退 1 秒", lambda: self.seek_relative(-1000))
        self.seek_forward_action = command("seek_forward", "前进 1 秒", lambda: self.seek_relative(1000))
        self.fine_back_action = command("fine_back", "后退 500 毫秒", lambda: self.seek_relative(-500))
        self.fine_forward_action = command("fine_forward", "前进 500 毫秒", lambda: self.seek_relative(500))
        self.new_segment_action = command("new_segment", "用选区新建句段", self.new_segment_from_selection)
        self.edit_text_action = command("edit_text", "编辑当前句文本", self.focus_segment_editor)
        self.delete_segment_action = command("delete_segment", "移除时间匹配（保留文本）", self.delete_segments)
        self.split_action = command("split", "拆分", self.split_segment)
        self.split_cursor_action = command(
            "split_cursor", "在文本光标处拆分", self.split_segment_at_text_cursor
        )
        self.split_punctuation_action = command(
            "split_punctuation", "按标点拆成多句", self.split_segment_by_punctuation
        )
        self.restore_source_action = command(
            "restore_source", "恢复原始段落", self.restore_source_fragment
        )
        self.restore_chapter_action = command(
            "restore_source_chapter", "恢复整章原文", self.restore_source_chapter
        )
        self.merge_action = command("merge", "合并后句", lambda: self.merge_segment(1))
        self.lock_action = command("lock", "锁定/解锁", self.toggle_lock)
        self.detect_silence_action = command("detect_silence", "检测静音区", self.detect_silence)
        self.auto_silence_action = command("auto_silence", "从当前句按静音自动分配", self.auto_align_from_silence)
        self.waveform_action = command("waveform", "显示波形", lambda: self.set_visualization_mode(AudioVisualizationMode.WAVEFORM), checkable=True)
        self.spectrum_action = command("spectrogram", "显示频谱", lambda: self.set_visualization_mode(AudioVisualizationMode.SPECTROGRAM), checkable=True)
        self.combined_action = command("combined", "组合显示", lambda: self.set_visualization_mode(AudioVisualizationMode.COMBINED), checkable=True)
        self.visualization_action_group = QActionGroup(self)
        self.visualization_action_group.setExclusive(True)
        for visualization_action in (self.waveform_action, self.spectrum_action, self.combined_action):
            self.visualization_action_group.addAction(visualization_action)
        self.combined_action.setChecked(True)
        self.speed_down_action = command("speed_down", "降低播放速度", lambda: self.adjust_playback_rate(-0.05))
        self.speed_up_action = command("speed_up", "提高播放速度", lambda: self.adjust_playback_rate(0.05))
        self.speed_reset_action = command("speed_reset", "恢复正常速度", lambda: self.set_playback_rate(1.0))
        self.follow_action = command("follow", "播放头居中", self._toggle_follow_action, checkable=True)
        self.follow_action.setChecked(bool(self.preferences.get("follow_playhead", True)))
        self.return_playhead_action = command("return_playhead", "回到播放头", self.return_to_playhead)
        self.list_audio_action = command(
            "list_audio_visual", "显示列表迷你图", self._set_list_audio_visible, checkable=True
        )
        self.list_audio_action.setChecked(bool(self.preferences.get("show_list_audio_visual", True)))
        self.article_audio_action = command(
            "article_audio_visual", "显示文章行音频图", self._set_article_audio_visible, checkable=True
        )
        self.article_audio_action.setChecked(bool(self.preferences.get("show_article_audio_visual", True)))
        # Replaced by the explicit three-state groups below; keep the QAction
        # objects out of the command/shortcut registry so obsolete binary
        # visibility switches are not shown to users.
        self.command_actions.pop("list_audio_visual", None)
        self.command_actions.pop("article_audio_visual", None)
        self.list_visualization_group = QActionGroup(self)
        self.list_visualization_group.setExclusive(True)
        self.list_none_action = command(
            "list_visual_none", "句子视图：无图",
            lambda: self._set_list_visualization_mode(AudioVisualizationMode.NONE), checkable=True,
        )
        self.list_spectrum_action = command(
            "list_visual_spectrum", "句子视图：频谱",
            lambda: self._set_list_visualization_mode(AudioVisualizationMode.SPECTROGRAM), checkable=True,
        )
        self.list_waveform_action = command(
            "list_visual_waveform", "句子视图：波形",
            lambda: self._set_list_visualization_mode(AudioVisualizationMode.WAVEFORM), checkable=True,
        )
        for action in (self.list_none_action, self.list_spectrum_action, self.list_waveform_action):
            self.list_visualization_group.addAction(action)
        self.list_spectrum_action.setChecked(True)

        self.article_visualization_group = QActionGroup(self)
        self.article_visualization_group.setExclusive(True)
        self.article_none_action = command(
            "article_visual_none", "文章视图：无图",
            lambda: self._set_article_visualization_mode(AudioVisualizationMode.NONE), checkable=True,
        )
        self.article_spectrum_action = command(
            "article_visual_spectrum", "文章视图：频谱",
            lambda: self._set_article_visualization_mode(AudioVisualizationMode.SPECTROGRAM), checkable=True,
        )
        self.article_waveform_action = command(
            "article_visual_waveform", "文章视图：波形",
            lambda: self._set_article_visualization_mode(AudioVisualizationMode.WAVEFORM), checkable=True,
        )
        for action in (self.article_none_action, self.article_spectrum_action, self.article_waveform_action):
            self.article_visualization_group.addAction(action)
        self.article_spectrum_action.setChecked(True)
        self.silence_display_group = QActionGroup(self)
        self.silence_display_group.setExclusive(True)
        self.silence_hidden_action = QAction("隐藏", self, checkable=True)
        self.silence_key_action = QAction("关键", self, checkable=True)
        self.silence_all_action = QAction("全部", self, checkable=True)
        for action, mode in (
            (self.silence_hidden_action, SilenceDisplayMode.HIDDEN),
            (self.silence_key_action, SilenceDisplayMode.KEY),
            (self.silence_all_action, SilenceDisplayMode.ALL),
        ):
            self.silence_display_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, value=mode: self._set_silence_display_mode(value)
            )
        try:
            silence_mode = SilenceDisplayMode(
                self.preferences.get("silence_display_mode", SilenceDisplayMode.KEY.value)
            )
        except ValueError:
            silence_mode = SilenceDisplayMode.KEY
        {
            SilenceDisplayMode.HIDDEN: self.silence_hidden_action,
            SilenceDisplayMode.KEY: self.silence_key_action,
            SilenceDisplayMode.ALL: self.silence_all_action,
        }[silence_mode].setChecked(True)
        self.start_normal_speed_action = QAction("启动时使用 1.0×", self, checkable=True)
        self.start_normal_speed_action.setChecked(bool(self.preferences.get("always_start_1x", False)))
        self.start_normal_speed_action.toggled.connect(
            lambda enabled: self.preferences.__setitem__("always_start_1x", enabled)
        )

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addActions([self.new_action, self.open_action, self.open_folder_action, self.library_action])
        file_menu.addSeparator()
        file_menu.addActions([self.save_action, self.save_archive_action, self.save_folder_action])
        file_menu.addSeparator()
        file_menu.addActions([self.import_book_action, self.import_srt_action, self.import_audio_action])
        export_menu = file_menu.addMenu("导出")
        export_menu.addActions([
            self.export_html_action, self.export_srt_action, self.export_vtt_action,
            self.export_json_action, self.export_epub_action,
        ])
        edit_menu = self.menuBar().addMenu("编辑")
        edit_menu.addActions([self.undo_action, self.redo_action])
        edit_menu.addSeparator()
        edit_menu.addActions([self.find_action, self.find_next_action, self.find_previous_action])
        edit_menu.addSeparator()
        edit_menu.addAction(self.mapping_action)
        recognize_menu = self.menuBar().addMenu("识别")
        recognize_menu.addAction(self.recognize_action)
        recognize_menu.addSeparator()
        recognize_menu.addActions([
            self.clear_chapter_timing_action,
            self.reset_chapter_alignment_action,
        ])
        book_menu = recognize_menu.addMenu("全书批量处理")
        book_menu.addActions([
            self.recognize_book_action,
            self.refresh_book_action,
            self.detect_book_silence_action,
        ])
        options_menu = self.menuBar().addMenu("选项")
        options_menu.addAction("快捷键…", self.edit_shortcuts)
        options_menu.addAction("运行时组件…", self.manage_runtime_components)
        options_menu.addAction(self.start_normal_speed_action)
        view_menu = self.menuBar().addMenu("视图")
        list_menu = view_menu.addMenu("句子视图音频图")
        list_menu.addActions([self.list_none_action, self.list_spectrum_action, self.list_waveform_action])
        article_menu = view_menu.addMenu("文章视图音频图")
        article_menu.addActions([self.article_none_action, self.article_spectrum_action, self.article_waveform_action])
        silence_menu = view_menu.addMenu("静音显示")
        silence_menu.addActions([
            self.silence_hidden_action, self.silence_key_action, self.silence_all_action,
        ])

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.addActions([self.new_action, self.open_action, self.save_action])
        toolbar.addSeparator()
        toolbar.addActions([self.import_book_action, self.import_audio_action, self.mapping_action, self.recognize_action])
        self.addToolBar(toolbar)

    def manage_runtime_components(self) -> None:
        RuntimeManagerDialog(self.paths, self).exec()
        self._runtime_probe_cache.clear()
        self._update_model_status()

    def _tool_button(self, text: str, tooltip: str, callback=None, *, checkable: bool = False) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip.split("（", 1)[0])
        button.setStatusTip(tooltip)
        button.setCheckable(checkable)
        font = QFont()
        font.setFamilies(["Segoe UI Symbol", "Segoe UI Emoji", "Microsoft YaHei UI"])
        font.setPointSize(12)
        button.setFont(font)
        # Reserve enough room for multi-character symbols such as "⟦⇥" and
        # "1×"; a fixed 32 px button is elided by Qt to "...".
        button_width = max(32, min(54, button.fontMetrics().horizontalAdvance(text) + 14))
        button.setFixedSize(button_width, 30)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        if callback:
            button.clicked.connect(callback)
        return button

    def _action_button(self, text: str, action: QAction) -> QToolButton:
        shortcut = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        tooltip = action.text() + (f"（{shortcut}）" if shortcut else "")
        button = self._tool_button(text, tooltip, checkable=action.isCheckable())
        # setDefaultAction() would copy the full menu caption back into this
        # compact button whenever QAction changes, producing a visible "...".
        def trigger_action(_checked: bool = False) -> None:
            action.trigger()
            if action.isCheckable():
                button.setChecked(action.isChecked())

        button.clicked.connect(trigger_action)
        if action.isCheckable():
            button.setChecked(action.isChecked())
            action.toggled.connect(button.setChecked)
        button.setToolTip(tooltip)
        button.setAccessibleName(action.text())
        button.setStatusTip(action.statusTip() or action.text())
        if action.isCheckable():
            button.setStyleSheet(
                "QToolButton:checked { background:#2f6fa8; color:white; border:1px solid #77b8ef; }"
                "QToolButton[followSuspended='true'] { background:#7a5a16; color:white; border:1px solid #d7aa43; }"
            )
        return button

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Orientation.Vertical)
        upper = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.project_label = QLabel("尚未打开项目")
        self.project_label.setWordWrap(True)
        left_layout.addWidget(self.project_label)
        left_layout.addWidget(QLabel("章节 / 音频配对"))
        self.chapter_list = QListWidget()
        self.chapter_list.currentRowChanged.connect(self._chapter_selected)
        left_layout.addWidget(self.chapter_list)
        mapping_button = QPushButton("配对管理器…")
        mapping_button.clicked.connect(self.edit_mappings)
        left_layout.addWidget(mapping_button)
        upper.addWidget(left)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.segment_model = SegmentTableModel()
        self.segment_model.beforeEdit.connect(self._push_history)
        self.segment_model.segmentEdited.connect(self._segment_edited)
        self.segment_table = QTableView()
        self.segment_table.setModel(self.segment_model)
        self.mini_delegate = MiniSpectrogramDelegate(self.segment_table)
        self.segment_table.setItemDelegate(self.mini_delegate)
        self.segment_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.segment_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.segment_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.segment_table.setAlternatingRowColors(True)
        self.segment_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.segment_table.customContextMenuRequested.connect(self._segment_table_context_menu)
        self.segment_table.verticalHeader().hide()
        header = self.segment_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.segment_table.selectionModel().currentRowChanged.connect(self._segment_selected)
        self.segment_table.selectionModel().selectionChanged.connect(self._table_selection_changed)
        self.segment_table.doubleClicked.connect(lambda index: self._segment_double_activated(index.row()))
        self.article_view = ArticleSpectrogramView()
        self.article_view.segmentActivated.connect(self._select_segment_row)
        self.article_view.segmentDoubleClicked.connect(self._segment_double_activated)
        self.article_view.rangeSelected.connect(self._article_range_selected)
        self.article_view.rangeSelectionFinished.connect(self._article_range_finished)
        self.article_view.seekRequested.connect(self._article_seek)
        self.article_view.modeChanged.connect(lambda mode: self.preferences.__setitem__("article_visualization_mode", mode))
        self.article_view.modeChanged.connect(self._article_visualization_changed)
        self.content_tabs = QTabWidget()
        self.content_tabs.addTab(self.segment_table, "字幕表格")
        self.content_tabs.addTab(self.article_view, "文章频谱")
        self.original_book_view = OriginalBookView()
        self.original_book_view.segmentActivated.connect(self._original_book_segment_activated)
        self.content_tabs.addTab(self.original_book_view, "原书")
        self.asr_comparison = ASRComparisonView()
        self.asr_comparison.seekRequested.connect(self._asr_word_seek)
        self.asr_comparison.jumpRequested.connect(self._asr_word_jump)
        self.content_tabs.addTab(self.asr_comparison, "ASR 对比")
        self.content_tabs.currentChanged.connect(self._content_tab_changed)
        self.segment_table.setColumnHidden(2, not self.list_audio_action.isChecked())
        self.article_view.set_audio_visible(self.article_audio_action.isChecked())
        center_layout.addWidget(self.content_tabs)
        self._build_segment_editor(center_layout)
        upper.addWidget(center)

        right = QWidget()
        form = QFormLayout(right)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("faster-whisper · 识别后对齐", WORKFLOW_FASTER_WHISPER)
        self.mode_combo.addItem("WhisperX · 识别并精确对齐", WORKFLOW_WHISPERX)
        self.mode_combo.addItem("Qwen3-ASR · 识别后对齐", WORKFLOW_QWEN_ASR)
        self.mode_combo.addItem("Qwen ForcedAligner · 强制对齐", WORKFLOW_QWEN_FORCED)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["small", "medium", "large-v3", "turbo"])
        self.language_combo = QComboBox()
        self.language_combo.setEditable(False)
        self.vad_spin = QDoubleSpinBox()
        self.vad_spin.setRange(0.05, 0.95)
        self.vad_spin.setSingleStep(0.05)
        self.vad_spin.setValue(0.5)
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(20, 5000)
        self.min_silence_spin.setValue(350)
        self.min_silence_spin.setSuffix(" ms")
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 1000)
        self.padding_spin.setValue(80)
        self.padding_spin.setSuffix(" ms")
        self.snap_spin = QSpinBox()
        self.snap_spin.setRange(0, 5000)
        self.snap_spin.setValue(250)
        self.snap_spin.setSuffix(" ms")
        self.overlap_policy_combo = QComboBox()
        self.overlap_policy_combo.addItem("不改变相邻句（默认）", SegmentOverlapPolicy.CLAMP_CURRENT)
        self.overlap_policy_combo.addItem("自动缩放相邻句", SegmentOverlapPolicy.TRIM_NEIGHBORS)
        self.overlap_policy_combo.addItem("允许片段重叠", SegmentOverlapPolicy.ALLOW_OVERLAP)
        self.mode_combo.currentIndexChanged.connect(self._alignment_mode_changed)
        for control in (self.mode_combo, self.model_combo, self.language_combo, self.overlap_policy_combo,
                        self.vad_spin, self.min_silence_spin, self.padding_spin, self.snap_spin):
            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._settings_changed)
            else:
                control.valueChanged.connect(self._settings_changed)
        form.addRow("对齐方式", self.mode_combo)
        form.addRow("识别模型", self.model_combo)
        self.language_label = QLabel("语言")
        form.addRow(self.language_label, self.language_combo)
        form.addRow("VAD 阈值", self.vad_spin)
        form.addRow("最短静音", self.min_silence_spin)
        form.addRow("边界留白", self.padding_spin)
        form.addRow("Shift 静音吸附窗口", self.snap_spin)
        form.addRow("片段重叠", self.overlap_policy_combo)
        self.model_status_label = QLabel()
        self.model_status_label.setWordWrap(True)
        form.addRow("模型状态", self.model_status_label)
        self.recognize_button = QPushButton("识别并自动对齐")
        self.recognize_button.clicked.connect(self.recognize_current)
        self.refresh_recognition_button = QPushButton("强制重新识别")
        self.refresh_recognition_button.clicked.connect(lambda: self.recognize_current(force=True))
        self.clear_recognition_button = QPushButton("清除当前识别缓存")
        self.clear_recognition_button.clicked.connect(self.clear_recognition_cache)
        self.qwen_align_button = QToolButton()
        self.qwen_align_button.setText("Qwen 强制对齐 ▾")
        self.qwen_align_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.qwen_align_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.qwen_align_button.setToolTip("选择 Qwen ForcedAligner 的作用范围")
        self.qwen_align_menu = QMenu(self.qwen_align_button)
        self.qwen_align_menu.addAction("对齐当前句", self.qwen_align_current_segment)
        self.qwen_align_menu.addAction("对齐所选句子 ↔ 音频选区", self.qwen_align_selected_range)
        self.qwen_align_menu.addAction("块级对齐（从当前句/时间向后）", self.qwen_align_from_current_anchor)
        self.qwen_align_button.setMenu(self.qwen_align_menu)
        silence_button = QPushButton("检测静音区")
        silence_button.clicked.connect(self.detect_silence)
        apply_button = QPushButton("按静音优化未锁定边界")
        apply_button.clicked.connect(self.apply_silence)
        auto_silence_button = QPushButton("从当前句按静音自动分配")
        auto_silence_button.clicked.connect(self.auto_align_from_silence)
        form.addRow(self.recognize_button)
        form.addRow(self.refresh_recognition_button)
        form.addRow(self.clear_recognition_button)
        form.addRow(self.qwen_align_button)
        form.addRow(silence_button)
        form.addRow(apply_button)
        form.addRow(auto_silence_button)
        upper.addWidget(right)
        upper.setSizes([230, 1000, 270])

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(6, 2, 6, 6)
        self.spectrogram = AudioVisualizerEditor()
        tools = QHBoxLayout()
        self.play_button = self._action_button("▶", self.play_pause_action)
        selection_play = self._tool_button("▷", "播放选区", self.play_selection)
        self.loop_button = self._action_button("↻", self.loop_current_action)
        previous = self._action_button("⏮", self.previous_action)
        following = self._action_button("⏭", self.next_action)
        bind = self._tool_button("⤓", "将选区绑定到当前句（Enter）", self.bind_selection)
        distribute = self._tool_button("½", "平均分配给所选多句", self.distribute_selection)
        proportional = self._tool_button("%", "按已有时长比例分配", lambda: self.distribute_selection(True))
        set_start = self._action_button("⟦", self.set_start_action)
        shift_start = self._tool_button("⟦⇥", "设置开始并平移后续", lambda: self.set_boundary("start", True))
        set_end = self._action_button("⟧", self.set_end_action)
        split = self._action_button("✂", self.split_action)
        merge = self._action_button("⇥", self.merge_action)
        lock = self._action_button("🔒", self.lock_action)
        reset_view = self._tool_button("⌂", "恢复原始缩放比例", self.spectrogram.reset_view)
        self.waveform_button = self._action_button("∿", self.waveform_action)
        self.spectrum_button = self._action_button("▥", self.spectrum_action)
        self.combined_button = self._action_button("▤", self.combined_action)
        self.follow_button = self._action_button("⊙", self.follow_action)
        return_head = self._action_button("◎", self.return_playhead_action)
        for button in (self.play_button, selection_play, self.loop_button, previous, following, bind,
                       distribute, proportional, set_start, shift_start, set_end, split, merge, lock,
                       reset_view, self.waveform_button, self.spectrum_button, self.combined_button,
                       self.follow_button, return_head):
            tools.addWidget(button)
        self.speed_combo = QComboBox()
        self.speed_combo.setEditable(True)
        self.speed_combo.addItems(["0.25×", "0.50×", "0.75×", "1.00×", "1.25×", "1.50×", "2.00×", "2.50×", "3.00×"])
        self.speed_combo.setCurrentText(f"{self.playback_rate:.2f}×")
        self.speed_combo.setFixedWidth(76)
        self.speed_combo.setToolTip("播放速度")
        self.speed_combo.activated.connect(lambda _index: self._speed_combo_changed())
        self.speed_combo.lineEdit().editingFinished.connect(self._speed_combo_changed)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 300)
        self.speed_slider.setValue(round(self.playback_rate * 100))
        self.speed_slider.setFixedWidth(105)
        self.speed_slider.setToolTip("播放速度 0.25–3.00×")
        self.speed_slider.valueChanged.connect(lambda value: self.set_playback_rate(value / 100))
        speed_reset = self._action_button("1×", self.speed_reset_action)
        tools.addWidget(self.speed_combo)
        tools.addWidget(self.speed_slider)
        tools.addWidget(speed_reset)
        tools.addStretch()
        self.time_label = QLabel("00:00:00.000 / 00:00:00.000")
        tools.addWidget(self.time_label)
        bottom_layout.addLayout(tools)
        self.spectrogram.seekRequested.connect(self._audio_seek_requested)
        self.spectrogram.selectionChanged.connect(self._selection_changed)
        self.spectrogram.timeActivated.connect(self._audio_time_activated)
        self.spectrogram.segmentDoubleClickedAt.connect(self._segment_double_activated)
        self.spectrogram.segmentSelected.connect(self._audio_segment_selected)
        self.spectrogram.boundaryDragStarted.connect(self._begin_segment_drag)
        self.spectrogram.boundaryMoved.connect(self._boundary_moved)
        self.spectrogram.segmentShiftRequested.connect(self._shift_segment)
        self.spectrogram.segmentEditFinished.connect(self._commit_dragged_segment)
        self.spectrogram.viewChanged.connect(self._spectrum_view_changed)
        self.spectrogram.followStateChanged.connect(self._follow_state_changed)
        self.spectrogram.bindSelectionRequested.connect(self.bind_selection)
        self.spectrogram.distributeSelectionRequested.connect(self.distribute_selection)
        self.spectrogram.selectionStartRequested.connect(
            lambda: self.set_boundary_from_selection("start")
        )
        self.spectrogram.selectionEndRequested.connect(
            lambda: self.set_boundary_from_selection("end")
        )
        self.spectrogram.forceAlignSelectionRequested.connect(self.qwen_align_selected_range)
        self.spectrogram.newSegmentRequested.connect(self.new_segment_from_selection)
        self.spectrogram.splitRequested.connect(self.split_segment)
        self.spectrogram.mergePreviousRequested.connect(lambda: self.merge_segment(-1))
        self.spectrogram.mergeNextRequested.connect(lambda: self.merge_segment(1))
        self.spectrogram.deleteRequested.connect(self.delete_segments)
        self.spectrogram.clearTimingRequested.connect(self.clear_timing)
        self.spectrogram.nextSilenceRequested.connect(self.find_next_silence)
        self.spectrogram.playCurrentRequested.connect(self.play_current_segment)
        self.spectrogram.editTextRequested.connect(self.focus_segment_editor)
        self.spectrogram.splitPunctuationRequested.connect(self.split_segment_by_punctuation)
        self.spectrogram.lockRequested.connect(self.toggle_lock)
        self.spectrogram.restoreSourceRequested.connect(self.restore_source_fragment)
        self.spectrogram.insertBeforeRequested.connect(lambda: self.insert_segment(-1))
        self.spectrogram.insertAfterRequested.connect(lambda: self.insert_segment(1))
        self.overview = AudioVisualizerOverview()
        self.overview.seekRequested.connect(self._overview_jump)
        self.overview.windowRequested.connect(self._overview_window)
        self.overview.issueRequested.connect(self._overview_issue_requested)
        self.overview.interactionStarted.connect(lambda: self.spectrogram.suspend_follow("overview"))
        bottom_layout.addWidget(self.spectrogram)
        bottom_layout.addWidget(self.overview)
        root.addWidget(upper)
        root.addWidget(bottom)
        root.setSizes([600, 320])
        self.setCentralWidget(root)
        self.setStyleSheet(
            self.styleSheet()
            + "QToolButton:checked { background:#2f78bd; color:white; border:1px solid #6bb7ff; }"
        )
        saved_mode = self.preferences.get("visualization_mode", AudioVisualizationMode.COMBINED.value)
        try:
            self.set_visualization_mode(AudioVisualizationMode(saved_mode))
        except ValueError:
            self.set_visualization_mode(AudioVisualizationMode.COMBINED)
        article_mode = self.preferences.get("article_visualization_mode", AudioVisualizationMode.SPECTROGRAM.value)
        try:
            self.article_view.set_mode(AudioVisualizationMode(article_mode))
        except ValueError:
            self.article_view.set_mode(AudioVisualizationMode.SPECTROGRAM)
        list_mode = self.preferences.get("list_visualization_mode", AudioVisualizationMode.SPECTROGRAM.value)
        try:
            self._set_list_visualization_mode(AudioVisualizationMode(list_mode))
        except ValueError:
            self._set_list_visualization_mode(AudioVisualizationMode.SPECTROGRAM)
        self._article_visualization_changed(self.article_view.canvas.mode.value)
        self.spectrogram.set_follow_enabled(self.follow_action.isChecked())
        checked_silence_action = self.silence_display_group.checkedAction()
        silence_mode = {
            self.silence_hidden_action: SilenceDisplayMode.HIDDEN,
            self.silence_key_action: SilenceDisplayMode.KEY,
            self.silence_all_action: SilenceDisplayMode.ALL,
        }.get(checked_silence_action, SilenceDisplayMode.KEY)
        self._set_silence_display_mode(silence_mode)
        self._alignment_mode_changed()
        self._update_model_status()

    def _build_segment_editor(self, parent_layout: QVBoxLayout) -> None:
        editor = QWidget(self)
        editor.setObjectName("currentSegmentEditor")
        layout = QVBoxLayout(editor)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        top = QHBoxLayout()
        self.editor_info = QLabel("未选择句段")
        self.editor_info.setMinimumWidth(180)
        top.addWidget(self.editor_info)
        self.editor_lock = self._tool_button("🔒", "锁定/解锁当前句（Ctrl+L）", self.toggle_lock)
        top.addWidget(self.editor_lock)
        for label, name in (("开始", "editor_start"), ("结束", "editor_end"), ("持续", "editor_duration")):
            top.addWidget(QLabel(label))
            field = ShortcutSafeLineEdit("00:00:00.000")
            field.setFixedWidth(112)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            field.setToolTip(f"{label}时间，格式 HH:MM:SS.mmm")
            field.editingFinished.connect(self._commit_segment_editor)
            setattr(self, name, field)
            top.addWidget(field)
        top.addStretch()
        for button in (
            self._action_button("⏮", self.previous_action),
            self._action_button("▷", self.play_current_action),
            self._action_button("⏭", self.next_action),
            self._action_button("⟦", self.set_start_action),
            self._action_button("⟧", self.set_end_action),
        ):
            top.addWidget(button)
        layout.addLayout(top)
        self.editor_text = SegmentTextEdit()
        self.editor_text.setPlaceholderText("选择一句后在这里编辑文本")
        self.editor_text.setFixedHeight(76)
        self.editor_text.textChanged.connect(self._segment_editor_changed)
        self.editor_text.cursorPositionChanged.connect(self._segment_editor_cursor_changed)
        self.editor_text.commitRequested.connect(self._commit_segment_editor)
        self.editor_text.splitRequested.connect(self.split_segment_at_text_cursor)
        text_row = QHBoxLayout()
        text_row.setSpacing(4)
        text_row.addWidget(self.editor_text, 1)
        split_cursor_button = self._action_button("✂", self.split_cursor_action)
        split_cursor_button.setToolTip("在文本框当前光标处拆分（Ctrl+Shift+Enter）")
        split_cursor_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        text_row.addWidget(split_cursor_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text_row)
        parent_layout.addWidget(editor)

    def focus_segment_editor(self) -> None:
        row = self.segment_table.currentIndex().row()
        if 0 <= row < len(self.segment_model.segments):
            self._load_segment_editor(row)
            if not self.segment_model.segments[row].locked:
                self.editor_text.setFocus()

    def show_find_dialog(self) -> None:
        if self._find_dialog is None:
            self._find_dialog = FindDialog(self)
            self._find_dialog.findRequested.connect(self._perform_find)
        selected = ""
        if self.editor_text.hasFocus():
            selected = self.editor_text.textCursor().selectedText().replace("\u2029", "\n")
        self._find_dialog.show_for_search(selected)

    def find_again(self, backwards: bool = False) -> None:
        if self._find_dialog is None or not self._find_dialog.query_edit.text():
            self.show_find_dialog()
            return
        self._find_dialog.request_find(backwards)

    def _perform_find(
        self,
        query: str,
        regular_expression: bool,
        case_sensitive: bool,
        whole_word: bool,
        scope: str,
        backwards: bool,
    ) -> None:
        if self._find_dialog is None or not self.session:
            return
        if not query:
            self._find_dialog.set_status("请输入查找内容", error=True)
            return
        expression = query if regular_expression else re.escape(query)
        if whole_word:
            expression = rf"(?<!\w)(?:{expression})(?!\w)"
        try:
            pattern = re.compile(expression, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            self._find_dialog.set_status(f"正则表达式错误：{exc}", error=True)
            return

        chapters = self.session.repository.chapters()
        chapter_index = {chapter.id or 0: index for index, chapter in enumerate(chapters)}
        if scope == "project":
            sources = [
                (index, chapter.id or 0, chapter.title, self.session.repository.segments(chapter.id or 0))
                for index, chapter in enumerate(chapters)
            ]
        else:
            index = chapter_index.get(self.current_chapter_id or 0, 0)
            title = chapters[index].title if chapters and index < len(chapters) else "当前章节"
            sources = [(index, self.current_chapter_id or 0, title, self.segment_model.segments)]

        matches: list[tuple[int, int, str, int, int, int]] = []
        for chapter_row, chapter_id, title, segments in sources:
            for segment_row, segment in enumerate(segments):
                for match in pattern.finditer(segment.text):
                    if match.end() > match.start():
                        matches.append((
                            chapter_row, chapter_id, title, segment_row,
                            match.start(), match.end(),
                        ))
        if not matches:
            self._find_state = None
            self._find_dialog.set_status("没有找到匹配内容", error=True)
            return

        signature = (query, regular_expression, case_sensitive, whole_word, scope)
        current_row = self.segment_table.currentIndex().row()
        current_chapter_row = chapter_index.get(self.current_chapter_id or 0, 0)
        state = self._find_state
        state_is_current = bool(
            state and state[0] == signature
            and state[1] == (self.current_chapter_id or 0)
            and state[2] == current_row
        )
        if state_is_current:
            anchor = (
                current_chapter_row,
                current_row,
                state[3] if backwards else state[4],
            )
        else:
            anchor = (current_chapter_row, max(0, current_row), 10**9 if backwards else -1)

        selected_match = None
        ordered = reversed(matches) if backwards else iter(matches)
        for candidate in ordered:
            key = (candidate[0], candidate[3], candidate[4])
            if (backwards and key < anchor) or (not backwards and key >= anchor):
                selected_match = candidate
                break
        if selected_match is None:
            selected_match = matches[-1] if backwards else matches[0]

        chapter_row, chapter_id, title, segment_row, start, end = selected_match
        if chapter_id != self.current_chapter_id:
            self.chapter_list.setCurrentRow(chapter_row)
        self._select_segment_row(segment_row)
        cursor = self.editor_text.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        self.editor_text.setTextCursor(cursor)
        self.editor_text.ensureCursorVisible()
        self._find_state = (signature, chapter_id, segment_row, start, end)
        wrapped = (
            (not backwards and (chapter_row, segment_row, start) < anchor)
            or (backwards and (chapter_row, segment_row, start) > anchor)
        )
        suffix = "（已从头继续）" if wrapped and not backwards else ("（已从末尾继续）" if wrapped else "")
        self._find_dialog.set_status(
            f"{title} · 第 {segment_row + 1} 句 · 字符 {start + 1}–{end}{suffix}"
        )

    def _segment_table_context_menu(self, position, *, execute: bool = True) -> QMenu:
        index = self.segment_table.indexAt(position)
        if index.isValid():
            selected_before = self._selected_rows()
            if index.row() not in selected_before:
                self._select_only_segment_row(index.row())
            else:
                # Keep the multi-selection while making the clicked row the
                # current row used by single-row commands.
                self.segment_table.selectionModel().setCurrentIndex(
                    self.segment_model.index(index.row(), 0),
                    QItemSelectionModel.SelectionFlag.NoUpdate,
                )
        rows = self._selected_rows()
        count = len(rows)
        has_row = count > 0
        multi = count > 1
        selection = self.spectrogram.selection
        has_time_selection = bool(
            selection and selection.end_ms > selection.start_ms
            and not self.spectrogram.selection_tracks_segment
        )
        consecutive = bool(rows) and rows == list(range(rows[0], rows[-1] + 1))
        unlocked = bool(rows) and not any(
            self.segment_model.segments[row].locked for row in rows
        )
        menu = QMenu(self.segment_table)

        def add_action(label: str, callback, enabled: bool = True) -> QAction:
            action = menu.addAction(label)
            action.setEnabled(enabled)
            action.triggered.connect(callback)
            return action

        if multi:
            heading = menu.addAction(f"已选择 {count} 个句段")
            heading.setEnabled(False)
            if has_time_selection:
                menu.addSeparator()
                add_action(
                    "将音频选区平均分配给所选句",
                    lambda: self.distribute_selection(False),
                    consecutive and unlocked,
                )
                add_action(
                    "按原时长比例分配音频选区",
                    lambda: self.distribute_selection(True),
                    consecutive and unlocked,
                )
                add_action(
                    "将首句开始设为选区开始",
                    lambda: self.set_boundary_from_selection("start"),
                    not self.segment_model.segments[rows[0]].locked,
                )
                add_action(
                    "将末句结束设为选区结束",
                    lambda: self.set_boundary_from_selection("end"),
                    not self.segment_model.segments[rows[-1]].locked,
                )
                add_action(
                    f"Qwen 强制对齐所选 {count} 句 ↔ 音频选区",
                    self.qwen_align_selected_range,
                    consecutive and unlocked,
                )
            menu.addSeparator()
            add_action("清除所选句时间对应", self.clear_timing)
            add_action(
                f"合并所选 {count} 句",
                self.merge_selected_segments,
                consecutive and unlocked,
            )
            add_action("锁定/解锁所选句", self.toggle_lock)
            add_action("删除所选句时间匹配", self.delete_segments)
        else:
            add_action("播放当前句", self.play_current_segment, has_row)
            add_action("编辑文本", self.focus_segment_editor, has_row)
            menu.addSeparator()
            add_action("在当前句前插入", lambda: self.insert_segment(-1), has_row)
            add_action("在当前句后插入", lambda: self.insert_segment(1), has_row)
            if has_time_selection:
                menu.addSeparator()
                add_action("将选区绑定到当前句", self.bind_selection, has_row)
                add_action("将当前句开始设为选区开始", lambda: self.set_boundary_from_selection("start"), has_row)
                add_action("将当前句结束设为选区结束", lambda: self.set_boundary_from_selection("end"), has_row)
                add_action("Qwen 强制对齐当前句 ↔ 音频选区", self.qwen_align_selected_range, unlocked)
            menu.addSeparator()
            add_action("按播放头拆分", self.split_segment, has_row)
            add_action("按标点拆成多句", self.split_segment_by_punctuation, has_row)
            add_action("与前一句合并", lambda: self.merge_segment(-1), has_row)
            add_action("与后一句合并", lambda: self.merge_segment(1), has_row)
            menu.addSeparator()
            add_action("清除时间对应", self.clear_timing, has_row)
            add_action("锁定/解锁", self.toggle_lock, has_row)
            add_action("恢复原始段落", self.restore_source_fragment, has_row)
            add_action("删除句段", self.delete_segments, has_row)
            menu.addSeparator()
            add_action(
                "Qwen 块级对齐（从当前句/时间向后）",
                self.qwen_align_from_current_anchor,
                has_row and unlocked,
            )
        menu.addSeparator()
        add_action("恢复整章原文", self.restore_source_chapter, bool(self.current_chapter_id))
        add_action(
            "清除本章全部时间对应（保留文本）",
            self.clear_chapter_timing,
            bool(self.current_chapter_id),
        )
        add_action(
            "重置本章字幕并重新匹配…",
            self.reset_chapter_and_rematch,
            bool(self.current_chapter_id and self.current_parts),
        )
        if execute:
            menu.exec(self.segment_table.viewport().mapToGlobal(position))
        return menu

    def insert_segment(self, relative: int = 1) -> None:
        if not self.session or self.current_chapter_id is None:
            return
        rows = self._selected_rows()
        row = rows[0] if rows else len(self.segment_model.segments) - 1
        insert_at = max(0, min(len(self.segment_model.segments), row + (1 if relative > 0 else 0)))
        start = self._local_position()
        chapter_end = self.current_parts[-1][4] if self.current_parts else start + 2000
        following = self.segment_model.segments[insert_at] if insert_at < len(self.segment_model.segments) else None
        end = min(chapter_end, start + 2000)
        if following and following.end_ms > following.start_ms and following.start_ms > start:
            end = min(end, following.start_ms)
        if end <= start:
            end = min(chapter_end, start + 500)
        self._push_history()
        segment = TextSegment(
            None, self.current_chapter_id, 0, "", start, end, 1.0,
            SegmentStatus.MANUAL, origin=SegmentOrigin.USER,
        )
        values = list(self.segment_model.segments)
        values.insert(insert_at, segment)
        self._replace_segments(values)
        self._select_segment_row(insert_at)
        self.focus_segment_editor()

    def _build_status_bar(self) -> None:
        bar = QStatusBar(self)
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)
        self.status_stage = QLabel()
        self.status_stage.setFixedWidth(300)
        self.status_stage.setToolTip("当前任务阶段")
        self.status_progress = QProgressBar()
        self.status_progress.setFixedWidth(180)
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_chapters = QLabel("章节 0/0")
        self.status_chapters.setFixedWidth(92)
        self.status_time = QLabel("耗时 00:00 · 剩余 --:--")
        self.status_time.setFixedWidth(158)
        self.queue_label = QLabel("队列 0")
        self.queue_label.setFixedWidth(120)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setFixedWidth(48)
        self.pause_button.clicked.connect(self.pause_tasks)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setFixedWidth(48)
        self.cancel_button.clicked.connect(lambda: self.tasks.cancel_lane(TaskLane.INFERENCE))
        self.media_status = QLabel("媒体空闲")
        self.media_status.setFixedWidth(135)
        self.media_progress = QProgressBar()
        self.media_progress.setFixedWidth(90)
        self.media_progress.setRange(0, 100)
        self.media_progress.setValue(0)
        self.media_cancel = QToolButton()
        self.media_cancel.setFixedWidth(26)
        self.media_cancel.setText("×")
        self.media_cancel.setToolTip("取消波形/频谱与媒体缓存任务")
        self.media_cancel.clicked.connect(lambda: self.tasks.cancel_lane(TaskLane.MEDIA))
        bar.addWidget(self.status_stage, 1)
        for widget in (self.status_progress, self.status_chapters, self.status_time, self.queue_label,
                       self.pause_button, self.cancel_button, self.media_status,
                       self.media_progress, self.media_cancel):
            bar.addPermanentWidget(widget)

    def _ensure_project(self, title: str, *, force_new: bool = False) -> bool:
        if self.session and not force_new:
            return True
        try:
            self._set_session(ProjectSession.create(title))
            self.status_stage.setText(f"项目已创建：{self.session.root}")
            return True
        except ProjectExistsError:
            existing = self.paths.project_dir(title)
            box = QMessageBox(self)
            box.setWindowTitle("项目已存在")
            box.setText(f"{existing}\n\n同名项目不会被覆盖。")
            open_button = box.addButton("打开已有项目", QMessageBox.ButtonRole.AcceptRole)
            rename_button = box.addButton("修改名称", QMessageBox.ButtonRole.ActionRole)
            box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is open_button:
                self._open_project(str(existing))
                return self.session is not None
            if box.clickedButton() is rename_button:
                replacement, ok = QInputDialog.getText(self, "修改项目名称", "新名称：", text=title)
                if ok and replacement.strip():
                    return self._ensure_project(replacement.strip(), force_new=force_new)
            return False
        except Exception as exc:
            self._show_error(str(exc))
            return False

    def _set_session(self, session: ProjectSession) -> None:
        if self.session and self.session is not session and hasattr(self, "editor_commit_timer"):
            self.editor_commit_timer.stop()
            self._commit_segment_editor()
        self._session_generation += 1
        self.tasks.set_session_generation(self._session_generation)
        if self.session and self.session is not session:
            self._playback_requested = False
            self.player.stop()
            self._pending_media_position = None
            self._pending_media_autoplay = False
            self._pending_media_deadline = 0.0
            self._player_asset_id = None
            self._logical_playhead_ms = 0
            self._proxy_attempted.clear()
            self.player.setSource(QUrl())
            if hasattr(self, "original_book_view"):
                self.original_book_view.clear_book(self._session_generation)
            self._release_current_cache()
            self.session.close()
        self.session = session
        self._history.clear()
        self._future.clear()
        self._find_state = None
        if self._find_dialog is not None:
            self._find_dialog.set_status("")
        self._media_revision = 0
        self.current_chapter_id = None
        self._logical_playhead_ms = 0
        self._clear_play_range()
        self._sync_settings_from_manifest()
        self._reload_chapters()
        self._update_title()
        self._update_undo_actions()

    def _release_current_cache(self) -> None:
        if self.current_cache:
            self.mini_delegate.set_cache(None)
            self.spectrogram.set_cache(None)
            self.overview.set_cache(None)
            self.current_cache.close()
        self.current_cache = None

    def _project_context(self) -> ProjectContextKey | None:
        if not self.session:
            return None
        return ProjectContextKey(
            str(self.session.repository.database.resolve()),
            self.session.manifest.project_id,
            self._session_generation,
        )

    def _update_title(self) -> None:
        if not self.session:
            self.setWindowTitle("AudioAlignTool")
            self.project_label.setText("尚未打开项目")
            return
        marker = " *" if self.session.dirty else ""
        self.setWindowTitle(f"{self.session.manifest.title}{marker} — AudioAlignTool")
        self.project_label.setText(f"{self.session.manifest.title}\n{self.session.archive_path or self.session.root}")

    def _reload_chapters(self, select: int = 0) -> None:
        self.chapter_list.blockSignals(True)
        self.chapter_list.clear()
        chapters = self.session.repository.chapters() if self.session else []
        for chapter in chapters:
            chapter_id = chapter.id or 0
            links = self.session.repository.chapter_links(chapter_id)
            prefix = "⚠ " if self.session.repository.chapter_media_needs_review(chapter_id) else ("♫ " if links else "□ ")
            self.chapter_list.addItem(prefix + chapter.title)
        self.chapter_list.blockSignals(False)
        self.status_chapters.setText(f"章节 {min(select + 1, len(chapters)) if chapters else 0}/{len(chapters)}")
        if chapters:
            self.chapter_list.setCurrentRow(max(0, min(select, len(chapters) - 1)))
        else:
            self.segment_model.set_segments([])
            self.article_view.set_content([], [], None)
            self.original_book_view.clear_book(self._session_generation)

    def _chapter_selected(self, row: int) -> None:
        if not self.session or row < 0:
            return
        if self.current_chapter_id is not None and self._editing_row >= 0:
            self._commit_segment_editor()
        chapters = self.session.repository.chapters()
        if row >= len(chapters):
            return
        chapter = chapters[row]
        self._clear_play_range()
        self.current_chapter_id = chapter.id
        self.status_chapters.setText(f"章节 {row + 1}/{len(chapters)}")
        segments = self.session.repository.segments(chapter.id or 0)
        self.segment_model.set_segments(segments)
        current_view = self.content_tabs.currentWidget()
        anchors = self.session.repository.anchors(chapter.id or 0)
        self._article_content_dirty = current_view is not self.article_view
        self._asr_content_dirty = current_view is not self.asr_comparison
        self._original_book_content_dirty = current_view is not self.original_book_view
        if current_view is self.article_view:
            self.article_view.set_content(segments, anchors, None)
        elif current_view is self.asr_comparison:
            self.asr_comparison.set_content(
                segments,
                self.session.repository.asr_tokens(chapter.id or 0),
                anchors,
            )
        elif current_view is self.original_book_view:
            self.original_book_view.set_chapter(
                self.session, chapter.id or 0, segments, self._session_generation,
            )
        self._editing_row = -1
        if not segments:
            self._load_segment_editor(-1)
        self.spectrogram.set_segments(segments)
        self.overview.set_segments(segments)
        self.spectrogram.set_silences([])
        self.overview.set_silences([])
        self.silence_candidates = []
        self._release_current_cache()
        self.current_asset = None
        self.current_link = None
        self.current_parts = []
        self._logical_playhead_ms = 0
        links = self.session.repository.chapter_links(chapter.id or 0)
        if not links:
            self._playback_requested = False
            self._pending_media_position = None
            self._pending_media_autoplay = False
            self._pending_media_deadline = 0.0
            self._player_asset_id = None
            self.player.setSource(QUrl())
            self.spectrogram.set_cache(None)
            self.overview.set_cache(None)
            if segments:
                self._select_segment_row(0)
            self._update_undo_actions()
            return
        local_cursor = 0
        slices: list[tuple[Path, int, int]] = []
        for link in links:
            asset = self.session.repository.audio(link.audio_id)
            path = self.session.resolve_audio(asset)
            if not asset or not path:
                self._show_error("找不到已配对的音频，请重新定位或调整配对。")
                return
            part_duration = max(0, link.source_end_ms - link.source_start_ms)
            self.current_parts.append((link, asset, path, local_cursor, local_cursor + part_duration))
            slices.append((path, link.source_start_ms, link.source_end_ms))
            local_cursor += part_duration
        first_link, first_asset, first_path, _start, _end = self.current_parts[0]
        self.current_asset, self.current_link = first_asset, first_link
        self._pending_seek_target = 0
        self._pending_seek_deadline = time.monotonic() + 3.0
        self._request_media_position(
            first_asset, first_path, first_link.source_start_ms, autoplay=False,
        )
        cache_token = hashlib.sha1(
            "|".join(f"{link.audio_id}:{link.source_start_ms}:{link.source_end_ms}" for link in links).encode("ascii")
        ).hexdigest()[:16]
        duration = max(1, local_cursor)
        # Establish the real chapter timeline immediately.  The visualization
        # cache may still be building on the media lane, but the time axis must
        # not remain in the empty one-millisecond placeholder state meanwhile.
        self.spectrogram.set_cache(None, duration)
        self.overview.set_cache(None, duration)
        cache_dir = self.session.root / "cache" / f"visualization-chapter-{chapter.id}-{cache_token}"
        compatibility_cache = self.session.root / "cache" / f"spectrogram-chapter-{chapter.id}-{cache_token}"
        if is_visualization_cache(compatibility_cache):
            cache_dir = compatibility_cache
        self.silence_candidates = self.session.repository.silence_candidates(
            chapter.id or 0, self._silence_signature()
        )
        self._refresh_silence_overlays()
        if is_visualization_cache(cache_dir):
            self._load_visualization(cache_dir, duration)
        else:
            def job(progress):
                if is_visualization_cache(cache_dir):
                    return cache_dir
                cache_dir.parent.mkdir(parents=True, exist_ok=True)
                building = Path(tempfile.mkdtemp(prefix=cache_dir.name + "-", suffix=".building", dir=cache_dir.parent))
                try:
                    build_audio_visualization_cache_from_slices(
                        slices,
                        building,
                        progress=lambda value: progress(value, "解码 / 波形峰值 / FFT / 写入可视化缓存"),
                    )
                    # Another process may have completed the same immutable cache first.
                    if is_visualization_cache(cache_dir):
                        shutil.rmtree(building)
                        return cache_dir
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    os.replace(building, cache_dir)
                    return cache_dir
                except Exception:
                    if building.exists():
                        shutil.rmtree(building, ignore_errors=True)
                    raise

            chapter_id = chapter.id

            def cache_ready(result) -> None:
                if self.current_chapter_id == chapter_id:
                    self._load_visualization(Path(result), duration)

            self.tasks.submit(
                "生成波形/频谱缓存", job, cache_ready,
                lane=TaskLane.MEDIA, priority=100,
                session_generation=self._session_generation,
            )
        if segments:
            self._select_segment_row(0)
        self._update_undo_actions()

    def _load_visualization(self, cache_dir: Path, duration: int) -> None:
        if not cache_dir.exists():
            return
        self.current_cache = AudioVisualizationCache(cache_dir)
        self.mini_delegate.set_cache(self.current_cache)
        self.spectrogram.set_cache(self.current_cache, duration)
        self.spectrogram.set_segments(self.segment_model.segments)
        self.overview.set_segments(self.segment_model.segments)
        self.overview.set_cache(self.current_cache, duration)
        self.overview.set_mode(self.spectrogram.mode)
        if self.content_tabs.currentWidget() is self.article_view:
            anchors = self.session.repository.anchors(self.current_chapter_id or 0) if self.session else []
            self.article_view.set_content(self.segment_model.segments, anchors, self.current_cache)
            self._article_content_dirty = False
        else:
            self._article_content_dirty = True

    def _load_spectrogram(self, cache_dir: Path, duration: int) -> None:
        """Compatibility entry point used by older tests and integrations."""
        self._load_visualization(cache_dir, duration)

    def new_project(self) -> None:
        title, ok = QInputDialog.getText(self, "新建项目", "项目名称：")
        if ok and title.strip():
            self._commit_segment_editor()
            if self.session and self.session.dirty:
                if self.session.archive_path:
                    choice = QMessageBox.question(
                        self,
                        "保存当前项目",
                        "当前压缩项目有尚未写回的修改，是否先保存？",
                        QMessageBox.StandardButton.Save
                        | QMessageBox.StandardButton.Discard
                        | QMessageBox.StandardButton.Cancel,
                    )
                    if choice == QMessageBox.StandardButton.Cancel:
                        return
                    if choice == QMessageBox.StandardButton.Save:
                        try:
                            self.session.save()
                        except Exception as exc:
                            self._show_error(str(exc))
                            return
                else:
                    try:
                        self.session.save()
                    except Exception as exc:
                        self._show_error(str(exc))
                        return
            self._ensure_project(title.strip(), force_new=True)

    def open_archive(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "AudioAlignTool 项目 (*.aatproj)")
        if path:
            self._open_project(path)

    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "打开项目文件夹")
        if path:
            self._open_project(path)

    def open_library(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "程序项目库", str(self.paths.projects))
        if path:
            self._open_project(path)

    def _open_project(self, path: str) -> None:
        self._commit_segment_editor()
        if self.session and self.session.dirty:
            if self.session.archive_path:
                choice = QMessageBox.question(
                    self,
                    "保存当前项目",
                    "当前压缩项目有尚未写回的修改，是否先保存？",
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel,
                )
                if choice == QMessageBox.StandardButton.Cancel:
                    return
                if choice == QMessageBox.StandardButton.Save:
                    try:
                        self.session.save()
                    except Exception as exc:
                        self._show_error(str(exc))
                        return
            else:
                try:
                    self.session.save()
                except Exception as exc:
                    self._show_error(str(exc))
                    return
        try:
            self._set_session(ProjectSession.open(path))
        except Exception as exc:
            self._show_error(str(exc))

    def save_project(self) -> None:
        if not self.session:
            return
        try:
            self._settings_changed()
            self.session.save()
            self._update_title()
            self.status_stage.setText("项目已保存")
        except Exception as exc:
            self._show_error(str(exc))

    def save_as_archive(self) -> None:
        if not self.session:
            return
        path, _ = QFileDialog.getSaveFileName(self, "另存为 .aatproj", self.session.manifest.title + ".aatproj", "AudioAlignTool 项目 (*.aatproj)")
        if not path:
            return
        include_audio = QMessageBox.question(self, "包含音频", "是否把原始音频复制到项目？") == QMessageBox.StandardButton.Yes
        try:
            target = self.session.save_as_archive(path, include_audio=include_audio, include_cache=True)
            self._set_session(ProjectSession.open(target))
        except Exception as exc:
            self._show_error(str(exc))

    def save_as_folder(self) -> None:
        if not self.session:
            return
        parent = QFileDialog.getExistingDirectory(self, "选择项目文件夹的上级目录")
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "项目文件夹", "文件夹名称：", text=self.session.manifest.title)
        if ok and name.strip():
            try:
                target = self.session.save_as_folder(Path(parent) / sanitize_project_name(name), include_cache=True)
                self._set_session(ProjectSession.open(target))
            except Exception as exc:
                self._show_error(str(exc))

    def import_text(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入文本或电子书（可多选）", "",
            "书籍与文档 (*.txt *.md *.markdown *.html *.htm *.epub)",
        )
        if not paths or not self._ensure_project(Path(paths[0]).stem):
            return
        try:
            prepared: list[tuple[Path, list, list[Path], list[str]]] = []
            one_per_file = len(paths) > 1
            for path in paths:
                source = Path(path).resolve()
                imported = import_book(
                    source,
                    one_chapter=one_per_file and source.suffix.lower() != ".epub",
                    anki_furigana=self.session.manifest.anki_furigana,
                )
                if source.suffix.lower() in {".html", ".htm"}:
                    resources, warnings = collect_local_html_resources(source)
                elif source.suffix.lower() in {".md", ".markdown"}:
                    resources, warnings = collect_local_markdown_resources(source)
                else:
                    resources, warnings = [], []
                prepared.append((source, imported, resources, warnings))

            base = len(self.session.repository.chapters())
            next_position = base
            all_warnings: list[str] = []
            for source, imported, resources, warnings in prepared:
                token = fingerprint_file(source)[:10]
                document_root = self.session.root / "source" / f"{sanitize_project_name(source.stem)}-{token}"
                document_root.mkdir(parents=True, exist_ok=True)
                destination = document_root / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
                for resource in resources:
                    relative = resource.relative_to(source.parent)
                    target = document_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(resource, target)
                kind = imported[0].source_kind if imported else SourceDocumentKind.TXT
                document_id = self.session.repository.add_source_document(SourceDocument(
                    None, kind, str(source), str(destination.relative_to(self.session.root)),
                    fingerprint_file(source), source.name,
                    str(document_root.relative_to(self.session.root)),
                ))
                if not self.session.manifest.source_name:
                    self.session.manifest.source_name = str(destination.relative_to(self.session.root))
                for item in imported:
                    chapter_id = self.session.repository.add_chapter(
                        Chapter(None, item.title, next_position, item.source_html)
                    )
                    next_position += 1
                    self.session.repository.set_chapter_source_document(
                        chapter_id, document_id, entry_path=item.entry_path, selector=item.selector,
                    )
                    if item.source_parts:
                        self.session.repository.set_chapter_source_parts(
                            chapter_id, document_id,
                            [(part.entry_path, part.selector) for part in item.source_parts],
                        )
                    fragments = [
                        SourceFragment(
                            None, chapter_id, fragment.position, fragment.kind, fragment.text,
                            fragment.source_start_char, fragment.source_end_char,
                        )
                        for fragment in item.fragments
                    ]
                    self.session.repository.replace_source_fragments(chapter_id, fragments)
                    stored_fragments = self.session.repository.source_fragments(chapter_id)
                    for fragment in stored_fragments:
                        self.session.repository.set_fragment_locator(fragment.id or 0, f"block:{fragment.position}")
                    segments: list[TextSegment] = []
                    for fragment in stored_fragments:
                        for text, start, end in split_sentences_with_offsets(fragment.text):
                            segments.append(TextSegment(
                                None, chapter_id, len(segments), text,
                                origin=SegmentOrigin.SOURCE,
                                source_fragment_id=fragment.id,
                                source_start_char=fragment.source_start_char + start,
                                source_end_char=fragment.source_start_char + end,
                            ))
                    self.session.repository.replace_segments(chapter_id, segments)
                all_warnings.extend(f"{source.name}：{warning}" for warning in warnings)
            self._mark_dirty()
            self._reload_chapters(base)
            if all_warnings:
                QMessageBox.warning(self, "部分资源未导入", "\n".join(all_warnings[:20]))
        except Exception as exc:
            self._show_error(str(exc))

    def import_srt(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入 SRT（可多选；多选时每个文件建立一章）", "", "SubRip 字幕 (*.srt)"
        )
        if not paths or not self._ensure_project(Path(paths[0]).stem):
            return
        try:
            imported_files = [(Path(path).resolve(), import_srt(path)) for path in paths]
        except Exception as exc:
            self._show_error(str(exc))
            return
        path, cues = imported_files[0]
        modes = ["新增为章节", "匹配到当前章节", "替换当前章节"]
        if len(imported_files) > 1:
            mode, ok = "新增为章节", True
        else:
            mode, ok = QInputDialog.getItem(self, "SRT 导入方式", "处理方式：", modes, 0, False)
            if not ok:
                return
        if mode != "新增为章节" and self.current_chapter_id is not None:
            message = (
                f"将用 {len(cues)} 条 SRT 字幕替换当前章节句段。原始来源片段仍会保留，是否继续？"
                if mode == "替换当前章节"
                else f"将预览匹配 {len(cues)} 条 SRT 字幕到当前章节并写入匹配到的时间，是否继续？"
            )
            if QMessageBox.question(
                self, mode, message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
        source = Path(path).resolve()
        if len(imported_files) > 1:
            base = len(self.session.repository.chapters())
            for source, cues in imported_files:
                self._import_srt_as_chapter(source, cues)
            self._mark_dirty()
            self._reload_chapters(base)
            return
        token = fingerprint_file(source)[:10]
        document_root = self.session.root / "source" / f"{sanitize_project_name(source.stem)}-{token}"
        document_root.mkdir(parents=True, exist_ok=True)
        destination = document_root / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        document_id = self.session.repository.add_source_document(SourceDocument(
            None, SourceDocumentKind.SRT, str(source), str(destination.relative_to(self.session.root)),
            fingerprint_file(source), source.name, str(document_root.relative_to(self.session.root)),
        ))
        if mode == "新增为章节" or self.current_chapter_id is None:
            chapter_position = len(self.session.repository.chapters())
            chapter_id = self.session.repository.add_chapter(Chapter(None, source.stem, chapter_position))
            self.session.repository.set_chapter_source_document(chapter_id, document_id, entry_path=source.name)
            cursor = 0
            fragments: list[SourceFragment] = []
            segments: list[TextSegment] = []
            for position, cue in enumerate(cues):
                end = cursor + len(cue.text)
                fragments.append(SourceFragment(None, chapter_id, position, "subtitle", cue.text, cursor, end))
                cursor = end
            self.session.repository.replace_source_fragments(chapter_id, fragments)
            stored = self.session.repository.source_fragments(chapter_id)
            for position, (cue, fragment) in enumerate(zip(cues, stored)):
                self.session.repository.set_fragment_locator(fragment.id or 0, f"cue:{cue.index}")
                segments.append(TextSegment(
                    None, chapter_id, position, cue.text, cue.start_ms, cue.end_ms, 1.0,
                    SegmentStatus.MANUAL, False, SegmentOrigin.SUBTITLE, fragment.id,
                    fragment.source_start_char, fragment.source_end_char,
                ))
            self.session.repository.replace_segments(chapter_id, segments)
            self._mark_dirty()
            self._reload_chapters(chapter_position)
            return

        chapter_id = self.current_chapter_id
        current = copy.deepcopy(self.segment_model.segments)
        self._push_history()
        if mode == "替换当前章节":
            replacement = [TextSegment(
                None, chapter_id, position, cue.text, cue.start_ms, cue.end_ms, 1.0,
                SegmentStatus.MANUAL, False, SegmentOrigin.SUBTITLE,
            ) for position, cue in enumerate(cues)]
            self._replace_segments(replacement, [])
        else:
            cue_cursor = 0
            anchors = copy.deepcopy(self.session.repository.anchors(chapter_id))
            for segment in current:
                best_index, best_score = -1, 0.0
                target = normalize_for_match(segment.text)
                for index in range(cue_cursor, min(len(cues), cue_cursor + 8)):
                    score = SequenceMatcher(None, target, normalize_for_match(cues[index].text)).ratio()
                    if score > best_score:
                        best_index, best_score = index, score
                if best_index < 0 or best_score < 0.35:
                    continue
                cue = cues[best_index]
                cue_cursor = best_index + 1
                segment.start_ms, segment.end_ms = cue.start_ms, cue.end_ms
                segment.confidence = best_score
                segment.status = SegmentStatus.MANUAL if best_score >= 0.7 else SegmentStatus.LOW_CONFIDENCE
                if segment.source_start_char is not None and segment.source_end_char is not None:
                    anchors.append(TextAudioAnchor(
                        None, chapter_id, segment.id, segment.source_start_char,
                        segment.source_end_char, cue.start_ms, cue.end_ms, best_score, "srt-match",
                    ))
            self._replace_segments(current, anchors)
        self._mark_dirty()

    def _import_srt_as_chapter(self, source: Path, cues) -> int:
        """Import one already-validated SRT as one immutable source chapter."""
        token = fingerprint_file(source)[:10]
        document_root = self.session.root / "source" / f"{sanitize_project_name(source.stem)}-{token}"
        document_root.mkdir(parents=True, exist_ok=True)
        destination = document_root / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        document_id = self.session.repository.add_source_document(SourceDocument(
            None, SourceDocumentKind.SRT, str(source), str(destination.relative_to(self.session.root)),
            fingerprint_file(source), source.name, str(document_root.relative_to(self.session.root)),
        ))
        chapter_position = len(self.session.repository.chapters())
        chapter_id = self.session.repository.add_chapter(Chapter(None, source.stem, chapter_position))
        self.session.repository.set_chapter_source_document(chapter_id, document_id, entry_path=source.name)
        cursor = 0
        fragments: list[SourceFragment] = []
        for position, cue in enumerate(cues):
            end = cursor + len(cue.text)
            fragments.append(SourceFragment(None, chapter_id, position, "subtitle", cue.text, cursor, end))
            cursor = end
        self.session.repository.replace_source_fragments(chapter_id, fragments)
        stored = self.session.repository.source_fragments(chapter_id)
        segments = []
        for position, (cue, fragment) in enumerate(zip(cues, stored)):
            self.session.repository.set_fragment_locator(fragment.id or 0, f"cue:{cue.index}")
            segments.append(TextSegment(
                None, chapter_id, position, cue.text, cue.start_ms, cue.end_ms, 1.0,
                SegmentStatus.MANUAL, False, SegmentOrigin.SUBTITLE, fragment.id,
                fragment.source_start_char, fragment.source_end_char,
            ))
        self.session.repository.replace_segments(chapter_id, segments)
        return chapter_id

    def import_audio(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "导入音频或视频", "", self.AUDIO_FILTER)
        if not files or not self._ensure_project(Path(files[0]).stem):
            return
        dialog = ChapterAudioMappingDialog(self.session, self)
        try:
            dialog.add_files(files=files)
        except Exception as exc:
            self._show_error(str(exc))
            return
        if dialog.exec():
            self._media_graph_changed()
            self._reload_chapters(max(0, self.chapter_list.currentRow()))
            self._mark_dirty()
            if dialog.created_audio_only_chapters:
                self.status_stage.setText(
                    f"已创建 {dialog.created_audio_only_chapters} 个纯音频章节 · "
                    "运行“识别并对齐当前章节”或“全书批量处理”可生成文本"
                )

    def edit_mappings(self) -> None:
        if not self.session:
            return
        current = max(0, self.chapter_list.currentRow())
        dialog = ChapterAudioMappingDialog(self.session, self)
        if dialog.exec():
            self._media_graph_changed()
            self._reload_chapters(current)
            self._mark_dirty()

    def _media_graph_changed(self) -> None:
        """Invalidate every callback and edit snapshot based on the old media graph."""
        self._media_revision += 1
        self._session_generation += 1
        self.tasks.set_session_generation(self._session_generation)
        self._history.clear()
        self._future.clear()
        self._clear_play_range()
        self._update_undo_actions()

    def _asr_options(self) -> ASROptions:
        link = self.current_link
        backend, mode = self._workflow_components()
        language = self._selected_language_code()
        return ASROptions(
            backend=backend,
            model=self.model_combo.currentText().strip() or "small",
            language=None if language == "auto" else language,
            mode=mode,
            vad_threshold=self.vad_spin.value(),
            min_silence_ms=self.min_silence_spin.value(),
            model_root=str(self.paths.models),
            clip_start_ms=link.source_start_ms if link else 0,
            clip_end_ms=link.source_end_ms if link else None,
        )

    def _chapter_audio_parts(
        self, chapter_id: int,
    ) -> tuple[list[tuple[ChapterAudioLink, AudioAsset, Path, int, int]], str]:
        """Resolve every ordered audio slice for a chapter without changing the UI."""
        if not self.session:
            return [], "项目尚未打开"
        parts: list[tuple[ChapterAudioLink, AudioAsset, Path, int, int]] = []
        local_cursor = 0
        links = self.session.repository.chapter_links(chapter_id)
        if not links:
            return [], "没有音频配对"
        for link in links:
            asset = self.session.repository.audio(link.audio_id)
            path = self.session.resolve_audio(asset)
            if not asset or not path:
                return [], "找不到已配对的音频文件"
            duration = max(0, link.source_end_ms - link.source_start_ms)
            if duration <= 0:
                return [], "音频切片时长无效"
            parts.append((link, asset, path, local_cursor, local_cursor + duration))
            local_cursor += duration
        return parts, ""

    @staticmethod
    def _parts_audio_signature(
        parts: list[tuple[ChapterAudioLink, AudioAsset, Path, int, int]],
    ) -> str:
        values: list[str] = []
        for link, asset, path, local_start, local_end in parts:
            fingerprint = asset.fingerprint
            if not fingerprint:
                try:
                    stat = path.stat()
                    fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
                except OSError:
                    fingerprint = str(path)
            values.append(
                f"{asset.id}:{fingerprint}:{link.source_start_ms}:{link.source_end_ms}:{local_start}:{local_end}"
            )
        return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()

    def _recognition_audio_signature(self) -> str:
        return self._parts_audio_signature(self.current_parts)

    @staticmethod
    def _chapter_cache_key(repository, cache_key: str, chapter_id: int) -> str:
        """Keep identical media slices reusable without cross-chapter cache aliasing."""
        existing = repository.recognition_run(cache_key)
        if existing is None or existing.chapter_id == chapter_id:
            return cache_key
        return hashlib.sha256(f"{cache_key}|chapter={chapter_id}".encode("utf-8")).hexdigest()

    def _confirm_qwen_cpu_run(self, operation: str, duration_ms: int) -> bool:
        """Allow Qwen on CPU, but make the potentially very long wait explicit."""
        status = runtime_status("Qwen3-ASR-0.6B", self.paths.models, ASRBackendId.QWEN3_ASR)
        if status.cuda_available:
            self.model_status_label.setText(status.message)
            return True
        reason = "PyTorch 没有检测到可用的 CUDA 设备"
        if qwen_cuda_disabled_reason():
            reason = f"本次运行 CUDA 已发生致命错误并熔断：{qwen_cuda_disabled_reason()}"
        try:
            import torch  # type: ignore

            if torch.version.cuda is None and not qwen_cuda_disabled_reason():
                reason = f"当前安装的是 CPU 版 PyTorch（{torch.__version__}）"
        except ImportError:
            reason = "PyTorch 运行库缺失"
        self.model_status_label.setText(f"Qwen · CPU · float32 · {reason}")
        self.status_stage.setText(f"{operation}将使用 CPU · 速度可能很慢")
        seconds = max(0, duration_ms) / 1000
        duration = f"{seconds / 60:.1f} 分钟" if seconds >= 60 else f"{seconds:.1f} 秒"
        answer = QMessageBox.warning(
            self,
            "Qwen 将使用 CPU",
            f"{operation}没有可用的 GPU，将改用 CPU。\n\n"
            f"原因：{reason}\n处理范围：{duration}\n\n"
            "CPU 推理可能非常慢，长范围可能需要很长时间并占用大量内存。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def recognize_current(self, force: bool = False) -> None:
        if not self.session or self.current_chapter_id is None or not self.current_parts:
            self._show_error("当前章节没有可识别的音频配对。")
            return
        if self.mode_combo.currentData() == WORKFLOW_QWEN_FORCED:
            self.qwen_align_chapter()
            return
        options = self._asr_options()
        status = runtime_status(options.model, self.paths.models, options.backend)
        if not status.runtime_available:
            self._show_error(status.message)
            return
        if not status.model_available:
            answer = QMessageBox.question(
                self,
                "下载识别模型",
                f"模型 {options.model} 尚未下载。是否现在下载到\n{self.paths.models}？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if (options.backend == ASRBackendId.FASTER_WHISPER
                and options.mode == AlignmentMode.PRECISE and not status.whisperx_available):
            self._show_error("当前工作流需要 WhisperX；faster-whisper 与 Qwen3-ASR 的识别后对齐不受影响。")
            return
        if options.backend == ASRBackendId.QWEN3_ASR and not status.cuda_available:
            duration = sum(max(0, item[4] - item[3]) for item in self.current_parts)
            if not self._confirm_qwen_cpu_run("Qwen3-ASR 识别", duration):
                return
        chapter_id = self.current_chapter_id
        parts = list(self.current_parts)
        audio_signature = self._recognition_audio_signature()
        cache_key, parameters_json = recognition_cache_key(audio_signature, options)
        cache_key = self._chapter_cache_key(self.session.repository, cache_key, chapter_id)
        if force:
            self.session.repository.reset_recognition_run(cache_key)
        run = self.session.repository.ensure_recognition_run(
            chapter_id=chapter_id,
            cache_key=cache_key,
            backend=options.backend.value,
            model=options.model,
            language=options.language or "auto",
            audio_signature=audio_signature,
            parameters_json=parameters_json,
        )
        if run.status == "complete" and not force:
            tokens = self.session.repository.recognition_tokens(run.id or 0)
            for position, token in enumerate(tokens):
                token.chapter_id = chapter_id
                token.position = position
            self.status_stage.setText(f"已使用识别缓存 · {run.backend} {run.model}")
            self._apply_recognition_tokens(chapter_id, tokens, run)
            return

        existing_chunks = self.session.repository.recognition_chunks(run.id or 0)
        preserve_existing_plan = bool(existing_chunks) and not force
        planned: list[tuple[int, object, object, Path, int]] = []
        position = 0
        for part_index, (link, asset, path, local_start, local_end) in enumerate(parts):
            candidates = [
                candidate for candidate in self.silence_candidates
                if local_start < candidate.time_ms < local_end
            ]
            for chunk in plan_recognition_chunks(
                local_start, local_end, candidates, options,
                preserve_existing_plan=preserve_existing_plan,
            ):
                planned.append((position, chunk, link, path, local_start))
                position += 1
        completed_positions = {
            chunk.position for chunk in existing_chunks
            if chunk.status == "complete"
        }
        database = self.session.repository.database

        def job(progress):
            transcriber = transcriber_for_options(options)
            total = max(1, len(planned))
            for completed_index, (chunk_position, chunk, link, path, local_start) in enumerate(planned):
                if chunk_position in completed_positions:
                    progress((completed_index + 1) / total, f"缓存 {completed_index + 1}/{total}")
                    continue
                part_options = copy.copy(options)
                part_options.clip_start_ms = link.source_start_ms + (chunk.start_ms - local_start)
                part_options.clip_end_ms = link.source_start_ms + (chunk.end_ms - local_start)
                started = time.monotonic()
                ensure_inference_memory_headroom(transcriber)
                try:
                    chunk_tokens = transcriber.transcribe(
                        path,
                        chapter_id,
                        part_options,
                        lambda value, message: progress(
                            -1.0 if value < 0 else (completed_index + value) / total,
                            f"{message} · {completed_index + 1}/{total}",
                        ),
                    )
                except Exception as exc:
                    if is_inference_out_of_memory(exc):
                        release_inference_memory(transcriber, aggressive=True)
                        raise InferenceMemoryPressureError(
                            "当前章节推理时内存不足；完整缓存块已经保留"
                        ) from exc
                    raise
                kept = []
                for token in chunk_tokens:
                    token.start_ms += chunk.start_ms
                    token.end_ms += chunk.start_ms
                    midpoint = (token.start_ms + token.end_ms) // 2
                    if chunk.core_start_ms <= midpoint <= chunk.core_end_ms:
                        token.position = len(kept)
                        kept.append(token)
                record = RecognitionChunk(
                    None, run.id or 0, chunk_position, chunk.start_ms, chunk.end_ms,
                    chunk.core_start_ms, chunk.core_end_ms, "complete",
                    "".join(token.text for token in kept),
                    round((time.monotonic() - started) * 1000), "",
                )
                write_recognition_chunk(database, record, kept)
                del record, kept, chunk_tokens
                if (completed_index + 1) % 4 == 0:
                    release_inference_memory(transcriber)
                progress((completed_index + 1) / total, f"识别并缓存 {completed_index + 1}/{total}")
            device = getattr(transcriber, "last_device_info", None)
            release_inference_memory(transcriber, aggressive=True)
            return run.id or 0, device

        def done(payload):
            if not self.session or chapter_id != self.current_chapter_id:
                return
            run_id, device = payload
            if device is not None:
                self.session.repository.complete_recognition_run(run_id, device)
            tokens = self.session.repository.recognition_tokens(run_id)
            for position, token in enumerate(tokens):
                token.chapter_id = chapter_id
                token.position = position
            refreshed = self.session.repository.recognition_run(cache_key)
            self._apply_recognition_tokens(chapter_id, tokens, refreshed)

        self.tasks.submit(
            "识别 / 原文对比 / 分块缓存", job, done,
            session_generation=self._session_generation,
        )

    @staticmethod
    def _alignment_text_hash(segments: list[TextSegment]) -> str:
        """Hash only alignment inputs; manual timing edits remain authoritative."""
        payload = [
            (
                segment.text,
                bool(segment.locked),
                segment.origin.value,
                segment.source_fragment_id,
                segment.source_start_char,
                segment.source_end_char,
            )
            for segment in segments
        ]
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def recognize_book(self, force: bool = False) -> None:
        """Recognize every paired chapter sequentially while reusing one loaded model."""
        if not self.session:
            return
        if self.mode_combo.currentData() == WORKFLOW_QWEN_FORCED:
            self._show_error(
                "全书批量处理请先选择 faster-whisper、WhisperX 或 Qwen3-ASR 的“识别后对齐”。\n\n"
                "Qwen3-ASR 已在内部使用 ForcedAligner 生成识别稿时间戳；显式 ForcedAligner 仍用于当前句、"
                "选定范围、锚点向后或单章，避免书籍前言与音频不一致时把误差扩散到全书。"
            )
            return

        options = copy.copy(self._asr_options())
        options.clip_start_ms = 0
        options.clip_end_ms = None
        status = runtime_status(options.model, self.paths.models, options.backend)
        if not status.runtime_available:
            self._show_error(status.message)
            return
        if not status.model_available:
            answer = QMessageBox.question(
                self,
                "下载识别模型",
                f"模型 {options.model} 尚未下载。是否在批处理开始时下载到\n{self.paths.models}？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if (options.backend == ASRBackendId.FASTER_WHISPER
                and options.mode == AlignmentMode.PRECISE and not status.whisperx_available):
            self._show_error("当前工作流需要 WhisperX；请安装后再执行全书批处理。")
            return

        prepared = []
        skipped: list[str] = []
        settings = self._silence_settings()
        for chapter in self.session.repository.chapters():
            chapter_id = chapter.id or 0
            parts, error = self._chapter_audio_parts(chapter_id)
            if not parts:
                skipped.append(f"{chapter.title}：{error}")
                continue
            silence_signature = self._parts_silence_signature(parts, settings)
            candidates = self.session.repository.silence_candidates(chapter_id, silence_signature)
            audio_signature = self._parts_audio_signature(parts)
            cache_key, _parameters_json = recognition_cache_key(audio_signature, options)
            cache_key = self._chapter_cache_key(self.session.repository, cache_key, chapter_id)
            existing_run = self.session.repository.recognition_run(cache_key)
            preserve_existing_plan = bool(
                existing_run
                and existing_run.id
                and self.session.repository.recognition_chunks(existing_run.id)
                and not force
            )
            prepared.append((
                chapter_id, chapter.title, parts, candidates, preserve_existing_plan,
            ))
        if not prepared:
            self._show_error("全书没有可处理的章节；请先在配对管理器中建立章节与音频关系。")
            return

        total_duration = sum(
            parts[-1][4]
            for _chapter_id, _title, parts, _candidates, _preserve in prepared
        )
        total_chunks = sum(
            len(plan_recognition_chunks(
                local_start,
                local_end,
                [candidate for candidate in candidates if local_start < candidate.time_ms < local_end],
                options,
                preserve_existing_plan=preserve_existing_plan,
            ))
            for _chapter_id, _title, parts, candidates, preserve_existing_plan in prepared
            for _link, _asset, _path, local_start, local_end in parts
        )
        work_plan = BookWorkPlan(len(prepared), total_chunks, total_duration)
        if options.backend == ASRBackendId.QWEN3_ASR and not status.cuda_available:
            if not self._confirm_qwen_cpu_run("Qwen3-ASR 全书识别", total_duration):
                return
        duration_text = f"{total_duration / 3_600_000:.1f} 小时"
        skip_text = f"\n将跳过 {len(skipped)} 个未配对或音频缺失章节。" if skipped else ""
        action_text = "强制重新识别并覆盖未锁定自动结果" if force else "优先使用已有分块缓存"
        answer = QMessageBox.question(
            self,
            "全书批量处理",
            f"将按当前方式处理 {len(prepared)} 个章节（音频约 {duration_text}）。\n"
            f"策略：{action_text}。{skip_text}\n\n"
            "锁定句段不会被覆盖；取消时保留已经完整完成的章节和识别分块。是否开始？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        database = Path(self.session.repository.database)
        alignment_language = options.language or self.session.manifest.language
        snap_asr_boundaries = _asr_uses_silence_snap(options.backend)
        boundary_policy = "vad-directional-v3" if snap_asr_boundaries else "model-timestamps-v1"
        alignment_algorithm = (
            f"monotonic-dp-v3:{alignment_language or 'auto'}:{boundary_policy}"
        )
        total_chapters = len(prepared)
        self._mark_dirty()

        def job(progress):
            repository = ProjectRepository(database)
            transcriber = transcriber_for_options(options)
            completed_count = 0
            cache_count = 0
            last_device = None
            processed_audio_ms = 0
            completed_chunks = 0
            try:
                for chapter_index, (
                    chapter_id, title, parts, candidates, preserve_existing_plan,
                ) in enumerate(prepared):
                    chapter_duration = parts[-1][4]
                    chapter_prefix = f"全书 {chapter_index + 1}/{total_chapters} · {title}"
                    progress(
                        processed_audio_ms / max(1, total_duration),
                        f"{chapter_prefix} · 准备 · 已处理 {_time(processed_audio_ms)}/{_time(total_duration)}",
                    )
                    audio_signature = self._parts_audio_signature(parts)
                    cache_key, parameters_json = recognition_cache_key(audio_signature, options)
                    cache_key = self._chapter_cache_key(repository, cache_key, chapter_id)
                    if force:
                        repository.reset_recognition_run(cache_key)
                    run = repository.ensure_recognition_run(
                        chapter_id=chapter_id,
                        cache_key=cache_key,
                        backend=options.backend.value,
                        model=options.model,
                        language=options.language or "auto",
                        audio_signature=audio_signature,
                        parameters_json=parameters_json,
                    )
                    used_cache = run.status == "complete" and not force
                    current = repository.segments(chapter_id)
                    text_hash = self._alignment_text_hash(current)
                    silence_signature = self._parts_silence_signature(parts, settings)
                    if used_cache and repository.alignment_is_current(
                        chapter_id,
                        run.id or 0,
                        text_hash,
                        alignment_algorithm,
                        silence_signature,
                    ):
                        cache_count += 1
                        cached_chunks = len(repository.recognition_chunks(run.id or 0))
                        completed_chunks += cached_chunks
                        work_plan.cache_hits += cached_chunks
                        completed_count += 1
                        processed_audio_ms += chapter_duration
                        progress(
                            processed_audio_ms / max(1, total_duration),
                            f"{chapter_prefix} · 识别与对齐缓存已命中 · "
                            f"已处理 {_time(processed_audio_ms)}/{_time(total_duration)}",
                        )
                        continue
                    if not used_cache:
                        existing_chunks = repository.recognition_chunks(run.id or 0)
                        preserve_existing_plan = (
                            preserve_existing_plan or (bool(existing_chunks) and not force)
                        )
                        planned = []
                        chunk_position = 0
                        for link, _asset, path, local_start, local_end in parts:
                            part_candidates = [
                                candidate for candidate in candidates
                                if local_start < candidate.time_ms < local_end
                            ]
                            for chunk in plan_recognition_chunks(
                                local_start, local_end, part_candidates, options,
                                preserve_existing_plan=preserve_existing_plan,
                            ):
                                planned.append((chunk_position, chunk, link, path, local_start))
                                chunk_position += 1
                        completed_positions = {
                            chunk.position for chunk in existing_chunks
                            if chunk.status == "complete"
                        }
                        chunk_total = max(1, len(planned))
                        for chunk_index, (position, chunk, link, path, local_start) in enumerate(planned):
                            if position in completed_positions:
                                completed_chunks += 1
                                work_plan.cache_hits += 1
                                local_fraction = (chunk_index + 1) / chunk_total
                                progress(
                                    (processed_audio_ms + chapter_duration * local_fraction) / max(1, total_duration),
                                    f"{chapter_prefix} · 缓存块 {chunk_index + 1}/{chunk_total} · "
                                    f"总块 {completed_chunks}/{work_plan.chunk_total} · "
                                    f"已处理 {_time(round(processed_audio_ms + chapter_duration * local_fraction))}/{_time(total_duration)}",
                                )
                                continue
                            part_options = copy.copy(options)
                            part_options.clip_start_ms = link.source_start_ms + (chunk.start_ms - local_start)
                            part_options.clip_end_ms = link.source_start_ms + (chunk.end_ms - local_start)
                            started = time.monotonic()
                            ensure_inference_memory_headroom(transcriber)
                            try:
                                chunk_tokens = transcriber.transcribe(
                                    path,
                                    chapter_id,
                                    part_options,
                                    lambda value, message, ci=chunk_index, ct=chunk_total: progress(
                                        -1.0 if value < 0 else
                                        (processed_audio_ms + chapter_duration * (ci + value) / ct) / max(1, total_duration),
                                        f"{chapter_prefix} · {message} · 块 {ci + 1}/{ct} · "
                                        f"已处理 {_time(round(processed_audio_ms + chapter_duration * ci / ct))}/{_time(total_duration)}",
                                    ),
                                )
                            except Exception as exc:
                                if is_inference_out_of_memory(exc):
                                    raise InferenceMemoryPressureError(
                                        f"{title} 第 {chunk_index + 1}/{chunk_total} 块推理时内存不足"
                                    ) from exc
                                raise
                            kept = []
                            for token in chunk_tokens:
                                token.start_ms += chunk.start_ms
                                token.end_ms += chunk.start_ms
                                midpoint = (token.start_ms + token.end_ms) // 2
                                if chunk.core_start_ms <= midpoint <= chunk.core_end_ms:
                                    token.position = len(kept)
                                    kept.append(token)
                            record = RecognitionChunk(
                                None, run.id or 0, position, chunk.start_ms, chunk.end_ms,
                                chunk.core_start_ms, chunk.core_end_ms, "complete",
                                "".join(token.text for token in kept),
                                round((time.monotonic() - started) * 1000), "",
                            )
                            write_recognition_chunk(database, record, kept)
                            del record, kept, chunk_tokens
                            if (chunk_index + 1) % 4 == 0:
                                release_inference_memory(transcriber)
                            completed_chunks += 1
                            progress(
                                (processed_audio_ms + chapter_duration * (chunk_index + 1) / chunk_total)
                                / max(1, total_duration),
                                f"{chapter_prefix} · 已完成块 {chunk_index + 1}/{chunk_total} · "
                                f"总块 {completed_chunks}/{work_plan.chunk_total}",
                            )
                        last_device = getattr(transcriber, "last_device_info", None)
                        if last_device is not None:
                            repository.complete_recognition_run(run.id or 0, last_device)
                    else:
                        cache_count += 1
                        cached_chunks = len(repository.recognition_chunks(run.id or 0))
                        completed_chunks += cached_chunks
                        work_plan.cache_hits += cached_chunks

                    tokens = repository.recognition_tokens(run.id or 0)
                    for token_position, token in enumerate(tokens):
                        token.chapter_id = chapter_id
                        token.position = token_position
                    # Projects created before alignment-result caching may already
                    # contain exactly this recognition result. Register it without
                    # repeating the expensive dynamic-programming match.
                    published_tokens = repository.asr_tokens(chapter_id) if used_cache else []
                    already_published = (
                        snap_asr_boundaries
                        and bool(current)
                        and len(published_tokens) == len(tokens)
                        and all(
                            left.text == right.text
                            and left.start_ms == right.start_ms
                            and left.end_ms == right.end_ms
                            for left, right in zip(published_tokens, tokens)
                        )
                    )
                    if already_published:
                        repository.record_alignment_run(
                            chapter_id, run.id or 0, text_hash,
                            alignment_algorithm, silence_signature,
                        )
                        completed_count += 1
                        processed_audio_ms += chapter_duration
                        progress(
                            processed_audio_ms / max(1, total_duration),
                            f"{chapter_prefix} · 已登记现有对齐缓存 · "
                            f"已处理 {_time(processed_audio_ms)}/{_time(total_duration)}",
                        )
                        del published_tokens, tokens, current
                        release_inference_memory(transcriber, aggressive=True)
                        continue
                    del published_tokens
                    aligned = (
                        segments_from_asr_tokens(
                            tokens, chapter_id=chapter_id, language=alignment_language,
                        )
                        if not current or _are_unedited_asr_segments(current)
                        else align_segments_to_tokens(current, tokens, language=alignment_language)
                    )
                    if candidates and snap_asr_boundaries:
                        aligned = snap_boundaries(
                            aligned, candidates,
                            window_ms=settings.snap_window_ms,
                            padding_ms=settings.boundary_padding_ms,
                        )
                    anchors = anchors_from_segments_tokens(aligned, tokens, language=alignment_language)
                    repository.replace_recognition_alignment(
                        chapter_id,
                        tokens,
                        aligned,
                        anchors,
                    )
                    repository.record_alignment_run(
                        chapter_id,
                        run.id or 0,
                        self._alignment_text_hash(aligned),
                        alignment_algorithm,
                        silence_signature,
                    )
                    completed_count += 1
                    processed_audio_ms += chapter_duration
                    progress(
                        processed_audio_ms / max(1, total_duration),
                        f"{chapter_prefix} · 已提交完整结果 · "
                        f"已处理 {_time(processed_audio_ms)}/{_time(total_duration)} · "
                        f"缓存章 {cache_count} · 缓存块 {work_plan.cache_hits}",
                    )
                    del anchors, aligned, current, tokens
                    release_inference_memory(transcriber, aggressive=True)
                return completed_count, cache_count, skipped, last_device, ""
            except InferenceMemoryPressureError as exc:
                release_inference_memory(transcriber, aggressive=True)
                return completed_count, cache_count, skipped, last_device, str(exc)
            finally:
                repository.close()

        def done(payload):
            if not self.session or Path(self.session.repository.database) != database:
                return
            completed_count, cache_count, skipped_chapters, device, stopped_reason = payload
            current_row = self.chapter_list.currentRow()
            if current_row >= 0:
                self._chapter_selected(current_row)
            if device is not None:
                self.model_status_label.setText(device.display_text)
            if stopped_reason:
                self.status_stage.setText(
                    f"内存不足保护：已安全停止 · 已完成 {completed_count}/{total_chapters} 章 · "
                    "完整识别块已保留，下次可从缓存继续"
                )
                QMessageBox.warning(
                    self,
                    "全书任务已安全停止",
                    f"{stopped_reason}。\n\n已完成的章节和识别块均已保存；关闭其他占用内存的程序后，"
                    "再次执行全书识别即可从缓存继续。",
                )
            else:
                self.status_stage.setText(
                    f"全书批量处理完成 · {completed_count}/{total_chapters} 章 · "
                    f"命中完整缓存 {cache_count} 章 · 跳过 {len(skipped_chapters)} 章"
                )
            self._mark_dirty()

        self.tasks.submit(
            "全书识别 / 原文对比 / 自动对齐", job, done,
            session_generation=self._session_generation,
        )

    def _apply_recognition_tokens(self, chapter_id: int, tokens, run=None) -> None:
        if not self.session or chapter_id != self.current_chapter_id:
            return
        self.session.repository.replace_asr_tokens(chapter_id, tokens)
        current = self.session.repository.segments(chapter_id)
        aligned = (
            segments_from_asr_tokens(
                tokens, chapter_id=chapter_id, language=self.session.manifest.language,
            )
            if not current or _are_unedited_asr_segments(current)
            else align_segments_to_tokens(current, tokens, language=self.session.manifest.language)
        )
        if self.silence_candidates and _asr_uses_silence_snap(
            getattr(run, "backend", None),
        ):
            settings = self._silence_settings()
            aligned = snap_boundaries(
                aligned, self.silence_candidates,
                window_ms=settings.snap_window_ms,
                padding_ms=settings.boundary_padding_ms,
            )
        self.session.repository.replace_segments(chapter_id, aligned)
        saved_segments = self.session.repository.segments(chapter_id)
        anchors = anchors_from_segments_tokens(saved_segments, tokens, language=self.session.manifest.language)
        self.session.repository.replace_anchors(chapter_id, anchors)
        self.segment_model.set_segments(self.session.repository.segments(chapter_id))
        self.spectrogram.set_segments(self.segment_model.segments)
        self.overview.set_segments(self.segment_model.segments)
        self.article_view.set_content(
            self.segment_model.segments, self.session.repository.anchors(chapter_id), self.current_cache,
        )
        self.asr_comparison.set_content(
            self.segment_model.segments, tokens,
            self.session.repository.anchors(chapter_id),
        )
        self._mark_dirty()
        self._update_model_status()
        if run:
            device = "GPU" if run.actual_device == "cuda" else "CPU"
            detail = f"{run.backend} {run.model} · {device} · {run.compute_type or '--'}"
            if run.device_name:
                detail += f" · {run.device_name}"
            if run.fallback_reason:
                detail += f" · CUDA 回退：{run.fallback_reason}"
            self.model_status_label.setText(detail)
            self.status_stage.setText(detail)

    def clear_recognition_cache(self) -> None:
        if not self.session or self.current_chapter_id is None:
            return
        options = self._asr_options()
        count = self.session.repository.delete_recognition_cache(
            self.current_chapter_id, options.backend.value, options.model,
        )
        self.status_stage.setText(f"已清除 {count} 个当前模型识别缓存")

    def qwen_align_current_segment(self) -> None:
        if not self.session or self.current_chapter_id is None:
            return
        row = self.segment_table.currentIndex().row()
        if not (0 <= row < len(self.segment_model.segments)):
            self._show_error("请先选择需要精确对齐的句子")
            return
        segment = self.segment_model.segments[row]
        if segment.locked:
            self._show_error("当前句已锁定，请先解锁")
            return
        selection = self.spectrogram.selection
        start_ms = selection.start_ms if selection and selection.end_ms > selection.start_ms else segment.start_ms
        end_ms = selection.end_ms if selection and selection.end_ms > selection.start_ms else segment.end_ms
        if end_ms <= start_ms or end_ms - start_ms > 240_000:
            self._show_error("Qwen ForcedAligner 选区必须大于 0 且不超过 240 秒")
            return
        part = next(
            (item for item in self.current_parts if item[3] <= start_ms and end_ms <= item[4]), None
        )
        if part is None:
            self._show_error("当前选区跨越多个音频资源，请缩小到单个音频切片内")
            return
        language = self._selected_language_code()
        if language == "auto":
            self._show_error("Qwen ForcedAligner 需要明确选择文本语言")
            return
        runtime = runtime_status("Qwen3-ForcedAligner-0.6B", self.paths.models, ASRBackendId.QWEN3_ASR)
        if not runtime.runtime_available:
            self._show_error(runtime.message)
            return
        if not runtime.cuda_available and not self._confirm_qwen_cpu_run(
            "Qwen 当前句强制对齐", end_ms - start_ms,
        ):
            return
        link, _asset, path, local_start, _local_end = part
        options = ASROptions(
            backend=ASRBackendId.QWEN3_ASR,
            model="Qwen3-ForcedAligner-0.6B",
            language=language,
            model_root=str(self.paths.models),
            clip_start_ms=link.source_start_ms + start_ms - local_start,
            clip_end_ms=link.source_start_ms + end_ms - local_start,
        )
        chapter_id = self.current_chapter_id

        def job(progress):
            aligner = Qwen3ForcedAligner()
            tokens = aligner.align(path, segment.text, language, chapter_id, options, progress)
            for token in tokens:
                token.start_ms += start_ms
                token.end_ms += start_ms
            return tokens, aligner.last_device_info

        def done(payload):
            if not self.session or self.current_chapter_id != chapter_id:
                return
            tokens, device = payload
            if not tokens:
                self._show_error("Qwen ForcedAligner 未返回有效时间")
                return
            self._push_history()
            current = self.segment_model.segments[row]
            current.start_ms = tokens[0].start_ms
            current.end_ms = tokens[-1].end_ms
            current.confidence = 0.9
            current.status = SegmentStatus.MANUAL
            self.session.repository.update_segment(current)
            source_start = sum(len(item.text) for item in self.segment_model.segments[:row])
            source_end = source_start + len(current.text)
            existing = [
                anchor for anchor in self.session.repository.anchors(chapter_id)
                if anchor.source_end_char <= source_start or anchor.source_start_char >= source_end
            ]
            cursor = source_start
            new_anchors = []
            for token in tokens:
                width = max(1, len(token.text.strip()))
                end = min(source_end, cursor + width)
                new_anchors.append(TextAudioAnchor(
                    None, chapter_id, current.id, cursor, end,
                    token.start_ms, token.end_ms, token.probability, "qwen-forced-aligner",
                ))
                cursor = end
            self.session.repository.replace_anchors(chapter_id, [*existing, *new_anchors])
            self.segment_model.set_segments(self.session.repository.segments(chapter_id))
            self.spectrogram.set_segments(self.segment_model.segments)
            self.overview.set_segments(self.segment_model.segments)
            self.article_view.set_content(
                self.segment_model.segments, self.session.repository.anchors(chapter_id), self.current_cache,
            )
            self.model_status_label.setText(device.display_text)
            self.status_stage.setText(f"Qwen 精确对齐完成 · {device.display_text}")
            self._select_segment_row(row)
            self._mark_dirty()

        self.tasks.submit(
            "Qwen ForcedAligner", job, done,
            session_generation=self._session_generation,
        )

    def qwen_align_selected_range(self) -> None:
        """Align contiguous selected sentences against one explicit audio selection."""
        if not self.session or self.current_chapter_id is None or not self.current_parts:
            self._show_error("当前章节没有可用于强制对齐的音频")
            return
        rows = sorted({index.row() for index in self.segment_table.selectionModel().selectedRows()})
        if not rows:
            self._show_error("请先在字幕表格中选择一个或多个连续句子")
            return
        if rows != list(range(rows[0], rows[-1] + 1)):
            self._show_error("范围对齐只接受连续句子，请重新选择连续行")
            return
        selected_segments = [copy.deepcopy(self.segment_model.segments[row]) for row in rows]
        if any(segment.locked for segment in selected_segments):
            self._show_error("所选范围包含锁定句，请先解锁或缩小句子范围")
            return
        selection = self.spectrogram.selection
        if selection is None or selection.end_ms <= selection.start_ms:
            self._show_error("请先在主音频图中框选与这些句子对应的音频范围")
            return
        start_ms, end_ms = selection.start_ms, selection.end_ms
        if end_ms - start_ms > 240_000:
            self._show_error("一次范围强制对齐不能超过 240 秒，请缩小音频选区后分批处理")
            return
        part = next(
            (item for item in self.current_parts if item[3] <= start_ms and end_ms <= item[4]), None
        )
        if part is None:
            self._show_error("音频选区跨越多个资源或切片，请缩小到单个音频切片内")
            return
        language = self._selected_language_code()
        if language == "auto":
            self._show_error("Qwen ForcedAligner 需要明确选择文本语言")
            return
        runtime = runtime_status("Qwen3-ForcedAligner-0.6B", self.paths.models, ASRBackendId.QWEN3_ASR)
        if not runtime.runtime_available:
            self._show_error(runtime.message)
            return
        if not runtime.cuda_available and not self._confirm_qwen_cpu_run(
            f"Qwen 范围强制对齐（{len(rows)} 句）", end_ms - start_ms,
        ):
            return

        link, _asset, path, local_start, _local_end = part
        options = ASROptions(
            backend=ASRBackendId.QWEN3_ASR,
            model="Qwen3-ForcedAligner-0.6B",
            language=language,
            model_root=str(self.paths.models),
            clip_start_ms=link.source_start_ms + start_ms - local_start,
            clip_end_ms=link.source_start_ms + end_ms - local_start,
        )
        chapter_id = self.current_chapter_id
        combined_text = "\n".join(segment.text for segment in selected_segments)
        source_base = sum(len(segment.text) for segment in self.segment_model.segments[:rows[0]])
        source_end = source_base + sum(len(segment.text) for segment in selected_segments)

        def job(progress):
            aligner = Qwen3ForcedAligner()
            tokens = aligner.align(path, combined_text, language, chapter_id, options, progress)
            for token in tokens:
                token.start_ms += start_ms
                token.end_ms += start_ms
            aligned = align_segments_to_tokens(selected_segments, tokens, language=language)
            anchors = anchors_from_segments_tokens(selected_segments, tokens, language=language)
            return tokens, aligned, anchors, aligner.last_device_info

        def done(payload):
            if not self.session or self.current_chapter_id != chapter_id:
                return
            tokens, aligned, anchors, device = payload
            if not tokens or any(item.end_ms <= item.start_ms for item in aligned):
                self._show_error("Qwen 未能为所选的每个句子生成有效时间；请收紧音频起止范围后重试")
                return
            updated = copy.deepcopy(self.segment_model.segments)
            for row, segment in zip(rows, aligned):
                segment.status = SegmentStatus.MANUAL
                updated[row] = segment
            existing_anchors = [
                anchor for anchor in self.session.repository.anchors(chapter_id)
                if anchor.source_end_char <= source_base or anchor.source_start_char >= source_end
            ]
            for anchor in anchors:
                anchor.source_start_char += source_base
                anchor.source_end_char += source_base
                anchor.method = "qwen-forced-aligner-range"
            self._push_history()
            self.session.repository.replace_segments_and_anchors(
                chapter_id, updated, [*existing_anchors, *anchors],
            )
            self.segment_model.set_segments(self.session.repository.segments(chapter_id))
            self.spectrogram.set_segments(self.segment_model.segments)
            self.overview.set_segments(self.segment_model.segments)
            self.article_view.set_content(
                self.segment_model.segments,
                self.session.repository.anchors(chapter_id),
                self.current_cache,
            )
            self.asr_comparison.set_content(
                self.segment_model.segments,
                self.session.repository.asr_tokens(chapter_id),
                self.session.repository.anchors(chapter_id),
            )
            self.model_status_label.setText(device.display_text)
            self.status_stage.setText(
                f"Qwen 范围强制对齐完成 · {len(rows)} 句 · {_time(start_ms)}–{_time(end_ms)} · {device.display_text}"
            )
            self._select_segment_rows(rows, current_row=rows[0])
            self._mark_dirty()

        self.tasks.submit(
            "Qwen ForcedAligner · 句子与音频范围", job, done,
            session_generation=self._session_generation,
        )

    def _start_qwen_block_alignment(
        self,
        start_row: int,
        start_ms: int,
        limit_ms: int,
        *,
        scope: str,
        allow_silence_detection: bool,
    ) -> None:
        """Run the shared, ASR-free block alignment engine."""
        if not self.session or self.current_chapter_id is None or not self.current_parts:
            self._show_error("当前章节没有可用于强制对齐的音频")
            return
        if not self.silence_candidates:
            if allow_silence_detection:
                self.status_stage.setText("块级强制对齐需要VAD候选，正在先检测静音区…")
                self.detect_silence(
                    lambda: self._start_qwen_block_alignment(
                        start_row, start_ms, limit_ms, scope=scope,
                        allow_silence_detection=False,
                    )
                )
            else:
                self._show_error("当前参数没有检测到有效静音区，无法规划强制对齐块")
            return
        segments = copy.deepcopy(self.segment_model.segments)
        if not 0 <= start_row < len(segments):
            self._show_error("请选择有效的文本起点")
            return
        chapter_end = self.current_parts[-1][4]
        start_ms = max(0, int(start_ms))
        limit_ms = min(chapter_end, int(limit_ms))
        if start_ms >= limit_ms:
            self._show_error("请选择章节范围内的音频起点")
            return
        language = self._selected_language_code()
        if language == "auto":
            self._show_error("Qwen ForcedAligner需要明确选择文本语言")
            return
        runtime = runtime_status(
            "Qwen3-ForcedAligner-0.6B", self.paths.models, ASRBackendId.QWEN3_ASR,
        )
        if not runtime.runtime_available:
            self._show_error(runtime.message)
            return
        if not runtime.cuda_available and not self._confirm_qwen_cpu_run(
            f"Qwen {scope}强制对齐", limit_ms - start_ms,
        ):
            return

        chapter_id = self.current_chapter_id
        parts = list(self.current_parts)
        candidates = list(self.silence_candidates)
        planner = ForcedAlignmentPlannerOptions()
        source_ranges: dict[int, tuple[int, int]] = {}
        source_cursor = 0
        for index, segment in enumerate(segments):
            source_ranges[index] = (source_cursor, source_cursor + len(segment.text))
            source_cursor += len(segment.text)

        def job(progress):
            progress_sink = progress
            highest_progress = 0.0

            def monotonic_progress(value: float, message: str) -> None:
                nonlocal highest_progress
                if value >= 0:
                    highest_progress = max(highest_progress, min(1.0, value))
                progress_sink(highest_progress, message)

            # Candidate fallback and overlap revalidation may revisit earlier
            # rows, but the user-visible total must never move backwards.
            progress = monotonic_progress
            aligner = Qwen3ForcedAligner()
            blocks: list[ForcedAlignmentBlockResult] = []
            observed_rate: float | None = None
            segment_index = start_row
            cursor_ms = start_ms
            total = max(1, len(segments) - start_row)
            pending: ForcedAlignmentHypothesis | None = None
            pending_alternatives: list[ForcedAlignmentHypothesis] = []
            backtracks = 0
            stopped_reason = ""
            model_call_count = 0

            def locate_part(milliseconds: int):
                return locate_contiguous_audio_part(parts, milliseconds)

            def build_hypotheses(
                block_start: int,
                audio_start: int,
                *,
                trial_label: str = "",
                trial_budget_override: int | None = None,
            ) -> list[ForcedAlignmentHypothesis]:
                nonlocal observed_rate, model_call_count
                part, audio_start = locate_part(audio_start)
                if part is None or audio_start >= limit_ms:
                    return []
                link, _asset, path, local_start, local_end = part
                block_limit = min(local_end, limit_ms)
                next_locked_start = next(
                    (
                        later.start_ms for later in segments[block_start + 1:]
                        if later.locked and later.end_ms > later.start_ms
                        and later.start_ms > audio_start
                    ),
                    None,
                )
                if next_locked_start is not None:
                    block_limit = min(block_limit, next_locked_start)
                group_ends = forced_alignment_text_group_ends(
                    segments, block_start, language=language,
                    observed_ms_per_unit=observed_rate, options=planner,
                )
                hypotheses: list[ForcedAlignmentHypothesis] = []
                trial_count = 0
                trial_budget = trial_budget_override or planner.candidate_trial_budget
                stable_candidates = 0
                stop_trials = False
                planned_groups: list[tuple[
                    int, list[TextSegment], str, int, list[tuple[int, bool]],
                ]] = []
                for group_end in group_ends:
                    group = segments[block_start:group_end]
                    # Keep explicit sentence boundaries for the aligner. Bad
                    # historical abbreviation splits are repaired at the text
                    # layer instead of weakening every model input boundary.
                    combined_text = "\n".join(item.text for item in group if item.text.strip())
                    if not combined_text:
                        continue
                    # The production forced-alignment path deliberately does
                    # not consult ASR or an existing/manual timeline.  Its only
                    # time cursor is the end proven by the previous forced-
                    # alignment block; VAD may suggest a window end, never a
                    # replacement start.
                    trial_audio_start = audio_start
                    if not local_start <= trial_audio_start < block_limit:
                        continue
                    window_ends = forced_alignment_window_ends(
                        trial_audio_start, block_limit, combined_text, candidates,
                        language=language, observed_ms_per_unit=observed_rate,
                        expected_duration_ms=sum(
                            estimate_spoken_duration_ms(
                                item.text, language, observed_rate,
                            )
                            for item in group if item.text.strip()
                        ),
                        options=planner,
                    )
                    if window_ends:
                        planned_groups.append(
                            (group_end, group, combined_text, trial_audio_start, window_ends)
                        )
                # Try the best VAD end for every nearby text end before trying
                # a second audio end for the first text block. Previously one
                # over-sized block could consume the entire trial budget and
                # the model-capacity-safe shorter alternatives were never run.
                maximum_window_count = max(
                    (len(item[4]) for item in planned_groups), default=0,
                )
                for window_rank in range(maximum_window_count):
                    for group_end, group, combined_text, trial_audio_start, window_ends in planned_groups:
                        if window_rank >= len(window_ends):
                            continue
                        audio_end, weak_boundary = window_ends[window_rank]
                        if trial_count >= trial_budget:
                            break
                        trial_count += 1
                        progress(
                            (block_start - start_row) / total,
                            f"Qwen {scope}块级对齐 · 第 {block_start + 1}–{group_end} 句"
                            f" · 候选 {trial_count}/{trial_budget}{trial_label}",
                        )
                        options = ASROptions(
                            backend=ASRBackendId.QWEN3_ASR,
                            model="Qwen3-ForcedAligner-0.6B",
                            language=language,
                            model_root=str(self.paths.models),
                            clip_start_ms=link.source_start_ms + trial_audio_start - local_start,
                            clip_end_ms=link.source_start_ms + audio_end - local_start,
                        )
                        try:
                            ensure_inference_memory_headroom(aligner)
                            model_call_count += 1
                            tokens = aligner.align(
                                path, combined_text, language, chapter_id, options,
                                lambda _value, message: progress(
                                    (block_start - start_row) / total,
                                    f"{message} · 第 {block_start + 1}–{group_end} 句",
                                ),
                            )
                        except ValueError:
                            continue
                        finally:
                            # Qwen's forced-aligner creates sizeable temporary
                            # tensors for every alternative.  A real chapter
                            # can contain hundreds of source segments, so do a
                            # bounded cleanup periodically rather than letting
                            # candidate trials accumulate until Windows kills
                            # the process.
                            release_inference_memory(
                                aligner, aggressive=bool(model_call_count and model_call_count % 4 == 0),
                            )
                        if not tokens:
                            continue
                        aligned_local = align_segments_to_tokens(group, tokens, language=language)
                        if not forced_alignment_has_usable_coverage(aligned_local):
                            # A long unmatched suffix is model/text-context
                            # truncation and consumes no source rows. Small
                            # internal holes are legitimate edition or heading
                            # differences and do not poison later sentences.
                            continue
                        score, reasons, _measured_rate = score_forced_alignment_hypothesis(
                            aligned_local, tokens,
                            audio_start_ms=trial_audio_start,
                            audio_end_ms=audio_end,
                            candidates=candidates,
                            language=language,
                            observed_ms_per_unit=observed_rate,
                            weak_boundary=weak_boundary,
                        )
                        if score < planner.review_score:
                            continue
                        aligned_absolute = copy.deepcopy(aligned_local)
                        absolute_tokens = copy.deepcopy(tokens)
                        for segment in aligned_absolute:
                            # Keep even an unmatched row in chapter-absolute
                            # coordinates.  Previously its zero-length local
                            # timestamp could become the next block's cursor and
                            # reset a chapter back toward its beginning.
                            segment.start_ms += trial_audio_start
                            segment.end_ms += trial_audio_start
                            if segment.end_ms <= segment.start_ms:
                                segment.confidence = 0.0
                                segment.status = SegmentStatus.UNMATCHED
                                continue
                            segment.confidence = score
                            segment.status = (
                                SegmentStatus.AUTO
                                if score >= planner.stable_score
                                else SegmentStatus.LOW_CONFIDENCE
                            )
                        for token in absolute_tokens:
                            token.start_ms += trial_audio_start
                            token.end_ms += trial_audio_start
                        hypotheses.append(ForcedAlignmentHypothesis(
                            block_start, group_end, trial_audio_start, audio_end,
                            aligned_absolute, absolute_tokens, score, reasons, weak_boundary,
                        ))
                        if forced_alignment_candidate_can_end_search(
                            aligned_local, score,
                            stable_score=planner.stable_score,
                        ):
                            stable_candidates += 1
                            if stable_candidates >= planner.stable_candidates_required:
                                stop_trials = True
                                break
                    if trial_count >= trial_budget or stop_trials:
                        break
                hypotheses.sort(
                    key=lambda item: forced_alignment_hypothesis_priority(
                        item, stable_score=planner.stable_score,
                    ),
                    reverse=True,
                )
                return hypotheses[:planner.beam_width]

            def append_block(hypothesis: ForcedAlignmentHypothesis, status: str) -> None:
                blocks.append(ForcedAlignmentBlockResult(
                    hypothesis.start_index, hypothesis.end_index, status,
                    [hypothesis], hypothesis,
                    "；".join(hypothesis.reasons),
                ))

            while segment_index < len(segments):
                progress(
                    (segment_index - start_row) / total,
                    f"Qwen {scope}块级强制对齐 · 从第 {segment_index + 1} 句开始",
                )
                if cursor_ms >= limit_ms:
                    stopped_reason = "已到达音频范围终点"
                    break
                current = segments[segment_index]
                if current.locked:
                    if pending is not None:
                        append_block(pending, "review" if pending.score < planner.stable_score else "stable")
                        pending = None
                    if current.end_ms <= current.start_ms or current.end_ms < cursor_ms:
                        stopped_reason = f"第 {segment_index + 1} 句是无可用时间的锁定锚点"
                        break
                    # A locked segment is already an explicit hard anchor. Its
                    # confirmed end, rather than a nearby VAD pause, owns the
                    # next audio cursor.
                    cursor_ms = current.end_ms
                    segment_index += 1
                    continue
                if not current.text.strip():
                    segment_index += 1
                    continue

                hypotheses = build_hypotheses(segment_index, cursor_ms)
                if not hypotheses:
                    # Source-side differences (an unspoken heading, edition
                    # sentence, etc.) may be skipped locally, but the audio
                    # cursor is immutable until a real alignment proves its
                    # next position.  Never search forward through VAD starts:
                    # that was the cause of repeated 50–90 second jumps.
                    maximum_skip = min(
                        planner.leading_text_search_segments,
                        len(segments) - segment_index - 1,
                    )
                    for skipped in range(1, maximum_skip + 1):
                        candidate_start = segment_index + skipped
                        if any(item.locked for item in segments[segment_index:candidate_start]):
                            break
                        hypotheses = build_hypotheses(
                            candidate_start, cursor_ms,
                            trial_label=" · 同一音频位置跳过不一致文本",
                            trial_budget_override=2,
                        )
                        if not hypotheses:
                            continue
                        blocks.append(ForcedAlignmentBlockResult(
                            segment_index, candidate_start,
                            "failed", [], None,
                            "局部文本与音频不一致；音频游标未前移",
                        ))
                        segment_index = candidate_start
                        for hypothesis in hypotheses:
                            hypothesis.score = min(
                                hypothesis.score, planner.stable_score - 0.01,
                            )
                            hypothesis.reasons = (
                                *hypothesis.reasons, "跳过局部源文本后在同一音频位置重新同步",
                            )
                        break
                if not hypotheses:
                    if pending is not None:
                        pending.score = min(pending.score, planner.stable_score - 0.01)
                        pending.reasons = (*pending.reasons, "后续块未能验证")
                        append_block(pending, "review")
                    blocks.append(ForcedAlignmentBlockResult(
                        segment_index,
                        min(len(segments), segment_index + planner.minimum_segments),
                        "failed", [], None,
                        "没有找到达到最低评分的候选块",
                    ))
                    stopped_reason = f"第 {segment_index + 1} 句附近无法重新建立稳定对齐"
                    break

                if pending is None:
                    pending = hypotheses[0]
                    pending_alternatives = hypotheses[1:]
                else:
                    matched = next(
                        (
                            hypothesis for hypothesis in hypotheses
                            if forced_alignment_overlap_score(
                                pending, hypothesis,
                                overlap_segments=planner.overlap_segments,
                            ) > 0
                        ),
                        None,
                    )
                    while matched is None and pending_alternatives and backtracks < planner.maximum_backtracks:
                        pending = pending_alternatives.pop(0)
                        backtracks += 1
                        segment_index = max(
                            pending.start_index,
                            pending.end_index - planner.overlap_segments,
                        )
                        cursor_ms = pending.segments[-min(
                            planner.overlap_segments, len(pending.segments)
                        )].start_ms
                        hypotheses = build_hypotheses(
                            segment_index, cursor_ms, trial_label=" · 回溯验证",
                        )
                        matched = next(
                            (
                                hypothesis for hypothesis in hypotheses
                                if forced_alignment_overlap_score(
                                    pending, hypothesis,
                                    overlap_segments=planner.overlap_segments,
                                ) > 0
                            ),
                            None,
                        )
                    if matched is None:
                        pending.score = min(pending.score, planner.stable_score - 0.01)
                        pending.reasons = (*pending.reasons, "与后续验证块不一致")
                        append_block(pending, "review")
                        blocks.append(ForcedAlignmentBlockResult(
                            segment_index,
                            min(len(segments), segment_index + planner.minimum_segments),
                            "failed", [], None, "验证失败；禁止向后跳过音频",
                        ))
                        stopped_reason = f"第 {segment_index + 1} 句附近无法在当前音频游标继续"
                        break
                    else:
                        overlap_score = forced_alignment_overlap_score(
                            pending, matched, overlap_segments=planner.overlap_segments,
                        )
                        if overlap_score < 0.35:
                            pending.score = min(
                                pending.score, planner.stable_score - 0.01,
                            )
                            pending.reasons = (*pending.reasons, "重叠句接近容差上限")
                        append_block(
                            pending,
                            "stable" if pending.score >= planner.stable_score else "review",
                        )
                        pending = matched
                        pending_alternatives = [item for item in hypotheses if item is not matched]

                timed_pending = [
                    item for item in pending.segments if item.end_ms > item.start_ms
                ]
                if not timed_pending:
                    append_block(pending, "review")
                    stopped_reason = f"第 {pending.start_index + 1} 句附近没有有效时间"
                    pending = None
                    break
                units = sum(spoken_unit_count(item.text, language) for item in timed_pending)
                measured_rate = (
                    timed_pending[-1].end_ms - timed_pending[0].start_ms
                ) / max(1, units)
                observed_rate = measured_rate if observed_rate is None else observed_rate * 0.7 + measured_rate * 0.3
                if pending.end_index >= len(segments):
                    append_block(
                        pending,
                        "stable" if pending.score >= planner.stable_score else "review",
                    )
                    pending = None
                    segment_index = len(segments)
                    break
                overlap = min(planner.overlap_segments, len(pending.segments) - 1)
                if overlap <= 0:
                    append_block(pending, "review" if pending.score < planner.stable_score else "stable")
                    # Qwen3-aligner-style strict cursor: only a model-confirmed
                    # sentence end may advance audio. VAD is allowed to choose
                    # a trial window end, never the next window start.
                    cursor_ms = timed_pending[-1].end_ms
                    segment_index = pending.end_index
                    pending = None
                else:
                    segment_index = pending.end_index - overlap
                    cursor_ms = pending.segments[-overlap].start_ms

            if pending is not None:
                pending.score = min(pending.score, planner.stable_score - 0.01)
                pending.reasons = (*pending.reasons, "没有后续重叠块确认")
                append_block(pending, "review")

            aligned_rows = {
                block.start_index + offset
                for block in blocks if block.selected is not None
                for offset, segment in enumerate(block.selected.segments)
                if segment.end_ms > segment.start_ms
            }
            review_rows = {
                block.start_index + offset
                for block in blocks if block.status == "review" and block.selected is not None
                for offset in range(len(block.selected.segments))
            }
            review_rows.update(
                index
                for block in blocks if block.status == "failed"
                for index in range(block.start_index, block.end_index)
            )
            progress(1.0, f"Qwen {scope}块级对齐结束 · 完成 {len(aligned_rows)} 句")
            device = aligner.last_device_info
            close_aligner = getattr(aligner, "close", None)
            if callable(close_aligner):
                close_aligner()
            return ForcedAlignmentRunResult(
                blocks, len(aligned_rows), len(review_rows), stopped_reason,
            ), device

        def done(payload):
            if not self.session or self.current_chapter_id != chapter_id:
                return
            result, device = payload
            updated = copy.deepcopy(self.segment_model.segments)
            final_rows: dict[int, tuple[TextSegment, list]] = {}
            affected: set[int] = set()
            for block in result.blocks:
                if block.selected is None:
                    for row_index in range(block.start_index, min(block.end_index, len(updated))):
                        target = updated[row_index]
                        if not target.locked and target.status != SegmentStatus.UNMATCHED:
                            target.status = SegmentStatus.LOW_CONFIDENCE
                            target.confidence = min(target.confidence, 0.49)
                            affected.add(row_index)
                    continue
                for offset, aligned_segment in enumerate(block.selected.segments):
                    row_index = block.start_index + offset
                    if not 0 <= row_index < len(updated) or updated[row_index].locked:
                        continue
                    if aligned_segment.end_ms <= aligned_segment.start_ms:
                        # Retain the original unmatched source text. A visual
                        # heading omitted by the narrator must never become a
                        # zero-duration AUTO segment at the block start.
                        continue
                    replacement = copy.deepcopy(aligned_segment)
                    if block.status == "review":
                        replacement.status = SegmentStatus.LOW_CONFIDENCE
                        replacement.confidence = min(replacement.confidence, planner.stable_score - 0.01)
                    else:
                        replacement.status = SegmentStatus.AUTO
                    overlapping_tokens = [
                        token for token in block.selected.tokens
                        if token.end_ms > replacement.start_ms and token.start_ms < replacement.end_ms
                    ]
                    # Overlap validates the next block; it is not a second
                    # writer. The earlier confirmed block owns the shared rows.
                    # Letting a later candidate overwrite them made one local
                    # mismatch repeatedly pull paragraphs toward chapter start.
                    final_rows.setdefault(row_index, (replacement, overlapping_tokens))
                    affected.add(row_index)
            _accepted_rows, regressive_rows = partition_monotonic_alignment_rows(
                [(row_index, replacement) for row_index, (replacement, _tokens) in final_rows.items()]
            )
            for row_index in regressive_rows:
                final_rows.pop(row_index, None)
                affected.discard(row_index)
            if regressive_rows:
                first_regression = min(regressive_rows)
                result.review_count += len(regressive_rows)
                regression_message = (
                    f"第 {first_regression + 1} 句返回时间早于已确认游标；"
                    f"已拒绝 {len(regressive_rows)} 个倒退结果"
                )
                result.stopped_reason = (
                    f"{result.stopped_reason}；{regression_message}"
                    if result.stopped_reason else regression_message
                )
            for row_index, (replacement, _tokens) in final_rows.items():
                target = updated[row_index]
                target.start_ms = replacement.start_ms
                target.end_ms = replacement.end_ms
                target.confidence = replacement.confidence
                target.status = replacement.status

            existing_anchors = [
                anchor for anchor in self.session.repository.anchors(chapter_id)
                if not any(
                    source_ranges[index][0] < anchor.source_end_char
                    and anchor.source_start_char < source_ranges[index][1]
                    for index in affected
                )
            ]
            new_anchors: list[TextAudioAnchor] = []
            for row_index, (replacement, tokens) in final_rows.items():
                if not tokens:
                    continue
                generated = anchors_from_segments_tokens([replacement], tokens, language=language)
                source_start, _source_end = source_ranges[row_index]
                for anchor in generated:
                    anchor.source_start_char += source_start
                    anchor.source_end_char += source_start
                    anchor.method = "qwen-forced-aligner-block"
                new_anchors.extend(generated)
            if not affected:
                self.model_status_label.setText(device.display_text)
                self.status_stage.setText(
                    f"Qwen {scope}未能建立首个稳定块 · 原数据已保留 · "
                    f"{result.stopped_reason or '请从正文句和对应音频位置建立锚点后重试'}"
                )
                return
            self._push_history()
            self.session.repository.replace_chapter_edit_state(
                chapter_id, updated, [*existing_anchors, *new_anchors],
            )
            self.segment_model.set_segments(self.session.repository.segments(chapter_id))
            self._refresh_segment_views(defer_hidden=True, refresh_audio=True)
            self.model_status_label.setText(device.display_text)
            if result.stopped_reason:
                self.status_stage.setText(
                    f"Qwen {scope}部分完成 · {result.completed_count} 句 · "
                    f"{result.review_count} 句待检查 · {result.stopped_reason}"
                )
            else:
                self.status_stage.setText(
                    f"Qwen {scope}完成 · {result.completed_count} 句 · "
                    f"{result.review_count} 句待检查 · {device.display_text}"
                )
            self._mark_dirty()

        self.tasks.submit(
            f"Qwen ForcedAligner · {scope}块级对齐", job, done,
            session_generation=self._session_generation,
        )

    def qwen_align_from_current_anchor(self) -> None:
        self._qwen_align_vad_progressive(allow_silence_detection=True)

    def _qwen_align_vad_progressive(self, *, allow_silence_detection: bool) -> None:
        """Start shared block alignment from the selected text/audio anchor."""
        if not self.session or self.current_chapter_id is None or not self.current_parts:
            self._show_error("当前章节没有可用于强制对齐的音频")
            return
        row = self.segment_table.currentIndex().row()
        if not 0 <= row < len(self.segment_model.segments):
            self._show_error("请先选择作为文本起点的句子")
            return
        if self.segment_model.segments[row].locked:
            self._show_error("起始句已锁定，请先解锁或选择后面的句子")
            return
        selection = self.spectrogram.selection
        explicit_selection = bool(
            selection and selection.end_ms > selection.start_ms
            and not self.spectrogram.selection_tracks_segment
        )
        start_ms = selection.start_ms if explicit_selection else self._local_position()
        chapter_end = self.current_parts[-1][4]
        limit_ms = selection.end_ms if explicit_selection else chapter_end
        if not 0 <= start_ms < limit_ms <= chapter_end:
            self._show_error("请选择章节范围内的音频起点")
            return
        self._start_qwen_block_alignment(
            row, start_ms, limit_ms,
            scope="从当前句向后",
            allow_silence_detection=allow_silence_detection,
        )

    def qwen_align_chapter(self) -> None:
        self._qwen_align_chapter(0, 0, anchored=False)

    def _qwen_align_chapter(self, start_row: int, start_ms: int, *, anchored: bool) -> None:
        """Start the shared block engine for a whole chapter or explicit anchor."""
        if not self.current_parts:
            self._show_error("当前章节没有可用于强制对齐的音频")
            return
        self._start_qwen_block_alignment(
            start_row, start_ms, self.current_parts[-1][4],
            scope="从锚点向后" if anchored else "整章",
            allow_silence_detection=True,
        )

    @staticmethod
    def _parts_silence_signature(
        parts: list[tuple[ChapterAudioLink, AudioAsset, Path, int, int]],
        settings: SilenceSettings,
    ) -> str:
        part_signature = "|".join(
            f"{link.audio_id}:{link.source_start_ms}:{link.source_end_ms}"
            for link, _asset, _path, _local_start, _local_end in parts
        )
        raw = (
            f"{part_signature}|vad={settings.vad_threshold:.3f}|min={settings.min_silence_ms}|"
            f"pad={settings.boundary_padding_ms}|snap={settings.snap_window_ms}|"
            f"energy={settings.energy_percentile:.3f}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _silence_signature(self) -> str:
        return self._parts_silence_signature(self.current_parts, self._silence_settings())

    def detect_book_silence(self) -> None:
        if not self.session:
            return
        settings = self._silence_settings()
        prepared = []
        skipped = []
        for chapter in self.session.repository.chapters():
            chapter_id = chapter.id or 0
            parts, error = self._chapter_audio_parts(chapter_id)
            if parts:
                prepared.append((chapter_id, chapter.title, parts))
            else:
                skipped.append(f"{chapter.title}：{error}")
        if not prepared:
            self._show_error("全书没有已配对且可读取的音频章节。")
            return
        answer = QMessageBox.question(
            self,
            "检测全书静音区",
            f"将按当前 VAD/静音参数检测 {len(prepared)} 个章节。"
            f"{' 将跳过 ' + str(len(skipped)) + ' 章。' if skipped else ''}\n\n"
            "检测只更新静音候选标记，不会直接修改句段时间。是否开始？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        database = Path(self.session.repository.database)
        total_chapters = len(prepared)
        total_duration = sum(parts[-1][4] for _chapter_id, _title, parts in prepared)
        self._mark_dirty()

        def job(progress):
            repository = ProjectRepository(database)
            counts: list[tuple[int, int]] = []
            processed_audio_ms = 0
            try:
                for chapter_index, (chapter_id, title, parts) in enumerate(prepared):
                    result = []
                    part_total = max(1, len(parts))
                    chapter_duration = parts[-1][4]
                    for part_index, (link, _asset, path, local_start, local_end) in enumerate(parts):
                        progress(
                            (processed_audio_ms + local_start) / max(1, total_duration),
                            f"全书 {chapter_index + 1}/{total_chapters} · {title} · "
                            f"解码/检测 {part_index + 1}/{part_total} · "
                            f"已处理 {_time(processed_audio_ms + local_start)}/{_time(total_duration)}",
                        )
                        samples, rate = decode_audio_mono(
                            path, 16000,
                            start_ms=link.source_start_ms,
                            end_ms=link.source_end_ms,
                        )
                        local_candidates = detect_silence_candidates(samples, rate, settings)
                        for candidate in local_candidates:
                            candidate.time_ms += local_start
                            if candidate.start_ms is not None:
                                candidate.start_ms += local_start
                            if candidate.end_ms is not None:
                                candidate.end_ms += local_start
                            result.append(candidate)
                    repository.replace_silence_candidates(
                        chapter_id,
                        result,
                        self._parts_silence_signature(parts, settings),
                    )
                    counts.append((chapter_id, len(result)))
                    processed_audio_ms += chapter_duration
                    progress(
                        processed_audio_ms / max(1, total_duration),
                        f"全书 {chapter_index + 1}/{total_chapters} · {title} · {len(result)} 个候选 · "
                        f"已处理 {_time(processed_audio_ms)}/{_time(total_duration)}",
                    )
                return counts, skipped
            finally:
                repository.close()

        def done(payload):
            if not self.session or Path(self.session.repository.database) != database:
                return
            counts, skipped_chapters = payload
            if self.current_chapter_id is not None:
                parts, _error = self._chapter_audio_parts(self.current_chapter_id)
                signature = self._parts_silence_signature(parts, settings) if parts else ""
                self.silence_candidates = self.session.repository.silence_candidates(
                    self.current_chapter_id, signature,
                ) if signature else []
                self._refresh_silence_overlays()
            self.status_stage.setText(
                f"全书静音检测完成 · {len(counts)}/{total_chapters} 章 · "
                f"{sum(count for _chapter_id, count in counts)} 个候选 · 跳过 {len(skipped_chapters)} 章"
            )
            self._mark_dirty()

        self.tasks.submit(
            "检测全书静音区", job, done,
            session_generation=self._session_generation,
        )

    def detect_silence(self, after=None) -> None:
        if not callable(after):
            after = None
        if not self.session or not self.current_parts:
            return
        settings = self._silence_settings()
        parts = list(self.current_parts)
        chapter_id = self.current_chapter_id
        signature = self._silence_signature()

        def job(progress):
            result = []
            total = max(1, len(parts))
            for index, (link, _asset, path, local_start, _local_end) in enumerate(parts):
                progress(index / total, f"解码并检测静音 {index + 1}/{total}")
                samples, rate = decode_audio_mono(
                    path, 16000, start_ms=link.source_start_ms, end_ms=link.source_end_ms,
                )
                local_candidates = detect_silence_candidates(samples, rate, settings)
                for candidate in local_candidates:
                    candidate.time_ms += local_start
                    if candidate.start_ms is not None:
                        candidate.start_ms += local_start
                    if candidate.end_ms is not None:
                        candidate.end_ms += local_start
                    result.append(candidate)
                progress((index + 1) / total, f"静音检测 {index + 1}/{total}")
            progress(1.0, "静音检测完成")
            return result

        def done(result):
            if not self.session or chapter_id != self.current_chapter_id:
                return
            self.silence_candidates = result
            self._refresh_silence_overlays()
            self.session.repository.replace_silence_candidates(chapter_id or 0, result, signature)
            self.status_stage.setText(f"静音检测完成：{len(result)} 个候选")
            self._mark_dirty()
            if after:
                after()

        self.tasks.submit(
            "静音检测", job, done,
            session_generation=self._session_generation,
        )

    def auto_align_from_silence(self) -> None:
        if not self.session or self.current_chapter_id is None or not self.current_parts:
            return
        if not self.silence_candidates:
            self.detect_silence(self.auto_align_from_silence)
            return
        row = self.segment_table.currentIndex().row() if self.segment_table.currentIndex().isValid() else 0
        if not (0 <= row < len(self.segment_model.segments)):
            return
        selection = self.spectrogram.selection
        start_ms = selection.start_ms if selection and selection.end_ms > selection.start_ms else self._local_position()
        chapter_end = self.current_parts[-1][4]
        end_ms = selection.end_ms if selection and selection.end_ms > selection.start_ms else chapter_end
        if end_ms <= start_ms:
            self.status_stage.setText("静音自动分配需要有效的起点和终点")
            return
        options = SilenceAlignmentOptions(
            start_segment_index=row,
            start_ms=start_ms,
            end_ms=end_ms,
            padding_ms=self.padding_spin.value(),
        )
        result = align_segments_from_silence(self.segment_model.segments, self.silence_candidates, options)
        if result.proposed_count <= 0:
            self.status_stage.setText("没有足够可靠的静音候选可用于粗分配")
            return
        low = sum(
            segment.status == SegmentStatus.LOW_CONFIDENCE
            for segment in result.segments[row:row + result.proposed_count]
        )
        suffix = "\n候选不足，已在误差可能扩散前停止。" if result.stopped_early else ""
        original_segments = list(self.segment_model.segments)
        self.spectrogram.set_segments(result.segments)
        self.overview.set_segments(result.segments)
        QApplication.processEvents()
        answer = QMessageBox.question(
            self,
            "预览静音粗分配",
            f"建议调整 {result.proposed_count} 句，其中 {low} 句标记为待检查。{suffix}\n\n是否应用这些时间？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.spectrogram.set_segments(original_segments)
            self.overview.set_segments(original_segments)
            self.status_stage.setText("已取消静音粗分配，项目时间轴未修改")
            return
        self._push_history()
        for index, segment in enumerate(result.segments):
            segment.position = index
            segment.chapter_id = self.current_chapter_id
        self.session.repository.replace_segments_and_anchors(
            self.current_chapter_id, result.segments, result.anchors
        )
        self.segment_model.set_segments(self.session.repository.segments(self.current_chapter_id))
        self.spectrogram.set_segments(self.segment_model.segments)
        self.overview.set_segments(self.segment_model.segments)
        self.article_view.set_content(
            self.segment_model.segments,
            self.session.repository.anchors(self.current_chapter_id),
            self.current_cache,
        )
        self.status_stage.setText(f"静音粗分配完成：{result.proposed_count} 句，{low} 句待检查")
        self._select_segment_row(row)
        self._mark_dirty()

    def apply_silence(self) -> None:
        if not self.session or self.current_chapter_id is None or not self.silence_candidates:
            return
        self._push_history()
        settings = self._silence_settings()
        updated = snap_boundaries(
            self.segment_model.segments,
            self.silence_candidates,
            window_ms=settings.snap_window_ms,
            padding_ms=settings.boundary_padding_ms,
        )
        changed = sum(
            (left.start_ms, left.end_ms) != (right.start_ms, right.end_ms)
            for left, right in zip(self.segment_model.segments, updated)
        )
        self._replace_segments(updated)
        self.status_stage.setText(f"已按静音调整 {changed} 个句段")

    def find_next_silence(self) -> None:
        position = self._local_position()
        following = [candidate.time_ms for candidate in self.silence_candidates if candidate.time_ms > position]
        if not following:
            self.status_stage.setText("当前位置之后没有已检测的静音候选")
            return
        target = min(following)
        self._seek_local(target)
        self.spectrogram.focus_time(target)

    def _silence_settings(self) -> SilenceSettings:
        return SilenceSettings(
            self.vad_spin.value(),
            self.min_silence_spin.value(),
            self.padding_spin.value(),
            self.snap_spin.value(),
        )

    def _nearest_silence(self, milliseconds: int, kind: str) -> int:
        window = self.snap_spin.value()
        targets = []
        for candidate in self.silence_candidates:
            previous_end, following_start = silence_boundary_edges(candidate)
            target = following_start if kind == "start" else previous_end
            if abs(target - milliseconds) <= window:
                targets.append(target)
        return min(targets, key=lambda value: abs(value - milliseconds)) if targets else milliseconds

    def _overlap_policy(self) -> SegmentOverlapPolicy:
        value = self.overlap_policy_combo.currentData() if hasattr(self, "overlap_policy_combo") else None
        return SegmentOverlapPolicy(value or SegmentOverlapPolicy.CLAMP_CURRENT)

    def _begin_segment_drag(self) -> None:
        self._push_history()
        self._drag_changed_rows.clear()

    @staticmethod
    def _has_timing(segment: TextSegment | None) -> bool:
        return bool(segment and segment.end_ms > segment.start_ms)

    def _nearest_timed_neighbor(self, row: int, direction: int) -> tuple[int, TextSegment | None]:
        """Find the nearest timed cue, skipping unmatched text-only rows."""
        index = row + direction
        while 0 <= index < len(self.segment_model.segments):
            candidate = self.segment_model.segments[index]
            if self._has_timing(candidate):
                return index, candidate
            index += direction
        return -1, None

    def _apply_adjacent_policy(self, row: int, kind: str) -> None:
        segments = self.segment_model.segments
        current = segments[row]
        previous_row, previous = self._nearest_timed_neighbor(row, -1)
        following_row, following = self._nearest_timed_neighbor(row, 1)
        policy = self._overlap_policy()
        minimum = 100
        if policy == SegmentOverlapPolicy.ALLOW_OVERLAP:
            return
        if kind in {"start", "body"} and self._has_timing(previous) and current.start_ms < previous.end_ms:
            if policy == SegmentOverlapPolicy.TRIM_NEIGHBORS and not previous.locked:
                trimmed = max(previous.start_ms + minimum, current.start_ms)
                if trimmed <= current.start_ms:
                    previous.end_ms = trimmed
                    previous.status = SegmentStatus.MANUAL
                    self._drag_changed_rows.add(previous_row)
                else:
                    current.start_ms = previous.end_ms
            else:
                current.start_ms = previous.end_ms
        if kind in {"end", "body"} and self._has_timing(following) and current.end_ms > following.start_ms:
            if policy == SegmentOverlapPolicy.TRIM_NEIGHBORS and not following.locked:
                trimmed = min(following.end_ms - minimum, current.end_ms)
                if trimmed >= current.end_ms:
                    following.start_ms = trimmed
                    following.status = SegmentStatus.MANUAL
                    self._drag_changed_rows.add(following_row)
                else:
                    current.end_ms = following.start_ms
            else:
                current.end_ms = following.start_ms
        if current.end_ms < current.start_ms:
            if kind == "start":
                current.start_ms = current.end_ms
            else:
                current.end_ms = current.start_ms

    def _boundary_moved(self, row: int, kind: str, milliseconds: int, snap: bool) -> None:
        if not (0 <= row < len(self.segment_model.segments)):
            return
        segment = self.segment_model.segments[row]
        if segment.locked:
            return
        selection_follows, manual_play_follows = self._ranges_follow_segment(segment)
        value = self._nearest_silence(milliseconds, kind) if snap else milliseconds
        if kind == "start":
            segment.start_ms = max(0, min(segment.end_ms, value))
        else:
            segment.end_ms = max(segment.start_ms, value)
        self._apply_adjacent_policy(row, kind)
        segment.status = SegmentStatus.MANUAL
        self._drag_changed_rows.add(row)
        self.spectrogram.preview_segment(row)
        for changed in self._drag_changed_rows - {row}:
            self.spectrogram.preview_segment(changed)
        self._sync_ranges_to_segment(segment, selection_follows, manual_play_follows)

    def _shift_segment(self, row: int, delta_ms: int) -> None:
        if not (0 <= row < len(self.segment_model.segments)) or not delta_ms:
            return
        segment = self.segment_model.segments[row]
        if segment.locked:
            return
        selection_follows, manual_play_follows = self._ranges_follow_segment(segment)
        duration = max(0, segment.end_ms - segment.start_ms)
        chapter_end = self.current_parts[-1][4] if self.current_parts else max(segment.end_ms, 0)
        start = max(0, min(max(0, chapter_end - duration), segment.start_ms + delta_ms))
        if self._overlap_policy() == SegmentOverlapPolicy.CLAMP_CURRENT:
            _previous_row, previous = self._nearest_timed_neighbor(row, -1)
            _following_row, following = self._nearest_timed_neighbor(row, 1)
            lower = previous.end_ms if self._has_timing(previous) else 0
            upper = following.start_ms - duration if self._has_timing(following) else chapter_end - duration
            start = max(lower, min(max(lower, upper), start))
        segment.start_ms, segment.end_ms = start, start + duration
        self._apply_adjacent_policy(row, "body")
        segment.status = SegmentStatus.MANUAL
        self._drag_changed_rows.add(row)
        self.spectrogram.preview_segment(row)
        for changed in self._drag_changed_rows - {row}:
            self.spectrogram.preview_segment(changed)
        self._sync_ranges_to_segment(segment, selection_follows, manual_play_follows)

    def _ranges_follow_segment(self, segment: TextSegment) -> tuple[bool, bool]:
        """Identify cue-derived ranges before mutating the cue's timestamps."""
        selection = self.spectrogram.selection
        selection_follows = bool(
            selection
            and self.spectrogram.selection_tracks_segment
            and selection.start_ms == segment.start_ms
            and selection.end_ms == segment.end_ms
        )
        manual_play_follows = bool(
            self._play_range_segment_id is None
            and self._play_range_segment_row is None
            and self._play_range_start == segment.start_ms
            and self._play_range_end == segment.end_ms
        )
        return selection_follows, manual_play_follows

    def _sync_ranges_to_segment(
        self, segment: TextSegment, selection_follows: bool, manual_play_follows: bool,
    ) -> None:
        """Move only cue-derived selection/play ranges; preserve manual ranges."""
        if selection_follows:
            if segment.end_ms > segment.start_ms:
                self.spectrogram.set_segment_selection(segment.start_ms, segment.end_ms)
            else:
                self.spectrogram.clear_selection()
        if manual_play_follows:
            self._play_range_start = segment.start_ms
            self._play_range_end = segment.end_ms

    def _commit_dragged_segment(self, row: int) -> None:
        """Commit one drag atomically after mouse release, not once per pixel."""
        if not self.session or not (0 <= row < len(self.segment_model.segments)):
            return
        segment = self.segment_model.segments[row]
        if segment.locked:
            return
        changed_rows = sorted(self._drag_changed_rows or {row})
        if len(changed_rows) == 1:
            self.session.repository.update_segment(self.segment_model.segments[changed_rows[0]])
        else:
            self.session.repository.update_segments([self.segment_model.segments[index] for index in changed_rows])
        first = self.segment_model.index(max(0, min(changed_rows) - 1), 0)
        last = self.segment_model.index(min(self.segment_model.rowCount() - 1, max(changed_rows) + 1), 3)
        self.segment_model.dataChanged.emit(first, last)
        self.overview.set_segments(self.segment_model.segments)
        self.article_view.set_content(
            self.segment_model.segments,
            self.session.repository.anchors(self.current_chapter_id or 0),
            self.current_cache,
        )
        if self.segment_table.currentIndex().row() == row:
            self._load_segment_editor(row)
        self._refresh_segment_play_range()
        self._drag_changed_rows.clear()
        self._mark_dirty()

    def _selected_rows(self) -> list[int]:
        rows = sorted({index.row() for index in self.segment_table.selectionModel().selectedRows()})
        if not rows and self.segment_table.currentIndex().isValid():
            rows = [self.segment_table.currentIndex().row()]
        return rows

    def bind_selection(self) -> None:
        rows = self._selected_rows()
        selection = self.spectrogram.selection
        if not rows or not selection:
            return
        segment = self.segment_model.segments[rows[0]]
        if segment.locked:
            return
        self._push_history()
        segment.start_ms, segment.end_ms = selection.start_ms, selection.end_ms
        segment.status = SegmentStatus.MANUAL
        self._persist_current_segments([rows[0]])

    def distribute_selection(self, proportional: bool = False) -> None:
        rows = self._selected_rows()
        selection = self.spectrogram.selection
        if not rows or not selection:
            return
        self._push_history()
        segments = [self.segment_model.segments[row] for row in rows]
        weights = [max(1, segment.end_ms - segment.start_ms) if proportional else 1 for segment in segments]
        total = sum(weights)
        cursor = selection.start_ms
        for index, (segment, weight) in enumerate(zip(segments, weights)):
            end = selection.end_ms if index == len(segments) - 1 else cursor + round((selection.end_ms - selection.start_ms) * weight / total)
            if not segment.locked:
                segment.start_ms, segment.end_ms, segment.status = cursor, end, SegmentStatus.MANUAL
            cursor = end
        self._persist_current_segments(rows)

    def new_segment_from_selection(self) -> None:
        if self.current_chapter_id is None or not self.spectrogram.selection:
            return
        text, ok = QInputDialog.getText(self, "新建句段", "文本：")
        if not ok:
            return
        self._push_history()
        selection = self.spectrogram.selection
        segments = self.segment_model.segments + [
            TextSegment(
                None, self.current_chapter_id, len(self.segment_model.segments), text,
                selection.start_ms, selection.end_ms, 1.0, SegmentStatus.MANUAL,
                origin=SegmentOrigin.USER,
            )
        ]
        segments.sort(key=lambda item: (item.start_ms, item.position))
        self._replace_segments(segments)

    def set_boundary(self, kind: str, shift_following: bool = False) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        self._push_history()
        row = rows[0]
        segment = self.segment_model.segments[row]
        local = self._local_position()
        old = segment.start_ms if kind == "start" else segment.end_ms
        if kind == "start":
            segment.start_ms = min(local, segment.end_ms)
        else:
            segment.end_ms = max(local, segment.start_ms)
        self._apply_adjacent_policy(row, kind)
        segment.status = SegmentStatus.MANUAL
        if shift_following:
            delta = (segment.start_ms if kind == "start" else segment.end_ms) - old
            for following in self.segment_model.segments[row + 1 :]:
                if not following.locked:
                    following.start_ms += delta
                    following.end_ms += delta
        changed_rows = list(range(row, len(self.segment_model.segments))) if shift_following else [row]
        if self._drag_changed_rows:
            changed_rows.extend(self._drag_changed_rows)
        self._persist_current_segments(sorted(set(changed_rows)))

    def set_boundary_from_selection(self, kind: str) -> None:
        """Apply an explicit audio selection edge to the selected cue range."""
        rows = self._selected_rows()
        selection = self.spectrogram.selection
        if (
            not rows or selection is None
            or selection.end_ms <= selection.start_ms
            or self.spectrogram.selection_tracks_segment
        ):
            self.status_stage.setText("请先在音频图中框选一个时间范围")
            return
        row = rows[0] if kind == "start" else rows[-1]
        segment = self.segment_model.segments[row]
        if segment.locked:
            self.status_stage.setText("目标句段已锁定，请先解锁")
            return
        value = selection.start_ms if kind == "start" else selection.end_ms
        old_value = segment.start_ms if kind == "start" else segment.end_ms
        if value == old_value:
            return
        self._push_history()
        self._drag_changed_rows.clear()
        if kind == "start":
            segment.start_ms = min(value, segment.end_ms)
        else:
            segment.end_ms = max(value, segment.start_ms)
        self._apply_adjacent_policy(row, kind)
        segment.status = SegmentStatus.MANUAL
        changed_rows = {row, *self._drag_changed_rows}
        self._persist_current_segments(sorted(changed_rows))
        target = "首句开始" if len(rows) > 1 and kind == "start" else (
            "末句结束" if len(rows) > 1 else ("当前句开始" if kind == "start" else "当前句结束")
        )
        actual = segment.start_ms if kind == "start" else segment.end_ms
        self.status_stage.setText(f"已将{target}设为 {_time(actual)}")

    def _time_for_text_offset(
        self, source: TextSegment, text: str, text_offset: int, fallback_ms: int,
        *, row: int | None = None,
    ) -> int:
        """Map a textual split boundary through ASR/forced-aligner anchors."""
        if source.end_ms - source.start_ms < 2:
            return 0
        result = int(fallback_ms)
        if self.session and self.current_chapter_id is not None:
            if row is None:
                row = next((
                    index for index, segment in enumerate(self.segment_model.segments)
                    if segment is source or (source.id is not None and segment.id == source.id)
                ), -1)
            if not 0 <= row < len(self.segment_model.segments):
                return max(source.start_ms + 1, min(source.end_ms - 1, int(result)))
            # TextAudioAnchor offsets are chapter-concatenated coordinates.
            # TextSegment.source_* offsets belong to an EPUB/HTML source
            # fragment and often restart at zero for every paragraph.  Mixing
            # those coordinate systems selected anchors from unrelated cues and
            # pushed cursor splits to one end of the timing range.
            chapter_source_start = sum(
                len(segment.text) for segment in self.segment_model.segments[:row]
            )
            chapter_source_span = max(1, len(source.text))
            chapter_source_end = chapter_source_start + chapter_source_span
            mapped_character = chapter_source_start + round(
                chapter_source_span * text_offset / max(1, len(text))
            )
            anchors = [
                anchor for anchor in self.session.repository.anchors(self.current_chapter_id)
                if anchor.source_end_char >= chapter_source_start
                and anchor.source_start_char <= chapter_source_end
                and anchor.end_ms > anchor.start_ms
                and (
                    "asr" in anchor.method.casefold()
                    or "align" in anchor.method.casefold()
                    or "whisper" in anchor.method.casefold()
                )
            ]
            # Inclusive range queries can pick up the first word of the next
            # cue. Keep anchors whose character centre actually belongs to the
            # source segment.
            anchors = [
                anchor for anchor in anchors
                if chapter_source_start
                <= (anchor.source_start_char + anchor.source_end_char) / 2
                <= chapter_source_end
            ]
            containing = [
                anchor for anchor in anchors
                if anchor.source_start_char <= mapped_character <= anchor.source_end_char
            ]
            if containing:
                anchor = min(
                    containing,
                    key=lambda item: item.source_end_char - item.source_start_char,
                )
                ratio = (
                    (mapped_character - anchor.source_start_char)
                    / max(1, anchor.source_end_char - anchor.source_start_char)
                )
                result = round(anchor.start_ms + ratio * (anchor.end_ms - anchor.start_ms))
            elif anchors:
                before = [
                    anchor for anchor in anchors
                    if anchor.source_end_char <= mapped_character
                ]
                after = [
                    anchor for anchor in anchors
                    if anchor.source_start_char >= mapped_character
                ]
                previous = max(before, key=lambda item: item.source_end_char) if before else None
                following = min(after, key=lambda item: item.source_start_char) if after else None
                if previous and following:
                    # A sentence beginning is anchored by the first ASR word
                    # on its right. Keep the intervening pause with the prior
                    # cue and start the new cue at that actual word boundary.
                    result = following.start_ms
                elif previous:
                    result = previous.end_ms
                elif following:
                    result = following.start_ms

            if anchors and not source.start_ms < result < source.end_ms:
                # An older coarse segmentation can be shifted relative to the
                # word anchors. Preserve the ASR speaking-rate information by
                # projecting its local time span into the cue, rather than
                # clamping an out-of-range word time to cue_end-1 ms.
                anchor_start = min(anchor.start_ms for anchor in anchors)
                anchor_end = max(anchor.end_ms for anchor in anchors)
                if anchor_end > anchor_start:
                    ratio = max(0.0, min(1.0, (result - anchor_start) / (anchor_end - anchor_start)))
                    result = round(source.start_ms + ratio * (source.end_ms - source.start_ms))
        return max(source.start_ms + 1, min(source.end_ms - 1, int(result)))

    def split_segment(self) -> None:
        rows = self._selected_rows()
        if not rows or self.current_chapter_id is None:
            return
        row = rows[0]
        source = self.segment_model.segments[row]
        position = self._local_position()
        if source.locked or not source.start_ms < position < source.end_ms:
            return
        approximate = round(len(source.text) * (position - source.start_ms) / max(1, source.end_ms - source.start_ms))
        anchors = self.session.repository.anchors(self.current_chapter_id) if self.session else []
        source_anchors = [anchor for anchor in anchors if anchor.segment_id == source.id]
        if source_anchors:
            chapter_source_start = sum(len(item.text) for item in self.segment_model.segments[:row])
            nearest = min(
                source_anchors,
                key=lambda anchor: abs((anchor.start_ms + anchor.end_ms) // 2 - position),
            )
            span = max(1, nearest.end_ms - nearest.start_ms)
            ratio = max(0.0, min(1.0, (position - nearest.start_ms) / span))
            mapped = nearest.source_start_char + round(
                (nearest.source_end_char - nearest.source_start_char) * ratio
            )
            approximate = mapped - chapter_source_start
        text_offset = preferred_split_offset(source.text, approximate)
        if not 0 < text_offset < len(source.text):
            return
        left_text = source.text[:text_offset].rstrip()
        right_leading = len(source.text[text_offset:]) - len(source.text[text_offset:].lstrip())
        right_text = source.text[text_offset + right_leading:]
        if not left_text or not right_text:
            return
        source_start = source.source_start_char
        left_end = source_start + len(left_text) if source_start is not None else None
        right_start = source_start + text_offset + right_leading if source_start is not None else None
        self._push_history()
        replacement = [
            TextSegment(
                None, self.current_chapter_id, 0, left_text, source.start_ms, position,
                source.confidence, SegmentStatus.MANUAL, origin=source.origin,
                source_fragment_id=source.source_fragment_id,
                source_start_char=source_start, source_end_char=left_end,
            ),
            TextSegment(
                None, self.current_chapter_id, 0, right_text, position, source.end_ms,
                source.confidence, SegmentStatus.MANUAL, origin=source.origin,
                source_fragment_id=source.source_fragment_id,
                source_start_char=right_start, source_end_char=source.source_end_char,
            ),
        ]
        self._replace_one_segment(row, replacement)

    def split_segment_at_text_cursor(self) -> None:
        """Split the current cue at the explicit cursor in the fixed text editor."""
        if not self.session or self.current_chapter_id is None:
            return
        row = self._editing_row
        if not 0 <= row < len(self.segment_model.segments):
            row = self.segment_table.currentIndex().row()
        if not 0 <= row < len(self.segment_model.segments):
            self.status_stage.setText("请先选择一个句段并把光标放到文本框中")
            return
        source = self.segment_model.segments[row]
        if source.locked:
            self.status_stage.setText("当前句已锁定，请先解锁")
            return
        text = self.editor_text.toPlainText().replace("\r\n", "\n").replace("\r", "\n")
        requested_offset = (
            self._editor_cursor_position
            if self._editor_cursor_row == row
            else self.editor_text.textCursor().position()
        )
        text_offset = cursor_split_offset(text, requested_offset)
        if not text_offset:
            self.status_stage.setText("光标两侧都必须包含文字，不能在句首或句尾拆分")
            return

        raw_left = text[:text_offset]
        left_text = raw_left.rstrip()
        right_leading = len(text[text_offset:]) - len(text[text_offset:].lstrip())
        right_text = text[text_offset + right_leading:]
        if not left_text or not right_text:
            self.status_stage.setText("光标两侧都必须包含文字")
            return

        has_timing = source.end_ms - source.start_ms >= 2
        split_time = 0
        if has_timing:
            fallback_time = source.start_ms + round(
                (source.end_ms - source.start_ms) * text_offset / max(1, len(text))
            )
            split_time = self._time_for_text_offset(
                source, text, text_offset, fallback_time, row=row,
            )

        source_start = source.source_start_char
        source_end = source.source_end_char
        left_source_end = right_source_start = None
        if source_start is not None and source_end is not None:
            source_span = max(0, source_end - source_start)
            if source_span == len(text):
                left_source_end = source_start + len(left_text)
                right_source_start = source_start + text_offset + right_leading
            else:
                mapped = source_start + round(source_span * text_offset / max(1, len(text)))
                left_source_end = right_source_start = mapped

        self.editor_commit_timer.stop()
        if not self._editor_history_pushed:
            self._push_history()
        status = SegmentStatus.MANUAL if has_timing else SegmentStatus.UNMATCHED
        replacement = [
            TextSegment(
                None, self.current_chapter_id, 0, left_text,
                source.start_ms if has_timing else 0, split_time if has_timing else 0,
                source.confidence, status, origin=source.origin,
                source_fragment_id=source.source_fragment_id,
                source_start_char=source_start, source_end_char=left_source_end,
            ),
            TextSegment(
                None, self.current_chapter_id, 0, right_text,
                split_time if has_timing else 0, source.end_ms if has_timing else 0,
                source.confidence, status, origin=source.origin,
                source_fragment_id=source.source_fragment_id,
                source_start_char=right_source_start, source_end_char=source_end,
            ),
        ]
        self._replace_one_segment(row, replacement)
        self._select_segment_row(row + 1)
        self.editor_text.setFocus()
        self.status_stage.setText(f"已在文本光标处拆分为第 {row + 1}、{row + 2} 句")

    def split_segment_by_punctuation(self) -> None:
        rows = self._selected_rows()
        if not rows or self.current_chapter_id is None:
            self.status_stage.setText("请先选择一个需要按标点拆分的句段")
            return
        if len(rows) != 1:
            self.status_stage.setText("按标点拆分一次只处理一个句段，请先单选")
            return
        row = rows[0]
        source = self.segment_model.segments[row]
        if source.locked:
            self.status_stage.setText("当前句已锁定，请先解锁")
            return
        live_text = (
            self.editor_text.toPlainText().replace("\r\n", "\n").replace("\r", "\n")
            if row == self._editing_row else source.text
        )
        pieces = split_sentences_with_offsets(live_text)
        if len(pieces) < 2:
            self.status_stage.setText("当前文本没有找到可用的句末、逗号或换行拆分点")
            return
        duration = max(0, source.end_ms - source.start_ms)
        has_timing = duration >= len(pieces)
        replacement: list[TextSegment] = []
        previous_end = source.start_ms if has_timing else 0
        minimum_piece_ms = (
            max(1, min(200, duration // max(1, len(pieces) * 4)))
            if has_timing else 0
        )
        for index, (text, start, end) in enumerate(pieces):
            piece_start = previous_end if has_timing else 0
            if not has_timing:
                piece_end = 0
            elif index == len(pieces) - 1:
                piece_end = source.end_ms
            else:
                fallback_end = source.start_ms + round(
                    duration * end / max(1, len(live_text))
                )
                piece_end = self._time_for_text_offset(
                    source, live_text, end, fallback_end, row=row,
                )
                latest = source.end_ms - (len(pieces) - index - 1) * minimum_piece_ms
                piece_end = min(latest, max(piece_start + minimum_piece_ms, piece_end))
            previous_end = piece_end
            base = source.source_start_char
            source_span = (
                source.source_end_char - base
                if base is not None and source.source_end_char is not None else None
            )
            if base is None or source_span is None:
                piece_source_start = piece_source_end = None
            elif source_span == len(live_text):
                piece_source_start, piece_source_end = base + start, base + end
            else:
                piece_source_start = base + round(source_span * start / max(1, len(live_text)))
                piece_source_end = base + round(source_span * end / max(1, len(live_text)))
            replacement.append(TextSegment(
                None, self.current_chapter_id, 0, text, piece_start, piece_end,
                source.confidence,
                SegmentStatus.MANUAL if has_timing else SegmentStatus.UNMATCHED,
                origin=source.origin,
                source_fragment_id=source.source_fragment_id,
                source_start_char=piece_source_start,
                source_end_char=piece_source_end,
            ))
        self.editor_commit_timer.stop()
        if not self._editor_history_pushed:
            self._push_history()
        self._replace_one_segment(row, replacement)
        self._select_segment_row(row)
        self.status_stage.setText(f"已按标点拆分为 {len(replacement)} 句")

    def restore_source_fragment(self) -> None:
        rows = self._selected_rows()
        if not rows or not self.session or self.current_chapter_id is None:
            return
        selected = self.segment_model.segments[rows[0]]
        fragment_id = selected.source_fragment_id
        if fragment_id is None:
            self.status_stage.setText("当前句段没有可恢复的原始段落")
            return
        affected = [
            (index, segment) for index, segment in enumerate(self.segment_model.segments)
            if segment.source_fragment_id == fragment_id
        ]
        if any(segment.locked for _index, segment in affected):
            self.status_stage.setText("原始段落中包含锁定句，请先解锁")
            return
        fragment = next(
            (item for item in self.session.repository.source_fragments(self.current_chapter_id)
             if item.id == fragment_id),
            None,
        )
        if fragment is None:
            return
        anchors = self.session.repository.anchors(self.current_chapter_id)
        pieces: list[TextSegment] = []
        for text, start, end in split_sentences_with_offsets(fragment.text):
            absolute_start = fragment.source_start_char + start
            absolute_end = fragment.source_start_char + end
            matching = [
                anchor for anchor in anchors
                if anchor.source_end_char > absolute_start and anchor.source_start_char < absolute_end
            ]
            if matching:
                start_ms = min(anchor.start_ms for anchor in matching)
                end_ms = max(anchor.end_ms for anchor in matching)
                confidence = min(anchor.confidence for anchor in matching)
                status = SegmentStatus.AUTO
            else:
                start_ms = end_ms = 0
                confidence = 0.0
                status = SegmentStatus.UNMATCHED
            pieces.append(TextSegment(
                None, self.current_chapter_id, 0, text, start_ms, end_ms, confidence, status,
                origin=SegmentOrigin.SOURCE, source_fragment_id=fragment.id,
                source_start_char=absolute_start, source_end_char=absolute_end,
            ))
        first = affected[0][0]
        affected_rows = {index for index, _segment in affected}
        remaining = [
            segment for index, segment in enumerate(self.segment_model.segments)
            if index not in affected_rows
        ]
        insert_at = sum(1 for index in range(first) if index not in affected_rows)
        self._push_history()
        self._replace_segments(remaining[:insert_at] + pieces + remaining[insert_at:])
        self.status_stage.setText(f"已从原始段落恢复并重新生成 {len(pieces)} 句")

    def restore_source_chapter(self) -> None:
        if not self.session or self.current_chapter_id is None:
            return
        if any(segment.locked for segment in self.segment_model.segments):
            self.status_stage.setText("当前章节包含锁定句，请先解锁")
            return
        fragments = self.session.repository.source_fragments(self.current_chapter_id)
        if not fragments:
            self.status_stage.setText("当前章节没有可恢复的原始文本")
            return
        anchors = self.session.repository.anchors(self.current_chapter_id)
        restored: list[TextSegment] = []
        for fragment in fragments:
            for text, start, end in split_sentences_with_offsets(fragment.text):
                absolute_start = fragment.source_start_char + start
                absolute_end = fragment.source_start_char + end
                matching = [
                    anchor for anchor in anchors
                    if anchor.source_end_char > absolute_start and anchor.source_start_char < absolute_end
                ]
                start_ms = min((anchor.start_ms for anchor in matching), default=0)
                end_ms = max((anchor.end_ms for anchor in matching), default=0)
                confidence = min((anchor.confidence for anchor in matching), default=0.0)
                restored.append(TextSegment(
                    None, self.current_chapter_id, len(restored), text, start_ms, end_ms,
                    confidence, SegmentStatus.AUTO if matching else SegmentStatus.UNMATCHED,
                    origin=SegmentOrigin.SOURCE, source_fragment_id=fragment.id,
                    source_start_char=absolute_start, source_end_char=absolute_end,
                ))
        self._push_history()
        self._replace_segments(restored)
        self.status_stage.setText(f"已从原始文本恢复整章，共 {len(restored)} 句")

    def merge_segment(self, direction: int = 1) -> None:
        rows = self._selected_rows()
        if not rows or self.current_chapter_id is None:
            return
        if len(rows) > 1:
            self.merge_selected_segments()
            return
        first = rows[0]
        second = first + direction
        if not 0 <= second < len(self.segment_model.segments):
            return
        low, high = sorted((first, second))
        left, right = self.segment_model.segments[low], self.segment_model.segments[high]
        if left.locked or right.locked:
            return
        self._push_history()
        same_source = left.source_fragment_id is not None and left.source_fragment_id == right.source_fragment_id
        merged = TextSegment(
            None, self.current_chapter_id, 0, join_segment_text(left.text, right.text),
            min(left.start_ms, right.start_ms), max(left.end_ms, right.end_ms),
            min(left.confidence, right.confidence), SegmentStatus.MANUAL,
            origin=left.origin if same_source else SegmentOrigin.USER,
            source_fragment_id=left.source_fragment_id if same_source else None,
            source_start_char=left.source_start_char if same_source else None,
            source_end_char=right.source_end_char if same_source else None,
        )
        if left.id is not None and right.id is not None and self.session:
            self.session.repository.merge_adjacent_segments(
                self.current_chapter_id, left.id, right.id, merged,
            )
            self._reload_after_structural_edit()
        else:
            self._replace_segments(
                self.segment_model.segments[:low] + [merged]
                + self.segment_model.segments[high + 1:]
            )

    def merge_selected_segments(self) -> None:
        """Merge one contiguous multi-selection as one undoable transaction."""
        rows = self._selected_rows()
        if len(rows) < 2 or not self.session or self.current_chapter_id is None:
            self.status_stage.setText("请连续选择至少两个句段")
            return
        if rows != list(range(rows[0], rows[-1] + 1)):
            self.status_stage.setText("只能合并连续句段")
            return
        selected = [self.segment_model.segments[row] for row in rows]
        if any(segment.locked for segment in selected):
            self.status_stage.setText("所选范围包含锁定句，请先解锁")
            return
        self._commit_segment_editor()
        selected = [self.segment_model.segments[row] for row in rows]
        text = selected[0].text
        for segment in selected[1:]:
            text = join_segment_text(text, segment.text)
        timed = [segment for segment in selected if segment.end_ms > segment.start_ms]
        start_ms = min((segment.start_ms for segment in timed), default=0)
        end_ms = max((segment.end_ms for segment in timed), default=0)
        same_source = (
            selected[0].source_fragment_id is not None
            and all(
                segment.source_fragment_id == selected[0].source_fragment_id
                for segment in selected
            )
        )
        merged = TextSegment(
            None, self.current_chapter_id, rows[0], text,
            start_ms, end_ms, min(segment.confidence for segment in selected),
            SegmentStatus.MANUAL if timed else SegmentStatus.UNMATCHED,
            origin=selected[0].origin if same_source else SegmentOrigin.USER,
            source_fragment_id=selected[0].source_fragment_id if same_source else None,
            source_start_char=selected[0].source_start_char if same_source else None,
            source_end_char=selected[-1].source_end_char if same_source else None,
        )
        self._push_history()
        ids = [segment.id for segment in selected]
        if all(identifier is not None for identifier in ids):
            self.session.repository.merge_segment_range(
                self.current_chapter_id,
                [int(identifier) for identifier in ids],
                merged,
            )
            self._reload_after_structural_edit()
        else:
            self._replace_segments(
                self.segment_model.segments[:rows[0]]
                + [merged]
                + self.segment_model.segments[rows[-1] + 1:]
            )
        self._select_segment_row(rows[0])
        self.status_stage.setText(f"已合并连续 {len(rows)} 句")

    def delete_segments(self) -> None:
        rows = self._selected_rows()
        if not rows or not self.session or self.current_chapter_id is None:
            return
        editable = [row for row in rows if not self.segment_model.segments[row].locked]
        if not editable:
            return
        source_rows = [
            row for row in editable
            if self.segment_model.segments[row].origin in {
                SegmentOrigin.SOURCE, SegmentOrigin.SUBTITLE,
            }
        ]
        removable = [row for row in editable if row not in source_rows]
        self._push_history()
        source_ids = [
            self.segment_model.segments[row].id for row in source_rows
            if self.segment_model.segments[row].id is not None
        ]
        delete_ids = [
            self.segment_model.segments[row].id for row in removable
            if self.segment_model.segments[row].id is not None
        ]
        self.session.repository.mark_segments_unmatched(source_ids)
        self.session.repository.delete_segments(self.current_chapter_id, delete_ids)
        self.segment_model.set_segments(self.session.repository.segments(self.current_chapter_id))
        self._refresh_segment_views()
        self.status_stage.setText(
            f"已保留 {len(source_rows)} 条原文为未匹配，删除 {len(removable)} 条新建句段"
        )
        self._mark_dirty()

    def clear_timing(self) -> None:
        self._clear_segment_matches(self._selected_rows())

    def clear_chapter_timing(self) -> None:
        if not self.segment_model.segments or self.current_chapter_id is None:
            return
        locked = sum(1 for segment in self.segment_model.segments if segment.locked)
        if locked:
            answer = QMessageBox.question(
                self,
                "清除本章时间对应",
                f"本章有 {locked} 个锁定句段；它们会保持不变。是否清除其余句段的时间对应？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._clear_segment_matches(list(range(len(self.segment_model.segments))))

    def reset_chapter_and_rematch(self) -> None:
        if (
            not self.session or self.current_chapter_id is None
            or not self.current_parts
        ):
            self.status_stage.setText("当前章节没有可用于重新匹配的媒体")
            return
        locked = [segment for segment in self.segment_model.segments if segment.locked]
        if locked:
            QMessageBox.information(
                self,
                "不能重置本章",
                f"本章还有 {len(locked)} 个锁定句段。请先解锁，避免重置操作绕过人工校正。",
            )
            return
        protected = [
            segment for segment in self.segment_model.segments
            if segment.origin in {SegmentOrigin.SOURCE, SegmentOrigin.SUBTITLE}
        ]
        generated = [
            segment for segment in self.segment_model.segments
            if segment.origin not in {SegmentOrigin.SOURCE, SegmentOrigin.SUBTITLE}
        ]
        answer = QMessageBox.question(
            self,
            "重置本章字幕并重新匹配",
            f"将保留 {len(protected)} 条原书/SRT 文本并清除其时间，"
            f"删除 {len(generated)} 条 ASR 或用户新建句段。\n\n"
            "随后优先复用识别缓存重新匹配；缓存不存在时才运行当前 ASR 模型。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._push_history()
        protected_ids = [segment.id for segment in protected if segment.id is not None]
        generated_ids = [segment.id for segment in generated if segment.id is not None]
        self.session.repository.mark_segments_unmatched(protected_ids)
        self.session.repository.delete_segments(self.current_chapter_id, generated_ids)
        self.segment_model.set_segments(
            self.session.repository.segments(self.current_chapter_id)
        )
        self._clear_play_range()
        self._refresh_segment_views(defer_hidden=True, refresh_audio=True)
        self._mark_dirty()
        self.status_stage.setText("本章字幕已重置，正在读取识别缓存并重新匹配…")
        QTimer.singleShot(0, lambda: self.recognize_current(force=False))

    def _clear_segment_matches(self, rows: list[int]) -> None:
        if not rows:
            return
        editable = [row for row in rows if not self.segment_model.segments[row].locked]
        if not editable or not self.session:
            return
        self._push_history()
        ids = []
        for row in editable:
            segment = self.segment_model.segments[row]
            segment.start_ms = segment.end_ms = 0
            segment.confidence = 0.0
            segment.status = SegmentStatus.UNMATCHED
            if segment.id is not None:
                ids.append(segment.id)
        self.session.repository.mark_segments_unmatched(ids)
        self._refresh_segment_views(defer_hidden=True, refresh_audio=True)
        first = self.segment_model.index(min(editable), 0)
        last = self.segment_model.index(max(editable), 3)
        self.segment_model.dataChanged.emit(first, last)
        self.status_stage.setText(f"已移除 {len(editable)} 个时间匹配，原文已保留为未匹配")
        self._mark_dirty()

    def _refresh_segment_views(
        self, *, defer_hidden: bool = True, refresh_audio: bool = True,
    ) -> None:
        if refresh_audio:
            self.spectrogram.set_segments(self.segment_model.segments)
            self.overview.set_segments(self.segment_model.segments)
        if not defer_hidden or self.content_tabs.currentWidget() is self.article_view:
            anchors = (
                self.session.repository.anchors(self.current_chapter_id or 0)
                if self.session else []
            )
            self.article_view.set_content(self.segment_model.segments, anchors, self.current_cache)
            self._article_content_dirty = False
        else:
            self._article_content_dirty = True
        if not defer_hidden or self.content_tabs.currentWidget() is self.asr_comparison:
            tokens = (
                self.session.repository.asr_tokens(self.current_chapter_id or 0)
                if self.session else []
            )
            comparison_anchors = (
                self.session.repository.anchors(self.current_chapter_id or 0)
                if self.session else []
            )
            self.asr_comparison.set_content(
                self.segment_model.segments, tokens, comparison_anchors,
            )
            self._asr_content_dirty = False
        else:
            self._asr_content_dirty = True
        if not defer_hidden or self.content_tabs.currentWidget() is self.original_book_view:
            self.original_book_refresh_timer.start()
            self._original_book_content_dirty = False
        else:
            self._original_book_content_dirty = True

    def _content_tab_changed(self, _index: int) -> None:
        if not self.session or self.current_chapter_id is None:
            return
        current = self.content_tabs.currentWidget()
        if current is self.article_view and self._article_content_dirty:
            self.article_view.set_content(
                self.segment_model.segments,
                self.session.repository.anchors(self.current_chapter_id),
                self.current_cache,
            )
            self._article_content_dirty = False
        elif current is self.asr_comparison and self._asr_content_dirty:
            self.asr_comparison.set_content(
                self.segment_model.segments,
                self.session.repository.asr_tokens(self.current_chapter_id),
                self.session.repository.anchors(self.current_chapter_id),
            )
            self._asr_content_dirty = False
        elif current is self.original_book_view and self._original_book_content_dirty:
            self._refresh_original_book()
            self._original_book_content_dirty = False
        row = self.segment_table.currentIndex().row()
        if 0 <= row < len(self.segment_model.segments):
            self._focus_segment_in_active_content_view(row)

    def _refresh_original_book(self) -> None:
        if not self.session or self.current_chapter_id is None:
            return
        if self.content_tabs.currentWidget() is not self.original_book_view:
            self._original_book_content_dirty = True
            return
        self.original_book_view.set_chapter(
            self.session, self.current_chapter_id,
            self.segment_model.segments, self._session_generation,
        )
        self._original_book_content_dirty = False

    def _original_book_segment_activated(
        self, segment_id: int, generation: int, jump: bool = False,
    ) -> None:
        if generation != self._session_generation:
            return
        row = next((
            index for index, segment in enumerate(self.segment_model.segments)
            if segment.id == segment_id
        ), -1)
        if row < 0:
            return
        segment = self.segment_model.segments[row]
        if segment.end_ms > segment.start_ms:
            self._audio_time_activated(segment.start_ms, row)
            if jump:
                self.spectrogram.suspend_follow("original-book")
                span = max(500, self.spectrogram.view_end - self.spectrogram.view_start)
                self.spectrogram.focus_time(segment.start_ms, span)
        else:
            self._select_segment_row(row)

    def toggle_lock(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        self._push_history()
        lock = not all(self.segment_model.segments[row].locked for row in rows)
        for row in rows:
            segment = self.segment_model.segments[row]
            segment.locked = lock
            segment.status = SegmentStatus.LOCKED if lock else SegmentStatus.MANUAL
        self._persist_current_segments(rows)
        current = self.segment_table.currentIndex().row()
        if current >= 0:
            self._load_segment_editor(current)

    def _reload_after_structural_edit(self) -> None:
        """Reload one changed cue structure without rewriting chapter data."""
        if not self.session or self.current_chapter_id is None:
            return
        self.segment_model.set_segments(
            self.session.repository.segments(self.current_chapter_id)
        )
        self._refresh_segment_views(defer_hidden=True, refresh_audio=True)
        self._refresh_segment_play_range()
        self._mark_dirty()

    def _replace_one_segment(
        self, row: int, replacements: list[TextSegment],
    ) -> None:
        """Use the local database path for interactive split operations."""
        if not self.session or self.current_chapter_id is None:
            return
        original = self.segment_model.segments[row]
        if original.id is None:
            self._replace_segments(
                self.segment_model.segments[:row] + replacements
                + self.segment_model.segments[row + 1:]
            )
            return
        self.session.repository.replace_one_segment(
            self.current_chapter_id, original.id, replacements,
        )
        self._reload_after_structural_edit()

    def _replace_segments(
        self, segments: list[TextSegment], anchors: list[TextAudioAnchor] | None = None,
    ) -> None:
        if not self.session or self.current_chapter_id is None:
            return
        chapter_id = self.current_chapter_id
        for index, segment in enumerate(segments):
            segment.position = index
            if segment.chapter_id != chapter_id:
                raise ValueError("Refusing to write a segment snapshot from another chapter")
        current = self.segment_model.segments
        same_structure = (
            len(current) == len(segments)
            and all(
                left.id is not None and left.id == right.id
                for left, right in zip(current, segments)
            )
        )
        changed_rows = [
            index for index, (left, right) in enumerate(zip(current, segments))
            if left != right
        ] if same_structure else []
        if anchors is None:
            anchors = self.session.repository.anchors(chapter_id)
        self.session.repository.replace_chapter_edit_state(chapter_id, segments, anchors)
        self.segment_model.set_segments(self.session.repository.segments(chapter_id))
        if same_structure:
            self.spectrogram.update_segments(self.segment_model.segments, changed_rows)
            self.overview.set_segments(self.segment_model.segments)
        self._refresh_segment_views(
            defer_hidden=True, refresh_audio=not same_structure,
        )
        self._refresh_segment_play_range()
        self._mark_dirty()

    def _persist_current_segments(self, changed_rows: list[int] | None = None) -> None:
        """Persist non-structural edits without rebuilding the whole chapter."""
        if not self.session or self.current_chapter_id is None:
            return
        rows = sorted(set(
            range(len(self.segment_model.segments)) if changed_rows is None else changed_rows
        ))
        rows = [row for row in rows if 0 <= row < len(self.segment_model.segments)]
        if not rows:
            return
        self.session.repository.update_segments(
            [self.segment_model.segments[row] for row in rows]
        )
        first = self.segment_model.index(max(0, rows[0] - 1), 0)
        last = self.segment_model.index(
            min(self.segment_model.rowCount() - 1, rows[-1] + 1),
            self.segment_model.columnCount() - 1,
        )
        self.segment_model.dataChanged.emit(first, last)
        self.spectrogram.update_segments(self.segment_model.segments, rows)
        self.overview.set_segments(self.segment_model.segments)
        # Timing/status edits do not change either side of the ASR comparison
        # or the original book text. Rebuilding their large HTML documents made
        # a simple bind take several seconds on audiobook-sized chapters.
        # The article audio rows do depend on timing, so rebuild them lazily.
        self._article_content_dirty = True
        self._refresh_segment_play_range()
        self._mark_dirty()

    def _segment_edited(self, segment: TextSegment) -> None:
        if self.session:
            self.session.repository.update_segment(segment)
            self._refresh_segment_views(defer_hidden=True, refresh_audio=True)
            self._refresh_segment_play_range()
            self._mark_dirty()

    def _push_history(self) -> None:
        context = self._project_context()
        chapter_id = self.current_chapter_id
        if not context or chapter_id is None or not self.session:
            return
        snapshot = ChapterEditSnapshot(
            context,
            chapter_id,
            copy.deepcopy(self.segment_model.segments),
            copy.deepcopy(self.session.repository.anchors(chapter_id)),
        )
        history = self._history.setdefault(chapter_id, [])
        history.append(snapshot)
        del history[:-100]
        self._future.setdefault(chapter_id, []).clear()
        self._update_undo_actions()

    def _current_edit_snapshot(self) -> ChapterEditSnapshot | None:
        context = self._project_context()
        chapter_id = self.current_chapter_id
        if not context or chapter_id is None or not self.session:
            return None
        return ChapterEditSnapshot(
            context, chapter_id, copy.deepcopy(self.segment_model.segments),
            copy.deepcopy(self.session.repository.anchors(chapter_id)),
        )

    def _snapshot_matches_current(self, snapshot: ChapterEditSnapshot) -> bool:
        return snapshot.context == self._project_context() and snapshot.chapter_id == self.current_chapter_id

    def _update_undo_actions(self) -> None:
        chapter_id = self.current_chapter_id
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(bool(chapter_id is not None and self._history.get(chapter_id)))
            self.redo_action.setEnabled(bool(chapter_id is not None and self._future.get(chapter_id)))

    def undo(self) -> None:
        chapter_id = self.current_chapter_id
        history = self._history.get(chapter_id or -1, [])
        current = self._current_edit_snapshot()
        if not history or current is None:
            return
        target = history[-1]
        if not self._snapshot_matches_current(target):
            history.clear()
            self._update_undo_actions()
            self.status_stage.setText("撤销记录不属于当前项目或章节，已安全丢弃")
            return
        history.pop()
        self._future.setdefault(chapter_id or 0, []).append(current)
        self._replace_segments(copy.deepcopy(target.segments), copy.deepcopy(target.anchors))
        self._update_undo_actions()

    def redo(self) -> None:
        chapter_id = self.current_chapter_id
        future = self._future.get(chapter_id or -1, [])
        current = self._current_edit_snapshot()
        if not future or current is None:
            return
        target = future[-1]
        if not self._snapshot_matches_current(target):
            future.clear()
            self._update_undo_actions()
            self.status_stage.setText("重做记录不属于当前项目或章节，已安全丢弃")
            return
        future.pop()
        self._history.setdefault(chapter_id or 0, []).append(current)
        self._replace_segments(copy.deepcopy(target.segments), copy.deepcopy(target.anchors))
        self._update_undo_actions()

    def _segment_selected(self, current, _previous) -> None:
        if not current.isValid():
            return
        row = current.row()
        if not self._playback_row_update:
            self._clear_play_range()
            self._manual_selection_until = time.monotonic() + 0.35
        self._focus_segment_in_active_content_view(row)
        self._load_segment_editor(row)

    def _focus_segment_in_active_content_view(self, row: int) -> None:
        """Synchronise only the visible document view; hidden views stay idle."""
        if not (0 <= row < len(self.segment_model.segments)):
            return
        current = self.content_tabs.currentWidget()
        if current is self.article_view:
            self.article_view.focus_segment(row, ensure_visible=True)
        elif current is self.original_book_view:
            self.original_book_view.focus_segment(
                self.segment_model.segments[row].id, ensure_visible=True,
            )
        elif current is self.asr_comparison:
            self.asr_comparison.focus_segment(row, ensure_visible=True)

    def _table_selection_changed(self, _selected=None, _deselected=None) -> None:
        rows = {
            index.row()
            for index in self.segment_table.selectionModel().selectedRows()
        }
        self.spectrogram.set_selected_segments(rows)

    def _audio_segment_selected(self, row: int, modifiers: int) -> None:
        """Mirror Ctrl/Shift cue selection into the table without collapsing it."""
        if not (0 <= row < self.segment_model.rowCount()):
            return
        selection_model = self.segment_table.selectionModel()
        index = self.segment_model.index(row, 0)
        control = bool(modifiers & int(Qt.KeyboardModifier.ControlModifier.value))
        shift = bool(modifiers & int(Qt.KeyboardModifier.ShiftModifier.value))
        if shift:
            current = self.segment_table.currentIndex()
            anchor = current.row() if current.isValid() else row
            selection_model.clearSelection()
            for selected_row in range(min(anchor, row), max(anchor, row) + 1):
                selection_model.select(
                    self.segment_model.index(selected_row, 0),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            selection_model.setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        elif control:
            selection_model.select(
                index,
                QItemSelectionModel.SelectionFlag.Toggle
                | QItemSelectionModel.SelectionFlag.Rows,
            )
            selection_model.setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        else:
            self._select_only_segment_row(row)

    def _load_segment_editor(self, row: int) -> None:
        self.editor_commit_timer.stop()
        self._editing_row = row
        self._editor_history_pushed = False
        if not (0 <= row < len(self.segment_model.segments)):
            self._editor_loading = True
            self.editor_info.setText("未选择句段")
            self.editor_start.setText("00:00:00.000")
            self.editor_end.setText("00:00:00.000")
            self.editor_duration.setText("00:00:00.000")
            self.editor_text.clear()
            self._editor_loading = False
            return
        segment = self.segment_model.segments[row]
        self._editor_loading = True
        try:
            self.editor_info.setText(
                f"#{row + 1} · {segment.status.value} · 置信度 {segment.confidence:.0%}"
            )
            self.editor_start.setText(_time(segment.start_ms))
            self.editor_end.setText(_time(segment.end_ms))
            self.editor_duration.setText(_time(max(0, segment.end_ms - segment.start_ms)))
            self.editor_text.setPlainText(segment.text)
            read_only = segment.locked
            for field in (self.editor_start, self.editor_end, self.editor_duration):
                field.setReadOnly(read_only)
            self.editor_text.setReadOnly(read_only)
            self.editor_lock.setText("🔒" if read_only else "🔓")
            self._last_editor_values = (segment.start_ms, segment.end_ms, segment.end_ms - segment.start_ms)
        finally:
            self._editor_loading = False
        self._editor_cursor_row = row
        self._editor_cursor_position = self.editor_text.textCursor().position()

    @staticmethod
    def _parse_timecode(value: str) -> int:
        text = value.strip().replace(",", ".")
        pieces = text.split(":")
        if len(pieces) != 3:
            raise ValueError("时间格式应为 HH:MM:SS.mmm")
        hours, minutes = int(pieces[0]), int(pieces[1])
        seconds = float(pieces[2])
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError("时间值超出范围")
        return round((hours * 3600 + minutes * 60 + seconds) * 1000)

    def _segment_editor_changed(self) -> None:
        if self._editor_loading:
            return
        if not self._editor_history_pushed:
            self._push_history()
            self._editor_history_pushed = True
        self.editor_commit_timer.start()

    def _segment_editor_cursor_changed(self) -> None:
        if self._editor_loading or self._editing_row < 0:
            return
        self._editor_cursor_row = self._editing_row
        self._editor_cursor_position = self.editor_text.textCursor().position()

    def _commit_segment_editor(self) -> None:
        if self._editor_loading or not (0 <= self._editing_row < len(self.segment_model.segments)):
            return
        segment = self.segment_model.segments[self._editing_row]
        if segment.locked or not self.session:
            return
        try:
            start = self._parse_timecode(self.editor_start.text())
            end = self._parse_timecode(self.editor_end.text())
            duration = self._parse_timecode(self.editor_duration.text())
        except ValueError as exc:
            self.status_stage.setText(str(exc))
            self._load_segment_editor(self._editing_row)
            return
        old_start, old_end, old_duration = getattr(
            self, "_last_editor_values", (segment.start_ms, segment.end_ms, segment.end_ms - segment.start_ms)
        )
        if duration != old_duration and start == old_start and end == old_end:
            end = start + duration
        if end < start:
            self.status_stage.setText("结束时间不能早于开始时间")
            self._load_segment_editor(self._editing_row)
            return
        edited_text = self.editor_text.toPlainText().replace("\r\n", "\n").replace("\r", "\n")
        if (start, end, edited_text) == (segment.start_ms, segment.end_ms, segment.text):
            self._last_editor_values = (start, end, end - start)
            self._editor_history_pushed = False
            return
        if not self._editor_history_pushed:
            self._push_history()
            self._editor_history_pushed = True
        segment.start_ms, segment.end_ms = start, end
        segment.text = edited_text
        segment.status = SegmentStatus.MANUAL
        self.session.repository.update_segment(segment)
        index = self.segment_model.index(self._editing_row, 0)
        self.segment_model.dataChanged.emit(index, self.segment_model.index(self._editing_row, 3))
        self._refresh_segment_views(defer_hidden=True, refresh_audio=True)
        self._last_editor_values = (start, end, end - start)
        self._editor_loading = True
        self.editor_end.setText(_time(end))
        self.editor_duration.setText(_time(end - start))
        self._editor_loading = False
        self._refresh_segment_play_range()
        self._mark_dirty()

    def _select_segment_row(self, row: int) -> None:
        self._select_only_segment_row(row)

    def _select_only_segment_row(self, row: int) -> None:
        if 0 <= row < self.segment_model.rowCount():
            index = self.segment_model.index(row, 0)
            self.segment_table.selectionModel().setCurrentIndex(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )

    def _select_segment_rows(self, rows: list[int], *, current_row: int | None = None) -> None:
        valid_rows = sorted({
            row for row in rows if 0 <= row < self.segment_model.rowCount()
        })
        if not valid_rows:
            return
        selection_model = self.segment_table.selectionModel()
        selection_model.clearSelection()
        for row in valid_rows:
            selection_model.select(
                self.segment_model.index(row, 0),
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        row = current_row if current_row in valid_rows else valid_rows[0]
        selection_model.setCurrentIndex(
            self.segment_model.index(row, 0),
            QItemSelectionModel.SelectionFlag.NoUpdate,
        )

    def _article_range_selected(self, start: int, end: int) -> None:
        self.spectrogram.set_selection(start, end)

    def _article_range_finished(self, start: int, end: int) -> None:
        self.spectrogram.focus_time((start + end) // 2, max(5000, (end - start) * 3))

    def _article_seek(self, milliseconds: int) -> None:
        self._audio_seek_requested(milliseconds)
        self.spectrogram.suspend_follow("article")
        self.spectrogram.focus_time(milliseconds, self.spectrogram.view_end - self.spectrogram.view_start)

    def _asr_word_seek(self, milliseconds: int) -> None:
        """A single click only moves the playback cursor."""
        self._clear_play_range()
        self._seek_local(milliseconds)

    def _asr_word_jump(self, milliseconds: int) -> None:
        """A double click locates and centres the word in the audio view."""
        self._asr_word_seek(milliseconds)
        self.spectrogram.suspend_follow("asr-comparison")
        span = max(500, self.spectrogram.view_end - self.spectrogram.view_start)
        self.spectrogram.focus_time(milliseconds, span)

    def _clear_play_range(self) -> None:
        self._play_range_start = None
        self._play_range_end = None
        self._play_range_segment_id = None
        self._play_range_segment_row = None

    def _set_segment_play_range(self, row: int) -> None:
        if not (0 <= row < len(self.segment_model.segments)):
            self._clear_play_range()
            return
        segment = self.segment_model.segments[row]
        self._play_range_start = None
        self._play_range_end = None
        self._play_range_segment_id = segment.id
        self._play_range_segment_row = row

    def _active_segment_play_row(self) -> int:
        if self._play_range_segment_id is not None:
            for row, segment in enumerate(self.segment_model.segments):
                if segment.id == self._play_range_segment_id:
                    self._play_range_segment_row = row
                    return row
            return -1
        row = self._play_range_segment_row
        return row if row is not None and 0 <= row < len(self.segment_model.segments) else -1

    def _active_play_bounds(self) -> tuple[int, int] | None:
        row = self._active_segment_play_row()
        if row >= 0:
            segment = self.segment_model.segments[row]
            if segment.end_ms > segment.start_ms:
                return segment.start_ms, segment.end_ms
            return None
        if self._play_range_start is not None and self._play_range_end is not None:
            return self._play_range_start, self._play_range_end
        return None

    def _refresh_segment_play_range(self) -> None:
        """Keep the displayed cue range synchronized with live segment timing."""
        row = self._active_segment_play_row()
        if row < 0:
            return
        segment = self.segment_model.segments[row]
        if segment.end_ms > segment.start_ms:
            self.spectrogram.set_segment_selection(segment.start_ms, segment.end_ms)
        else:
            self.spectrogram.clear_selection()

    def _audio_seek_requested(self, milliseconds: int) -> None:
        self._clear_play_range()
        self._seek_local(milliseconds)

    def _audio_time_activated(self, milliseconds: int, row: int, _modifiers: int = 0) -> None:
        """A click always seeks first and only then changes the current cue."""
        # Qt sends the first click of a double-click through this handler. Some
        # multimedia backends briefly report Paused while processing its seek,
        # so remember the state before that seek for the double-click handler.
        self._recent_audio_click_at = time.monotonic()
        self._recent_audio_click_row = row
        self._recent_audio_click_was_playing = (
            self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        self._clear_play_range()
        self._manual_selection_until = time.monotonic() + 0.35
        self._seek_local(milliseconds)
        # Cue selection follows in segmentSelected, after the seek, where
        # Ctrl/Shift modifiers can be applied without briefly collapsing a
        # multi-selection to one row.

    def _segment_double_activated(self, row: int, milliseconds: int | None = None) -> None:
        """Double-click selects; it only keeps playing when playback was already active."""
        if not (0 <= row < len(self.segment_model.segments)):
            return
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if (
            milliseconds is not None
            and self._recent_audio_click_row == row
            and time.monotonic() - self._recent_audio_click_at
            <= max(0.5, QApplication.doubleClickInterval() / 1000 + 0.15)
        ):
            was_playing = was_playing or self._recent_audio_click_was_playing
        self._recent_audio_click_at = 0.0
        self._recent_audio_click_row = -1
        self._recent_audio_click_was_playing = False
        segment = self.segment_model.segments[row]
        target = segment.start_ms if milliseconds is None else max(segment.start_ms, min(segment.end_ms, milliseconds))
        self._clear_play_range()
        self._manual_selection_until = time.monotonic() + 0.35
        self._seek_local(target)
        self._select_segment_row(row)
        self.spectrogram.select_segment(row)
        self.spectrogram.set_segment_selection(segment.start_ms, segment.end_ms)
        self._locate_segment_start_if_outside(segment)
        if was_playing and segment.end_ms > target:
            self._set_segment_play_range(row)
            self.spectrogram.restore_follow()
            self._resume_after_seek()

    def _locate_segment_start_if_outside(self, segment: TextSegment) -> None:
        """Explicit sentence activation locates only an off-screen cue start."""
        if self.spectrogram.view_start <= segment.start_ms <= self.spectrogram.view_end:
            return
        span = max(500, self.spectrogram.view_end - self.spectrogram.view_start)
        self.spectrogram.focus_time(segment.start_ms, span)

    def _move_row(self, amount: int) -> None:
        row = self.segment_table.currentIndex().row() if self.segment_table.currentIndex().isValid() else 0
        self._select_segment_row(max(0, min(self.segment_model.rowCount() - 1, row + amount)))

    def _seek_local(self, milliseconds: int) -> None:
        milliseconds = max(0, int(milliseconds))
        if self.current_parts:
            milliseconds = min(milliseconds, self.current_parts[-1][4])
        self._logical_playhead_ms = milliseconds
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._audio_watch_started_at = time.monotonic()
            self._last_audio_buffer_at = 0.0
        self._seek_generation += 1
        self._pending_seek_target = milliseconds
        self._pending_seek_deadline = time.monotonic() + 2.0
        self.spectrogram.set_playhead(milliseconds)
        self.overview.set_playhead(milliseconds)
        for index, (link, asset, path, local_start, local_end) in enumerate(self.current_parts):
            is_last = index == len(self.current_parts) - 1
            if local_start <= milliseconds < local_end or (is_last and milliseconds == local_end):
                was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                self.current_link, self.current_asset = link, asset
                absolute = link.source_start_ms + milliseconds - local_start
                self._request_media_position(asset, path, absolute, autoplay=was_playing)
                return
        offset = self.current_link.source_start_ms if self.current_link else 0
        if self.current_asset and self.current_link:
            current_path = next(
                (path for link, _asset, path, _start, _end in self.current_parts if link is self.current_link),
                None,
            )
            if current_path:
                self._request_media_position(
                    self.current_asset, current_path, offset + milliseconds,
                    autoplay=self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState,
                )

    def _request_media_position(
        self, asset: AudioAsset, path: Path, absolute_ms: int, *, autoplay: bool,
        force_source: bool = False,
    ) -> None:
        """Load an asset and apply its absolute seek only when Qt is ready."""
        target = max(0, int(absolute_ms))
        # Windows Media Foundation can acknowledge a deep M4A/M4B seek and
        # then silently jump back to the old position when it was issued while
        # Playing. Preserve the intent, pause first, seek, then resume.
        autoplay = bool(
            autoplay
            or self._pending_media_autoplay
            or self._playback_requested
            or self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        if autoplay:
            self._playback_requested = True
        self._pending_media_position = target
        self._pending_media_autoplay = autoplay
        self._pending_media_deadline = time.monotonic() + 8.0
        playback_path = path
        if (
            not force_source
            and self.session
            and asset.id is not None
            and path.suffix.lower() == ".m4b"
        ):
            proxy = self.session.root / "cache" / f"playback-{asset.id}.m4a"
            if proxy.is_file() and proxy.stat().st_size > 0:
                playback_path = proxy
                self._proxy_attempted.add(asset.id)
        source_loaded = not self.player.source().isEmpty()
        same_asset = asset.id is not None and self._player_asset_id == asset.id and source_loaded
        if force_source or not same_asset:
            self._player_asset_id = asset.id
            self._last_audio_buffer_at = 0.0
            self._audio_watch_started_at = time.monotonic() if autoplay else 0.0
            self.player.setSource(QUrl.fromLocalFile(str(playback_path)))
        self._apply_pending_media_position()

    def _apply_pending_media_position(self) -> None:
        if self._pending_media_position is None:
            return
        if self.player.mediaStatus() in {
            QMediaPlayer.MediaStatus.NoMedia,
            QMediaPlayer.MediaStatus.LoadingMedia,
            QMediaPlayer.MediaStatus.InvalidMedia,
        }:
            return
        resume = self._pending_media_autoplay and self._playback_requested
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        self._pending_media_deadline = time.monotonic() + 3.0
        self.player.setPosition(self._pending_media_position)
        if resume:
            self._pending_media_autoplay = False
            self.player.play()

    def _media_status_changed(self, _status) -> None:
        self._apply_pending_media_position()

    def _resume_after_seek(self) -> None:
        self._playback_requested = True
        if self._pending_media_position is not None:
            self._pending_media_autoplay = True
            self._apply_pending_media_position()
        else:
            self.player.play()

    def seek_relative(self, amount_ms: int) -> None:
        self._clear_play_range()
        self._seek_local(max(0, self._local_position() + amount_ms))

    def _local_position(self) -> int:
        if (
            self._pending_media_position is not None
            or (
                self._pending_seek_target is not None
                and time.monotonic() < self._pending_seek_deadline
            )
        ):
            return self._logical_playhead_ms
        for link, _asset, _path, local_start, local_end in self.current_parts:
            if link is self.current_link:
                absolute = self.player.position()
                # Loading and failed deep seeks can report resource position 0
                # even though the editor has a valid chapter-local playhead.
                if absolute + 250 < link.source_start_ms or absolute > link.source_end_ms + 250:
                    return self._logical_playhead_ms
                local = max(local_start, min(local_end, local_start + absolute - link.source_start_ms))
                self._logical_playhead_ms = local
                return local
        return self._logical_playhead_ms

    def toggle_play(self) -> None:
        if (
            self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            or self._playback_requested
        ):
            self._playback_requested = False
            self._pending_media_autoplay = False
            self.player.pause()
            self.play_button.setText("▶")
            self.play_button.setToolTip("播放（Space）")
        else:
            self._clear_play_range()
            if (
                self.current_parts
                and self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia
            ):
                self._seek_local(0)
            self._resume_after_seek()

    def play_selection(self) -> None:
        if self.spectrogram.selection:
            self._play_range_segment_id = None
            self._play_range_segment_row = None
            self._play_range_start = self.spectrogram.selection.start_ms
            self._play_range_end = self.spectrogram.selection.end_ms
            self._seek_local(self._play_range_start)
            self.spectrogram.restore_follow()
            self._resume_after_seek()

    def play_segment(self, row: int) -> None:
        if not (0 <= row < len(self.segment_model.segments)):
            return
        segment = self.segment_model.segments[row]
        if segment.end_ms <= segment.start_ms:
            self.status_stage.setText("当前句没有有效的音频时间")
            return
        self._select_segment_row(row)
        self.spectrogram.set_segment_selection(segment.start_ms, segment.end_ms)
        self.spectrogram.restore_follow()
        self._seek_local(segment.start_ms)
        self._set_segment_play_range(row)
        self._resume_after_seek()

    def play_current_segment(self) -> None:
        row = self.segment_table.currentIndex().row() if self.segment_table.currentIndex().isValid() else -1
        self.play_segment(row)

    def toggle_current_loop(self, enabled: bool) -> None:
        self._loop_selection = enabled
        if enabled:
            row = self.segment_table.currentIndex().row() if self.segment_table.currentIndex().isValid() else -1
            if 0 <= row < len(self.segment_model.segments):
                segment = self.segment_model.segments[row]
                self.spectrogram.set_segment_selection(segment.start_ms, segment.end_ms)
                self.play_segment(row)

    def _set_loop(self, enabled: bool) -> None:
        self._loop_selection = enabled

    def set_end_and_next(self) -> None:
        self.set_boundary("end")
        self._move_row(1)

    def set_visualization_mode(self, mode: AudioVisualizationMode | str) -> None:
        mode = AudioVisualizationMode(mode)
        actions = {
            AudioVisualizationMode.WAVEFORM: self.waveform_action,
            AudioVisualizationMode.SPECTROGRAM: self.spectrum_action,
            AudioVisualizationMode.COMBINED: self.combined_action,
        }
        actions[mode].setChecked(True)
        self.spectrogram.set_mode(mode)
        self.overview.set_mode(mode)
        self.preferences["visualization_mode"] = mode.value

    def _set_list_audio_visible(self, visible: bool = True) -> None:
        visible = bool(visible)
        if hasattr(self, "segment_table"):
            self.segment_table.setColumnHidden(2, not visible)
        self.preferences["show_list_audio_visual"] = visible

    def _set_article_audio_visible(self, visible: bool = True) -> None:
        visible = bool(visible)
        if hasattr(self, "article_view"):
            self.article_view.set_audio_visible(visible)
        self.preferences["show_article_audio_visual"] = visible

    def _set_silence_display_mode(self, mode: SilenceDisplayMode | str) -> None:
        mode = SilenceDisplayMode(mode)
        actions = {
            SilenceDisplayMode.HIDDEN: self.silence_hidden_action,
            SilenceDisplayMode.KEY: self.silence_key_action,
            SilenceDisplayMode.ALL: self.silence_all_action,
        }
        actions[mode].setChecked(True)
        self.preferences["silence_display_mode"] = mode.value
        self._refresh_silence_overlays(mode)

    def _refresh_silence_overlays(self, mode: SilenceDisplayMode | None = None) -> None:
        if mode is None:
            try:
                mode = SilenceDisplayMode(
                    self.preferences.get("silence_display_mode", SilenceDisplayMode.KEY.value)
                )
            except ValueError:
                mode = SilenceDisplayMode.KEY
        visible = mode != SilenceDisplayMode.HIDDEN
        displayed = self.silence_candidates
        if mode == SilenceDisplayMode.KEY:
            displayed = key_silence_candidates(
                self.silence_candidates, self.min_silence_spin.value(),
            )
        if hasattr(self, "spectrogram"):
            self.spectrogram.set_silences(displayed)
            self.spectrogram.set_silences_visible(visible)
        if hasattr(self, "overview"):
            self.overview.set_silences(displayed)
            self.overview.set_silences_visible(visible)

    def _set_list_visualization_mode(self, mode: AudioVisualizationMode | str) -> None:
        mode = AudioVisualizationMode(mode)
        if mode == AudioVisualizationMode.COMBINED:
            mode = AudioVisualizationMode.SPECTROGRAM
        actions = {
            AudioVisualizationMode.NONE: self.list_none_action,
            AudioVisualizationMode.SPECTROGRAM: self.list_spectrum_action,
            AudioVisualizationMode.WAVEFORM: self.list_waveform_action,
        }
        actions[mode].setChecked(True)
        if hasattr(self, "segment_table"):
            self.segment_table.setColumnHidden(2, mode == AudioVisualizationMode.NONE)
            self.mini_delegate.set_mode(mode)
        self.preferences["list_visualization_mode"] = mode.value

    def _set_article_visualization_mode(self, mode: AudioVisualizationMode | str) -> None:
        mode = AudioVisualizationMode(mode)
        if mode == AudioVisualizationMode.COMBINED:
            mode = AudioVisualizationMode.SPECTROGRAM
        actions = {
            AudioVisualizationMode.NONE: self.article_none_action,
            AudioVisualizationMode.SPECTROGRAM: self.article_spectrum_action,
            AudioVisualizationMode.WAVEFORM: self.article_waveform_action,
        }
        actions[mode].setChecked(True)
        if hasattr(self, "article_view") and self.article_view.canvas.mode != mode:
            self.article_view.set_mode(mode)
        self.preferences["article_visualization_mode"] = mode.value

    def _article_visualization_changed(self, mode: str) -> None:
        self._set_article_visualization_mode(AudioVisualizationMode(mode))

    def set_playback_rate(self, value: float) -> None:
        value = round(max(0.25, min(3.0, float(value))) * 20) / 20
        self.playback_rate = value
        self.player.setPlaybackRate(value)
        if hasattr(self, "speed_combo"):
            self.speed_combo.blockSignals(True); self.speed_combo.setCurrentText(f"{value:.2f}×"); self.speed_combo.blockSignals(False)
            self.speed_slider.blockSignals(True); self.speed_slider.setValue(round(value * 100)); self.speed_slider.blockSignals(False)
        self.preferences["playback_rate"] = value
        if value != 1.0 and not hasattr(self.player, "setPitchCompensation") and not self._speed_warning_shown:
            self._speed_warning_shown = True
            self.status_stage.setText("当前播放后端不保证变速不变调")

    def adjust_playback_rate(self, amount: float) -> None:
        self.set_playback_rate(self.playback_rate + amount)

    def _speed_combo_changed(self) -> None:
        text = self.speed_combo.currentText().strip().rstrip("×xX")
        try:
            self.set_playback_rate(float(text))
        except ValueError:
            self.speed_combo.setCurrentText(f"{self.playback_rate:.2f}×")

    def _toggle_follow_action(self, enabled: bool) -> None:
        self.spectrogram.set_follow_enabled(enabled)
        self.preferences["follow_playhead"] = enabled

    def return_to_playhead(self) -> None:
        self.spectrogram.restore_follow()

    def _follow_state_changed(self, state: str) -> None:
        value = PlaybackFollowState(state)
        self.follow_action.blockSignals(True)
        self.follow_action.setChecked(value == PlaybackFollowState.FOLLOWING)
        self.follow_action.blockSignals(False)
        self.follow_button.setChecked(value == PlaybackFollowState.FOLLOWING)
        self.follow_button.setProperty("followSuspended", value == PlaybackFollowState.SUSPENDED)
        self.follow_button.style().unpolish(self.follow_button)
        self.follow_button.style().polish(self.follow_button)
        self.follow_button.setText("⊙")
        labels = {
            PlaybackFollowState.DISABLED: "播放头居中：关闭（Ctrl+Alt+C）",
            PlaybackFollowState.FOLLOWING: "播放头居中：跟随中（Ctrl+Alt+C）",
            PlaybackFollowState.SUSPENDED: "播放头居中：因手动浏览暂时停止，点击◎恢复",
        }
        self.follow_button.setToolTip(labels[value])

    def edit_shortcuts(self) -> None:
        from .shortcut_dialog import ShortcutDialog
        dialog = ShortcutDialog(self.command_actions, self.preferences.get("shortcuts", {}), self)
        if dialog.exec():
            self.preferences["shortcuts"] = dialog.shortcuts()
            for command_id, sequence in dialog.shortcuts().items():
                self.command_actions[command_id].setShortcut(QKeySequence(sequence))

    def _play_state_changed(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        if playing and not self._playback_requested:
            # Suppress a stale play() from a completed proxy/load request.
            QTimer.singleShot(0, self.player.pause)
            return
        self.play_button.setText("⏸" if playing else "▶")
        self.play_button.setToolTip("暂停（Space）" if playing else "播放（Space）")
        if playing:
            self._audio_watch_started_at = time.monotonic()
            self._last_audio_buffer_at = 0.0
        else:
            self._audio_watch_started_at = 0.0

    def _audio_buffer_received(self, buffer) -> None:
        if buffer.isValid() and buffer.byteCount() > 0:
            self._last_audio_buffer_at = time.monotonic()

    def _check_m4b_audio_output(self) -> None:
        """Recover Qt's silent M4B failure, which emits no player error."""
        if (
            self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState
            or not self.current_asset
            or self.current_asset.path.suffix.lower() != ".m4b"
        ):
            return
        current_source = Path(self.player.source().toLocalFile())
        if current_source.suffix.lower() != ".m4b":
            return
        now = time.monotonic()
        if self._audio_watch_started_at <= 0:
            self._audio_watch_started_at = now
            return
        if self._last_audio_buffer_at >= self._audio_watch_started_at:
            return
        if now - self._audio_watch_started_at < 3.0:
            return
        self._audio_watch_started_at = now + 60.0
        self._ensure_m4b_proxy("检测到 M4B 正在走时但没有音频输出")

    def _on_position(self, absolute_ms: int) -> None:
        if self._pending_media_position is not None:
            if abs(absolute_ms - self._pending_media_position) <= 250:
                self._pending_media_position = None
                self._pending_media_deadline = 0.0
            elif time.monotonic() < self._pending_media_deadline:
                # Source changes emit a stale zero/old-source callback before
                # the deferred M4B chapter seek. Never expose that as chapter 0.
                return
            else:
                self._pending_media_position = None
                self._pending_media_autoplay = False
        local = self._local_position()
        self._logical_playhead_ms = local
        duration = self.current_parts[-1][4] if self.current_parts else self.player.duration()
        if self._pending_seek_target is not None:
            if abs(local - self._pending_seek_target) <= 180:
                self._pending_seek_target = None
            elif time.monotonic() < self._pending_seek_deadline:
                # Ignore a late position callback from the previous media
                # source/range.  It must not reselect or stop the old cue.
                return
            else:
                self._pending_seek_target = None
        play_bounds = self._active_play_bounds()
        if play_bounds is not None and local >= play_bounds[1]:
            if self._loop_selection:
                self._seek_local(play_bounds[0])
                self._resume_after_seek()
                return
            self._playback_requested = False
            self.player.pause()
            local = play_bounds[1]
            self._clear_play_range()
        if self.current_link and absolute_ms >= self.current_link.source_end_ms:
            current_index = next((index for index, part in enumerate(self.current_parts) if part[0] is self.current_link), -1)
            if 0 <= current_index + 1 < len(self.current_parts):
                next_start = self.current_parts[current_index + 1][3]
                self._seek_local(next_start)
                self._resume_after_seek()
                return
            self._playback_requested = False
            self.player.pause()
            local = duration
        self.spectrogram.follow_playhead(local)
        self.overview.set_playhead(local)
        self.time_label.setText(f"{_time(local)} / {_time(duration)} · {self.playback_rate:.2f}×")
        row = self.segment_model.row_for_time(local)
        if time.monotonic() < self._manual_selection_until:
            return
        if row >= 0 and row != self.segment_table.currentIndex().row():
            self._playback_row_update = True
            try:
                self._select_segment_row(row)
            finally:
                self._playback_row_update = False

    def _player_error(self, _error, message: str) -> None:
        if self._ensure_m4b_proxy(message or "Qt 无法播放 M4B"):
            return
        if message:
            self.status_stage.setText(f"播放器：{message}")

    def _ensure_m4b_proxy(self, reason: str) -> bool:
        asset = self.current_asset
        if not self.session or not asset or asset.id is None or asset.path.suffix.lower() != ".m4b":
            return False
        source = self.session.resolve_audio(asset)
        if not source:
            return False
        proxy = self.session.root / "cache" / f"playback-{asset.id}.m4a"
        current_source = Path(self.player.source().toLocalFile())
        if (
            proxy.is_file()
            and current_source
            and current_source.resolve() == proxy.resolve()
        ):
            return False
        chapter_id = self.current_chapter_id
        absolute_target = (
            self._pending_media_position
            if self._pending_media_position is not None
            else self._resource_position_for_local(self._logical_playhead_ms)
        )
        resume = bool(
            self._playback_requested
            or self._pending_media_autoplay
            or self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        self._playback_requested = resume
        if proxy.is_file() and proxy.stat().st_size > 0:
            self.status_stage.setText(f"{reason}，已切换 M4A 播放代理")
            self._request_media_position(
                asset, proxy, absolute_target, autoplay=resume, force_source=True,
            )
            return True
        if asset.id in self._proxy_attempted:
            self.status_stage.setText("正在准备 M4A 播放代理，请稍候…")
            return True
        self._proxy_attempted.add(asset.id)
        self.player.pause()
        if resume:
            self.play_button.setText("…")
            self.play_button.setToolTip("正在准备 M4A 代理；点击可取消自动播放")
        self.status_stage.setText(f"{reason}，正在无损封装 M4A 播放代理…")

        def job(progress):
            try:
                progress(0.05, "M4B 无声，正在无损封装为 M4A 播放代理")
                result = create_m4a_proxy(source, proxy)
                progress(1.0, "M4A 播放代理已就绪")
                return result
            except Exception:
                self._proxy_attempted.discard(asset.id or 0)
                raise

        def done(result):
            if (
                self.current_chapter_id != chapter_id
                or not self.current_asset
                or self.current_asset.id != asset.id
            ):
                return
            latest_target = self._resource_position_for_local(self._logical_playhead_ms)
            should_resume = self._playback_requested
            self._request_media_position(
                asset, Path(result), latest_target,
                autoplay=should_resume, force_source=True,
            )
            self.status_stage.setText(
                "已切换 M4A 播放代理并恢复原播放位置"
                if should_resume else "M4A 播放代理已就绪（保持暂停）"
            )

        self.tasks.submit(
            "M4B 播放代理", job,
            done,
            lane=TaskLane.MEDIA, priority=80,
            session_generation=self._session_generation,
        )
        return True

    def _resource_position_for_local(self, local_ms: int) -> int:
        local_ms = max(0, int(local_ms))
        for index, (link, _asset, _path, local_start, local_end) in enumerate(self.current_parts):
            is_last = index == len(self.current_parts) - 1
            if local_start <= local_ms < local_end or (is_last and local_ms == local_end):
                return link.source_start_ms + local_ms - local_start
        return self.current_link.source_start_ms if self.current_link else local_ms

    def _selection_changed(self, start: int, end: int) -> None:
        if end > start:
            self.status_stage.setText(f"选区 {_time(start)} – {_time(end)} · {end - start} ms")

    def _spectrum_view_changed(self, start: int, end: int) -> None:
        self.overview.set_window(start, end)

    def _overview_window(self, start: int, end: int) -> None:
        self.spectrogram.suspend_follow("overview")
        self.spectrogram.set_time_range(start, end)

    def _overview_jump(self, milliseconds: int) -> None:
        span = max(1_000, self.spectrogram.view_end - self.spectrogram.view_start)
        if self.spectrogram.follow_state == PlaybackFollowState.FOLLOWING:
            self._audio_seek_requested(milliseconds)
        self.spectrogram.focus_time(milliseconds, span)

    def _overview_issue_requested(self, row: int) -> None:
        if not 0 <= row < len(self.segment_model.segments):
            return
        self._select_segment_row(row)
        segment = self.segment_model.segments[row]
        span = max(1_000, self.spectrogram.view_end - self.spectrogram.view_start)
        self.spectrogram.focus_time((segment.start_ms + segment.end_ms) // 2, span)

    def export_output(self, kind: str) -> None:
        if not self.session:
            return
        try:
            if kind == "json":
                path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", "alignment.json", "JSON (*.json)")
                if path:
                    export_json(self.session, path)
            else:
                standalone_html = False
                if kind == "html":
                    mode, accepted = QInputDialog.getItem(
                        self,
                        "HTML 导出模式",
                        "导出方式：",
                        [
                            "整书阅读器（index.html + 章节页）",
                            "独立章节 HTML（无 index，章节间不互相引用）",
                        ],
                        0,
                        False,
                    )
                    if not accepted:
                        return
                    standalone_html = mode.startswith("独立章节")
                folder = QFileDialog.getExistingDirectory(self, f"导出 {kind.upper()}")
                if not folder:
                    return
                if kind == "html":
                    export_html(
                        self.session, folder,
                        standalone_chapters=standalone_html,
                    )
                else:
                    export_subtitles(self.session, folder, kind)
            self.status_stage.setText(f"{kind.upper()} 导出完成")
        except Exception as exc:
            self._show_error(str(exc))

    def export_epub_media_overlays(self) -> None:
        if not self.session:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 EPUB 3 Media Overlays",
            f"{self.session.manifest.title}-audio.epub", "EPUB (*.epub)",
        )
        if not path:
            return

        labels = ["自动兼容（推荐）", "强制 AAC-LC / M4A", "强制 MP3"]
        dialog = QDialog(self)
        dialog.setWindowTitle("EPUB 3 Media Overlays 导出选项")
        dialog.setMinimumWidth(460)
        form = QFormLayout(dialog)

        audio_policy_combo = QComboBox(dialog)
        audio_policy_combo.addItems(labels)
        form.addRow("音频处理方式：", audio_policy_combo)

        text_policy_combo = QComboBox(dialog)
        text_policy_combo.addItem(
            "保持原书正文（安全默认）", EpubTextPolicy.PRESERVE_SOURCE,
        )
        text_policy_combo.addItem(
            "尝试把编辑文字应用到导出副本", EpubTextPolicy.APPLY_EDITS,
        )
        text_policy_combo.setToolTip(
            "尝试应用时只替换能够可靠定位且不跨越样式、链接或 ruby 的文字；"
            "失败的句段保持原书正文并在导出报告中列出。"
        )
        form.addRow("正文处理：", text_policy_combo)

        extend_ends = QCheckBox("自动延长句段结束到下一句开始（推荐）", dialog)
        extend_ends.setChecked(True)
        extend_ends.setToolTip(
            "保留相邻句之间的短暂停顿；不会改动项目时间轴，也不会跨过未匹配句段。"
        )
        form.addRow(extend_ends)

        maximum_extension = QSpinBox(dialog)
        maximum_extension.setRange(100, 60_000)
        maximum_extension.setSingleStep(100)
        maximum_extension.setValue(2_000)
        maximum_extension.setSuffix(" 毫秒")
        maximum_extension.setToolTip("超过此长度的空白保持原样，防止吞掉长停顿。")
        form.addRow("最长补齐空隙：", maximum_extension)
        extend_ends.toggled.connect(maximum_extension.setEnabled)

        note = QLabel(
            "仅补齐两个相邻且已有时间的句段之间的空隙；重叠、未匹配段和章节末尾不处理。",
            dialog,
        )
        note.setWordWrap(True)
        form.addRow(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("导出")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        label = audio_policy_combo.currentText()
        policy = {
            labels[0]: AudioConversionPolicy.AUTO_COMPATIBLE,
            labels[1]: AudioConversionPolicy.FORCE_AAC,
            labels[2]: AudioConversionPolicy.FORCE_MP3,
        }[label]
        options = EpubMediaOverlayOptions(
            audio_policy=policy,
            text_policy=EpubTextPolicy(text_policy_combo.currentData()),
            extend_segment_ends=extend_ends.isChecked(),
            max_end_extension_ms=maximum_extension.value(),
        )
        session = self.session
        generation = self._session_generation

        def job(progress):
            worker_session = ProjectSession(session.root, copy.deepcopy(session.manifest))
            try:
                result = export_epub_media_overlay(
                    worker_session, path, options,
                    progress=lambda value, message: progress(value, message),
                )
                return result, list(options.warnings)
            finally:
                worker_session.close()

        def finished(payload) -> None:
            result, warnings = payload
            if generation == self._session_generation:
                if warnings:
                    self.status_stage.setText(
                        f"EPUB 导出完成 · {len(warnings)} 条警告：{result}"
                    )
                    QMessageBox.warning(
                        self, "EPUB 导出报告",
                        f"EPUB 已生成，共有 {len(warnings)} 条正文应用、映射或媒体范围警告。\n\n"
                        + "\n".join(warnings[:20]),
                    )
                else:
                    self.status_stage.setText(f"EPUB 3 Media Overlays 导出完成：{result}")

        self.tasks.submit(
            "导出 EPUB 3 Media Overlays", job, finished,
            lane=TaskLane.MEDIA, priority=30, session_generation=generation,
        )

    def _settings_changed(self, *_args) -> None:
        if not self.session:
            self._update_model_status()
            return
        backend, mode = self._workflow_components()
        self.session.manifest.alignment_mode = mode
        self.session.manifest.asr_backend = backend
        if backend == ASRBackendId.QWEN3_ASR:
            self.session.manifest.qwen_model = self.model_combo.currentText()
        else:
            self.session.manifest.whisper_model = self.model_combo.currentText()
        self.session.manifest.language = self._selected_language_code()
        self.session.manifest.silence = self._silence_settings()
        policy = self.overlap_policy_combo.currentData()
        self.session.manifest.segment_overlap_policy = SegmentOverlapPolicy(
            policy or SegmentOverlapPolicy.CLAMP_CURRENT
        )
        self.session.mark_dirty()
        self._update_title()
        self._update_model_status()

    def _sync_settings_from_manifest(self) -> None:
        manifest = self.session.manifest
        desired_language = manifest.language
        desired_model = manifest.qwen_model if manifest.asr_backend == ASRBackendId.QWEN3_ASR else manifest.whisper_model
        if manifest.alignment_mode == AlignmentMode.QWEN_FORCED:
            workflow = WORKFLOW_QWEN_FORCED
        elif manifest.alignment_mode == AlignmentMode.PRECISE:
            workflow = WORKFLOW_WHISPERX
        elif manifest.asr_backend == ASRBackendId.QWEN3_ASR:
            workflow = WORKFLOW_QWEN_ASR
        else:
            workflow = WORKFLOW_FASTER_WHISPER
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(workflow)))
        self.mode_combo.blockSignals(False)
        self._alignment_mode_changed()
        self.model_combo.setCurrentText(desired_model)
        self._populate_language_options(desired_language)
        self.vad_spin.setValue(manifest.silence.vad_threshold)
        self.min_silence_spin.setValue(manifest.silence.min_silence_ms)
        self.padding_spin.setValue(manifest.silence.boundary_padding_ms)
        self.snap_spin.setValue(manifest.silence.snap_window_ms)
        self.overlap_policy_combo.setCurrentIndex(max(
            0, self.overlap_policy_combo.findData(manifest.segment_overlap_policy)
        ))
        self._update_model_status()

    def _update_model_status(self) -> None:
        workflow, backend, model = self._runtime_status_target()
        key = f"{backend.value}|{model}"
        status = self._runtime_probe_cache.get(key)
        if status is None:
            status = runtime_status(model, self.paths.models, backend, probe_device=False)
            self._schedule_runtime_probe(key, model, backend)
        self._render_model_status(workflow, status)

    def _runtime_status_target(self) -> tuple[str, ASRBackendId, str]:
        workflow = self.mode_combo.currentData() or WORKFLOW_FASTER_WHISPER
        backend, _mode = self._workflow_components()
        if workflow == WORKFLOW_QWEN_FORCED:
            backend = ASRBackendId.QWEN3_ASR
            model = "Qwen3-ForcedAligner-0.6B"
        else:
            model = self.model_combo.currentText() or "small"
        return workflow, backend, model

    def _schedule_runtime_probe(self, key: str, model: str, backend: ASRBackendId) -> None:
        if key in self._runtime_probe_pending:
            return
        self._runtime_probe_pending.add(key)

        def launch() -> None:
            def probe() -> None:
                try:
                    if getattr(sys, "frozen", False):
                        command = [sys.executable, "--runtime-probe", backend.value, model, str(self.paths.models)]
                    else:
                        command = [
                            sys.executable,
                            str(self.paths.root / "run.py"),
                            "--runtime-probe",
                            backend.value,
                            model,
                            str(self.paths.models),
                        ]
                    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    environment = subprocess_runtime_environment()
                    environment["PYTHONIOENCODING"] = "utf-8"
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=20,
                        check=True,
                        env=environment,
                        creationflags=creation_flags,
                    )
                    payload = json.loads(result.stdout.strip().splitlines()[-1])
                    status = RuntimeStatus(**payload)
                except Exception as exc:
                    status = exc
                try:
                    self._runtime_probe_signals.finished.emit(key, status)
                except RuntimeError:
                    pass

            # Only the child process imports CUDA/PyTorch.  This thread waits
            # for its JSON result and therefore cannot hold Qt's Python thread.
            threading.Thread(target=probe, name="audioalign-runtime-probe", daemon=True).start()

        QTimer.singleShot(100, launch)

    def _runtime_probe_finished(self, key: str, status) -> None:
        self._runtime_probe_pending.discard(key)
        if isinstance(status, Exception):
            return
        self._runtime_probe_cache[key] = status
        workflow, backend, model = self._runtime_status_target()
        if key == f"{backend.value}|{model}":
            self._render_model_status(workflow, status)

    def _render_model_status(self, workflow: str, status) -> None:
        suffix = " · WhisperX 未安装" if workflow == WORKFLOW_WHISPERX and not status.whisperx_available else ""
        self.model_status_label.setText(status.message + suffix)
        precise_index = self.mode_combo.findData(WORKFLOW_WHISPERX)
        item = self.mode_combo.model().item(precise_index) if precise_index >= 0 else None
        if item:
            item.setEnabled(status.whisperx_available)

    def _workflow_components(self) -> tuple[ASRBackendId, AlignmentMode]:
        workflow = self.mode_combo.currentData() or WORKFLOW_FASTER_WHISPER
        if workflow == WORKFLOW_WHISPERX:
            return ASRBackendId.FASTER_WHISPER, AlignmentMode.PRECISE
        if workflow == WORKFLOW_QWEN_ASR:
            return ASRBackendId.QWEN3_ASR, AlignmentMode.BALANCED
        if workflow == WORKFLOW_QWEN_FORCED:
            return ASRBackendId.QWEN3_ASR, AlignmentMode.QWEN_FORCED
        return ASRBackendId.FASTER_WHISPER, AlignmentMode.BALANCED

    def _selected_language_code(self) -> str:
        value = self.language_combo.currentData()
        return str(value) if value else self.language_combo.currentText().strip()

    def _populate_language_options(self, preferred: str | None = None) -> None:
        workflow = self.mode_combo.currentData() or WORKFLOW_FASTER_WHISPER
        previous = preferred or self._selected_language_code() or "auto"
        if workflow == WORKFLOW_QWEN_FORCED:
            codes = QWEN_FORCED_LANGUAGE_CODES
            allow_auto = False
            self.language_label.setText("强制对齐语言")
        elif workflow == WORKFLOW_QWEN_ASR:
            codes = tuple(QWEN_ASR_LANGUAGE_NAMES)
            allow_auto = True
            self.language_label.setText("识别语言")
        else:
            codes = WHISPER_LANGUAGE_CODES
            allow_auto = True
            self.language_label.setText("识别语言")
        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        if allow_auto:
            self.language_combo.addItem("自动检测 (auto)", "auto")
        for code in codes:
            name = QWEN_ASR_LANGUAGE_NAMES.get(code)
            self.language_combo.addItem(f"{name} ({code})" if name else code, code)
        selected = self.language_combo.findData(previous)
        if selected < 0:
            selected = self.language_combo.findData("auto" if allow_auto else "zh")
        self.language_combo.setCurrentIndex(max(0, selected))
        self.language_combo.blockSignals(False)

    def _populate_model_options(self) -> None:
        workflow = self.mode_combo.currentData() or WORKFLOW_FASTER_WHISPER
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if workflow == WORKFLOW_QWEN_ASR:
            self.model_combo.addItems(["Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"])
            desired = self.session.manifest.qwen_model if self.session else "Qwen3-ASR-0.6B"
            self.model_combo.setEditable(False)
            self.model_combo.setEnabled(True)
        elif workflow == WORKFLOW_QWEN_FORCED:
            self.model_combo.addItem("Qwen3-ForcedAligner-0.6B")
            desired = "Qwen3-ForcedAligner-0.6B"
            self.model_combo.setEditable(False)
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.addItems(["small", "medium", "large-v3", "turbo"])
            desired = self.session.manifest.whisper_model if self.session else (current or "small")
            self.model_combo.setEditable(True)
            self.model_combo.setEnabled(True)
        self.model_combo.setCurrentText(desired)
        self.model_combo.blockSignals(False)

    def _alignment_mode_changed(self, *_args) -> None:
        workflow = self.mode_combo.currentData() or WORKFLOW_FASTER_WHISPER
        forced = workflow == WORKFLOW_QWEN_FORCED
        self._populate_model_options()
        self._populate_language_options(self.session.manifest.language if self.session else None)
        if hasattr(self, "recognize_button"):
            self.recognize_button.setText(
                "Qwen 强制对齐整个章节" if forced else "运行识别后对齐"
            )
            self.refresh_recognition_button.setVisible(not forced)
            self.clear_recognition_button.setVisible(not forced)
        self._settings_changed()
        self._update_model_status()

    def _mark_dirty(self) -> None:
        if self.session:
            self.session.mark_dirty()
            self._update_title()

    def _autosave(self) -> None:
        if self.session and self.session.dirty and not self.session.archive_path:
            try:
                self.session.autosave()
                self._update_title()
            except Exception as exc:
                self.status_stage.setText(f"自动保存失败：{exc}")

    def pause_tasks(self) -> None:
        self._tasks_paused = not self._tasks_paused
        self.tasks.pause(self._tasks_paused)
        self.pause_button.setText("继续" if self._tasks_paused else "暂停")
        self.status_stage.setText("任务已暂停" if self._tasks_paused else "任务继续")

    def _lane_task_started(self, lane: str, name: str, count: int) -> None:
        if TaskLane(lane) == TaskLane.INFERENCE:
            self._task_started(name, count)
        else:
            self._media_task_fraction = 0.0
            self.media_status.setText(name)
            self.media_progress.setValue(0)

    def _lane_task_progress(self, lane: str, fraction: float, message: str) -> None:
        if TaskLane(lane) == TaskLane.INFERENCE:
            self._task_progress(fraction, message)
        else:
            if fraction >= 0:
                self._media_task_fraction = max(self._media_task_fraction, min(1.0, fraction))
                self.media_progress.setValue(round(self._media_task_fraction * 100))
            self.media_status.setText(message or "媒体缓存")

    def _lane_task_completed(self, lane: str, name: str) -> None:
        if TaskLane(lane) == TaskLane.INFERENCE:
            self._task_completed(name)
        else:
            self.media_progress.setValue(100)
            self.media_status.setText(f"{name}完成")

    def _lane_task_failed(self, lane: str, name: str, details: str) -> None:
        if TaskLane(lane) == TaskLane.INFERENCE:
            self._task_failed(name, details)
        else:
            self.media_progress.setValue(0)
            self.media_status.setText(f"{name}失败")
            self._show_error(details)

    def _lane_task_cancelled(self, lane: str, name: str) -> None:
        if TaskLane(lane) == TaskLane.INFERENCE:
            self._task_cancelled(name)
        else:
            self.media_progress.setValue(0)
            self.media_status.setText(f"{name}已取消")

    def _lane_queue_changed(self, lane: str, count: int) -> None:
        source = TaskLane(lane)
        if source == TaskLane.INFERENCE:
            media = len(self.tasks.queues[TaskLane.MEDIA]) + int(
                self.tasks.currents[TaskLane.MEDIA] is not None
            )
            self.queue_label.setText(f"推理 {count} · 媒体 {media}")
        else:
            inference = len(self.tasks.queues[TaskLane.INFERENCE]) + int(
                self.tasks.currents[TaskLane.INFERENCE] is not None
            )
            self.queue_label.setText(f"推理 {inference} · 媒体 {count}")

    def _task_started(self, name: str, _count: int) -> None:
        self._task_started_at = time.monotonic()
        self._task_fraction = 0.0
        self.status_stage.setText(name)
        self.status_stage.setToolTip(name)
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)

    def _task_progress(self, fraction: float, message: str) -> None:
        chapter_match = re.search(r"全书\s+(\d+)/(\d+)", message)
        if chapter_match:
            self.status_chapters.setText(
                f"任务章节 {chapter_match.group(1)}/{chapter_match.group(2)}"
            )
        if fraction < 0:
            self.status_stage.setText(message)
            self.status_stage.setToolTip(message)
            return
        self._task_fraction = max(self._task_fraction, max(0.0, min(1.0, fraction)))
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(round(self._task_fraction * 100))
        self.status_stage.setText(message)
        self.status_stage.setToolTip(message)

    def _task_completed(self, name: str) -> None:
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(100)
        self.status_stage.setText(f"{name}完成")
        self.status_stage.setToolTip(f"{name}完成")

    def _task_failed(self, name: str, details: str) -> None:
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_stage.setText(f"{name}失败")
        self.status_stage.setToolTip(f"{name}失败")
        self._show_error(details)

    def _task_cancelled(self, name: str) -> None:
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_stage.setText(f"{name}已取消")
        self.status_stage.setToolTip(f"{name}已取消")

    def _update_task_clock(self) -> None:
        if not self.tasks.current:
            self.status_time.setText("耗时 00:00 · 剩余 --:--")
            return
        elapsed = max(0, time.monotonic() - self._task_started_at)
        remaining = elapsed * (1 - self._task_fraction) / self._task_fraction if self._task_fraction > 0.01 else -1
        elapsed_text = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
        remaining_text = "--:--" if remaining < 0 else f"{int(remaining) // 60:02d}:{int(remaining) % 60:02d}"
        self.status_time.setText(f"耗时 {elapsed_text} · 剩余 {remaining_text}")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "AudioAlignTool", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.editor_commit_timer.stop()
        self._commit_segment_editor()
        if self.session and self.session.archive_path and self.session.dirty:
            choice = QMessageBox.question(
                self,
                "保存项目",
                "压缩项目中有未写回的修改，是否保存？",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.StandardButton.Save:
                try:
                    self.session.save()
                except Exception as exc:
                    self._show_error(str(exc))
                    event.ignore()
                    return
        self.tasks.cancel_all()
        self.player.stop()
        self.player.setSource(QUrl())
        if self.current_cache:
            self.current_cache.close()
        if self.session:
            self.session.close()
            self.session = None
        self.preferences["main_window_geometry"] = bytes(self.saveGeometry().toBase64()).decode("ascii")
        self.paths.save_settings(self.preferences)
        event.accept()

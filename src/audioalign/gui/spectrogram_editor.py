from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import QMenu, QVBoxLayout, QWidget

from audioalign.core.models import (
    AudioVisualizationMode,
    BoundaryCandidate,
    PlaybackFollowState,
    SegmentStatus,
    SelectionRange,
    TextSegment,
)
from audioalign.core.spectrogram import AudioVisualizationCache


EMPTY_TIMELINE_MS = 30_000


def _colour_map() -> np.ndarray:
    stops = np.array([0, 55, 120, 185, 235, 255], dtype=float)
    colours = np.array(
        [[5, 8, 22, 255], [18, 35, 75, 255], [18, 115, 135, 255],
         [80, 185, 120, 255], [245, 205, 70, 255], [255, 250, 220, 255]],
        dtype=float,
    )
    values = np.arange(256)
    return np.column_stack(
        [np.interp(values, stops, colours[:, channel]) for channel in range(4)]
    ).astype(np.uint8)


class StableTimeAxis(pg.AxisItem):
    """A single-level time axis whose label interval changes only with zoom."""

    def __init__(self) -> None:
        super().__init__(orientation="bottom")
        self._step_seconds = 1.0

    def set_step(self, milliseconds: int) -> None:
        step = max(0.001, milliseconds / 1000)
        if abs(step - self._step_seconds) < 1e-12:
            return
        self._step_seconds = step
        self.picture = None
        self.update()

    def tickValues(self, minVal: float, maxVal: float, size: float):
        step = self._step_seconds
        first = int(np.ceil((minVal - step * 1e-9) / step))
        last = int(np.floor((maxVal + step * 1e-9) / step))
        values = [index * step for index in range(first, last + 1)]
        return [(step, values)]

    def tickStrings(self, values, scale, spacing):
        step = self._step_seconds
        if step >= 1:
            result = []
            for value in values:
                seconds = round(value)
                sign = "−" if seconds < 0 else ""
                absolute = abs(seconds)
                result.append(f"{sign}{absolute // 60}:{absolute % 60:02d}" if absolute >= 60 else f"{sign}{absolute}")
            return result
        decimals = 1 if step >= 0.1 else 2 if step >= 0.01 else 3
        return [f"{value:.{decimals}f}" for value in values]


class InteractiveViewBox(pg.ViewBox):
    dragStarted = Signal(float, int)
    dragMoved = Signal(float, int)
    dragFinished = Signal(float, int)
    clicked = Signal(float, int)
    doubleClicked = Signal(float)
    wheelInput = Signal(float, float, int)
    hoverMoved = Signal(float)
    hoverLeft = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self._last_hover_global = None

    def hoverEvent(self, event) -> None:
        if event.isExit():
            self._last_hover_global = None
            self.hoverLeft.emit()
        else:
            # ViewBox transforms during centred playback can synthesize hover
            # events under a stationary mouse.  They must not change the edit
            # cursor or its hit target.
            global_position = QCursor.pos()
            if self._last_hover_global == global_position:
                event.acceptClicks(Qt.MouseButton.LeftButton)
                event.acceptDrags(Qt.MouseButton.LeftButton)
                return
            self._last_hover_global = global_position
            point = self.mapSceneToView(event.scenePos())
            self.hoverMoved.emit(float(point.x()))
        event.acceptClicks(Qt.MouseButton.LeftButton)
        event.acceptDrags(Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, event, axis=None) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = int(event.modifiers().value)
            if event.isStart():
                # pyqtgraph emits isStart only after the pointer has crossed
                # the drag threshold.  Use the original press coordinate so a
                # boundary cursor cannot turn into a body move meanwhile.
                try:
                    scene_position = event.buttonDownScenePos()
                except (AttributeError, TypeError):
                    scene_position = event.scenePos()
                point = self.mapSceneToView(scene_position)
                self.dragStarted.emit(float(point.x()), modifiers)
            elif event.isFinish():
                point = self.mapSceneToView(event.scenePos())
                self.dragFinished.emit(float(point.x()), modifiers)
            else:
                point = self.mapSceneToView(event.scenePos())
                self.dragMoved.emit(float(point.x()), modifiers)
            event.accept()
            return
        super().mouseDragEvent(event, axis)

    def mouseClickEvent(self, event) -> None:
        point = self.mapSceneToView(event.scenePos())
        if event.button() == Qt.MouseButton.LeftButton:
            if event.double():
                self.doubleClicked.emit(float(point.x()))
            else:
                self.clicked.emit(float(point.x()), int(event.modifiers().value))
            event.accept()
            return
        super().mouseClickEvent(event)

    def wheelEvent(self, event, axis=None) -> None:
        point = self.mapSceneToView(event.scenePos())
        self.wheelInput.emit(float(point.x()), float(event.delta()), int(event.modifiers().value))
        event.accept()


@dataclass(slots=True)
class _DragTarget:
    kind: str
    segment_index: int = -1


@dataclass(slots=True)
class _Pane:
    kind: str
    plot: pg.PlotWidget
    view: InteractiveViewBox
    time_axis: StableTimeAxis
    play_line: pg.InfiniteLine
    selection: pg.LinearRegionItem
    cues: list
    cue_regions: dict[int, pg.LinearRegionItem]
    cue_labels: dict[int, pg.TextItem]
    silences: list
    grid_lines: list[pg.InfiniteLine]


class AudioVisualizerEditor(QWidget):
    seekRequested = Signal(int)
    selectionChanged = Signal(int, int)
    segmentSelected = Signal(int, int)
    segmentDoubleClicked = Signal(int)
    timeActivated = Signal(int, int, int)
    segmentDoubleClickedAt = Signal(int, int)
    boundaryDragStarted = Signal()
    boundaryMoved = Signal(int, str, int, bool)
    segmentShiftRequested = Signal(int, int)
    segmentEditFinished = Signal(int)
    viewChanged = Signal(int, int)
    modeChanged = Signal(str)
    followStateChanged = Signal(str)
    bindSelectionRequested = Signal()
    newSegmentRequested = Signal()
    splitRequested = Signal()
    mergePreviousRequested = Signal()
    mergeNextRequested = Signal()
    deleteRequested = Signal()
    clearTimingRequested = Signal()
    nextSilenceRequested = Signal()
    playCurrentRequested = Signal()
    editTextRequested = Signal()
    splitPunctuationRequested = Signal()
    lockRequested = Signal()
    restoreSourceRequested = Signal()
    insertBeforeRequested = Signal()
    insertAfterRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.cache: AudioVisualizationCache | None = None
        self.has_audio = False
        self.duration_ms = EMPTY_TIMELINE_MS
        self.view_start = 0
        self.view_end = EMPTY_TIMELINE_MS
        self.playhead = 0
        self.selection: SelectionRange | None = None
        self.segments: list[TextSegment] = []
        self.silences: list[BoundaryCandidate] = []
        self.silences_visible = True
        self.selected_segments: set[int] = set()
        self.mode = AudioVisualizationMode.COMBINED
        self.follow_state = PlaybackFollowState.FOLLOWING
        self._drag = _DragTarget("none")
        self._drag_origin = 0
        self._segment_drag_offset = 0
        self._drag_changed = False
        self._hover_target = _DragTarget("none")
        self._hover_position = -1
        self._syncing = False
        self._last_follow_render = 0.0
        self._last_follow_position = -1
        self._rendered_start = -1
        self._rendered_end = -1
        self._rendered_span = -1
        self._rendered_mode: AudioVisualizationMode | None = None

        self.pane_layout = QVBoxLayout(self)
        self.pane_layout.setContentsMargins(0, 0, 0, 0)
        self.pane_layout.setSpacing(1)
        self.wave_pane = self._make_pane("waveform")
        self.spectrum_pane = self._make_pane("spectrogram")
        # Equal widget and axis geometry keeps both timelines pixel-aligned in
        # combined mode.  Frequency/amplitude labels previously changed the
        # available plot widths independently.
        self.pane_layout.addWidget(self.wave_pane.plot, 1)
        self.pane_layout.addWidget(self.spectrum_pane.plot, 1)
        self._panes = [self.wave_pane, self.spectrum_pane]

        self.wave_min = pg.PlotDataItem(pen=pg.mkPen("#52a7ff", width=1))
        self.wave_max = pg.PlotDataItem(pen=pg.mkPen("#7bc8ff", width=1))
        self.wave_fill = pg.FillBetweenItem(self.wave_min, self.wave_max, brush=pg.mkBrush(55, 145, 235, 105))
        self.wave_pane.plot.addItem(self.wave_fill)
        self.wave_pane.plot.addItem(self.wave_min)
        self.wave_pane.plot.addItem(self.wave_max)
        self.wave_pane.plot.addLine(y=0, pen=pg.mkPen(150, 175, 205, 80))

        self.image = pg.ImageItem(axisOrder="col-major")
        self.image.setLookupTable(_colour_map())
        self.image.setLevels((0, 255))
        self.spectrum_pane.plot.addItem(self.image)
        self.set_mode(self.mode)
        self.set_cache(None)

    @property
    def plot(self) -> pg.PlotWidget:
        return self.spectrum_pane.plot if self.spectrum_pane.plot.isVisible() else self.wave_pane.plot

    def _make_pane(self, kind: str) -> _Pane:
        view = InteractiveViewBox(enableMenu=False)
        time_axis = StableTimeAxis()
        plot = pg.PlotWidget(viewBox=view, axisItems={"bottom": time_axis})
        plot.setBackground((7, 10, 20))
        plot.hideButtons()
        plot.showGrid(x=False, y=False)
        plot.setLabel("bottom", "时间", units="s")
        if kind == "spectrogram":
            plot.setLabel("left", "频率", units="Hz")
            plot.setYRange(50, 8000, padding=0)
            plot.setLimits(yMin=20, yMax=8000)
        else:
            plot.setLabel("left", "振幅")
            plot.setYRange(-1.05, 1.05, padding=0)
            plot.setLimits(yMin=-8, yMax=8)
        plot.getAxis("bottom").setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        plot.getAxis("left").setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        plot.getAxis("left").setWidth(58)
        plot.setMinimumHeight(0)
        view.setMouseEnabled(x=False, y=False)
        play_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#38d9e6", width=2))
        play_line.setZValue(20)
        plot.addItem(play_line)
        selection = pg.LinearRegionItem(values=(0, 0), movable=False, brush=pg.mkBrush(50, 210, 220, 65))
        selection.setVisible(False)
        selection.setZValue(9)
        plot.addItem(selection)
        pane = _Pane(kind, plot, view, time_axis, play_line, selection, [], {}, {}, [], [])
        for _ in range(16):
            grid_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(155, 175, 200, 48, width=1))
            grid_line.setZValue(1)
            grid_line.setVisible(False)
            plot.addItem(grid_line)
            pane.grid_lines.append(grid_line)
        view.dragStarted.connect(self._drag_start)
        view.dragMoved.connect(self._drag_move)
        view.dragFinished.connect(self._drag_finish)
        view.clicked.connect(self._click)
        view.doubleClicked.connect(self._double_click)
        view.wheelInput.connect(lambda centre, delta, modifiers, name=kind: self._wheel(name, centre, delta, modifiers))
        view.hoverMoved.connect(lambda seconds, source=view: self._hover(seconds, source))
        view.hoverLeft.connect(lambda source=view: self._hover_left(source))
        view.sigRangeChanged.connect(lambda _view, ranges, name=kind: self._range_changed(name, ranges))
        plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        plot.customContextMenuRequested.connect(lambda position, source=plot: self._context_menu(source, position))
        return pane

    def set_mode(self, mode: AudioVisualizationMode | str) -> None:
        mode = AudioVisualizationMode(mode)
        if mode == self.mode and self.isVisible():
            return
        self.mode = mode
        self._rendered_mode = None
        self.wave_pane.plot.setVisible(mode in {AudioVisualizationMode.WAVEFORM, AudioVisualizationMode.COMBINED})
        self.spectrum_pane.plot.setVisible(mode in {AudioVisualizationMode.SPECTROGRAM, AudioVisualizationMode.COMBINED})
        self.wave_pane.plot.getAxis("bottom").setVisible(mode != AudioVisualizationMode.COMBINED)
        self.pane_layout.setStretch(0, 1)
        self.pane_layout.setStretch(1, 1)
        self._render_visible()
        self.modeChanged.emit(mode.value)

    def set_cache(self, cache: AudioVisualizationCache | None, duration_ms: int | None = None) -> None:
        self.cache = cache
        self.has_audio = cache is not None or (duration_ms is not None and duration_ms > 0)
        self._rendered_start = self._rendered_end = self._rendered_span = -1
        self.duration_ms = (
            max(1, duration_ms or (cache.metadata.duration_ms if cache else 1))
            if self.has_audio else EMPTY_TIMELINE_MS
        )
        self.view_start, self.view_end = 0, min(self.duration_ms, 30_000)
        for pane in self._panes:
            pane.plot.setLimits(xMin=-self.duration_ms, xMax=self.duration_ms * 2)
        self.set_time_range(self.view_start, self.view_end)

    def set_segments(self, segments: list[TextSegment]) -> None:
        self.segments = segments
        self._render_cues()

    def set_silences(self, candidates: list[BoundaryCandidate]) -> None:
        self.silences = candidates
        self._render_silences()

    def set_silences_visible(self, visible: bool) -> None:
        self.silences_visible = bool(visible)
        for pane in self._panes:
            for item in pane.silences:
                item.setVisible(self.silences_visible)

    def set_playhead(self, milliseconds: int) -> None:
        self.playhead = max(0, min(self.duration_ms, milliseconds)) if self.has_audio else 0
        self._update_playhead_lines()

    def _update_playhead_lines(self) -> None:
        for pane in self._panes:
            pane.play_line.setValue(self.playhead / 1000)

    def set_selection(self, start_ms: int, end_ms: int) -> None:
        normalized = SelectionRange(start_ms, end_ms).normalized()
        self.selection = normalized
        for pane in self._panes:
            pane.selection.setRegion((normalized.start_ms / 1000, normalized.end_ms / 1000))
            pane.selection.setVisible(normalized.end_ms > normalized.start_ms)
        self.selectionChanged.emit(normalized.start_ms, normalized.end_ms)

    def clear_selection(self) -> None:
        self.selection = None
        for pane in self._panes:
            pane.selection.setVisible(False)
        self.selectionChanged.emit(0, 0)

    def set_time_range(self, start_ms: int, end_ms: int) -> None:
        if not self.has_audio:
            start_ms, end_ms = 0, EMPTY_TIMELINE_MS
        span = max(500, end_ms - start_ms)
        minimum = -span // 2
        maximum_start = self.duration_ms - span // 2
        start = max(minimum, min(maximum_start, int(start_ms)))
        end = start + span
        self._syncing = True
        try:
            for pane in self._panes:
                pane.plot.setXRange(start / 1000, end / 1000, padding=0)
        finally:
            self._syncing = False
        self.view_start, self.view_end = start, end
        if self._drag.kind == "none":
            self._clear_hover_state()
        self._update_time_grid()
        self._render_visible()
        self.viewChanged.emit(start, end)

    def _update_time_grid(self) -> None:
        """Draw a small, stable set of foreground time guides over opaque spectrogram pixels."""
        span = max(1, self.view_end - self.view_start)
        rough = span / 8
        magnitude = 10 ** int(np.floor(np.log10(max(1, rough))))
        step = magnitude
        for multiplier in (1, 2, 5, 10):
            candidate = magnitude * multiplier
            if candidate >= rough:
                step = candidate
                break
        first = int(np.floor(self.view_start / step) * step)
        positions = list(range(first, self.view_end + step, step))
        for pane in self._panes:
            pane.time_axis.set_step(step)
            for index, line in enumerate(pane.grid_lines):
                if index < len(positions):
                    line.setValue(positions[index] / 1000)
                    line.setVisible(True)
                else:
                    line.setVisible(False)

    def focus_time(self, milliseconds: int, span_ms: int = 30_000) -> None:
        if not self.has_audio:
            self.set_time_range(0, EMPTY_TIMELINE_MS)
            return
        span = min(max(self.duration_ms, 500), max(500, span_ms))
        self.set_time_range(milliseconds - span // 2, milliseconds + span // 2)

    def reset_view(self) -> None:
        self.wave_pane.plot.setYRange(-1.05, 1.05, padding=0)
        self.spectrum_pane.plot.setYRange(50, 8000, padding=0)
        self.focus_time(self.playhead, 30_000)

    def show_entire_chapter(self) -> None:
        self.wave_pane.plot.setYRange(-1.05, 1.05, padding=0)
        self.spectrum_pane.plot.setYRange(50, 8000, padding=0)
        self.set_time_range(0, self.duration_ms)

    def set_follow_enabled(self, enabled: bool) -> None:
        self.follow_state = PlaybackFollowState.FOLLOWING if enabled else PlaybackFollowState.DISABLED
        self.followStateChanged.emit(self.follow_state.value)
        if enabled:
            self.restore_follow()

    def suspend_follow(self, _reason: str = "manual") -> None:
        if self.follow_state == PlaybackFollowState.FOLLOWING:
            self.follow_state = PlaybackFollowState.SUSPENDED
            self.followStateChanged.emit(self.follow_state.value)

    def restore_follow(self) -> None:
        if self.follow_state != PlaybackFollowState.DISABLED:
            self.follow_state = PlaybackFollowState.FOLLOWING
            self.followStateChanged.emit(self.follow_state.value)
            self.follow_playhead(self.playhead, force=True)

    def follow_playhead(self, milliseconds: int, *, force: bool = False) -> None:
        if not self.has_audio:
            self.playhead = 0
            self._update_playhead_lines()
            return
        milliseconds = max(0, min(self.duration_ms, milliseconds))
        if self.follow_state != PlaybackFollowState.FOLLOWING:
            self.set_playhead(milliseconds)
            return
        self.playhead = milliseconds
        now = time.monotonic()
        span = max(500, self.view_end - self.view_start)
        pixel_ms = span / max(1, self.width())
        if not force and now - self._last_follow_render < 1 / 30 and abs(milliseconds - self._last_follow_position) < pixel_ms:
            return
        self._last_follow_render = now
        self._last_follow_position = milliseconds
        self.set_time_range(milliseconds - span // 2, milliseconds + span // 2)
        self._update_playhead_lines()

    def select_segment(self, index: int, additive: bool = False, toggle: bool = False) -> None:
        previous = set(self.selected_segments)
        if not additive:
            self.selected_segments.clear()
        if toggle and index in self.selected_segments:
            self.selected_segments.remove(index)
        elif index >= 0:
            self.selected_segments.add(index)
        # Selection changes happen at every sentence boundary during playback.
        # Recreating every region and text label here made long chapters stall
        # for one or two seconds.  Only recolour the affected existing regions.
        for changed_index in previous.symmetric_difference(self.selected_segments):
            colour = self._cue_colour(changed_index)
            for pane in self._panes:
                region = pane.cue_regions.get(changed_index)
                if region is not None:
                    region.setBrush(pg.mkBrush(*colour))

    def _render_visible(self) -> None:
        if not self.cache:
            self.image.clear()
            self.wave_min.clear()
            self.wave_max.clear()
            self._rendered_start = self._rendered_end = self._rendered_span = -1
            return
        actual_start = max(0, self.view_start)
        actual_end = min(self.duration_ms, self.view_end)
        if actual_end <= actual_start:
            self.image.clear()
            self.wave_min.clear()
            self.wave_max.clear()
            self._rendered_start = self._rendered_end = self._rendered_span = -1
            return
        span = max(1, self.view_end - self.view_start)
        scale_unchanged = self._rendered_span > 0 and abs(span - self._rendered_span) <= max(1, span // 1000)
        horizontal_margin = max(1, span // 5)
        buffered_view_is_available = (
            scale_unchanged
            and self._rendered_mode == self.mode
            and self._rendered_start <= actual_start - min(horizontal_margin, actual_start)
            and self._rendered_end >= actual_end + min(horizontal_margin, self.duration_ms - actual_end)
        )
        if buffered_view_is_available:
            return
        render_start = max(0, actual_start - span)
        render_end = min(self.duration_ms, actual_end + span)
        width = max(64, round(max(1, render_end - render_start) / span * self.width()))
        if self.spectrum_pane.plot.isVisible():
            data, start, end = self.cache.spectrogram_slice(render_start, render_end, width)
            self.image.setImage(data.T, autoLevels=False)
            self.image.setRect(start / 1000, 50, max(0.001, (end - start) / 1000), 7950)
        if self.wave_pane.plot.isVisible():
            minimum, maximum, start, end = self.cache.waveform_slice(render_start, render_end, width)
            x = np.linspace(start / 1000, end / 1000, minimum.size, endpoint=False)
            self.wave_min.setData(x, minimum)
            self.wave_max.setData(x, maximum)
        self._rendered_start = render_start
        self._rendered_end = render_end
        self._rendered_span = span
        self._rendered_mode = self.mode

    @staticmethod
    def _clear_items(plot: pg.PlotWidget, collection: list) -> None:
        for item in collection:
            plot.removeItem(item)
        collection.clear()

    def _render_cues(self) -> None:
        for pane in self._panes:
            self._clear_items(pane.plot, pane.cues)
            pane.cue_regions.clear()
            pane.cue_labels.clear()
            top = 0.92 if pane.kind == "waveform" else 7600
            for index, segment in enumerate(self.segments):
                if segment.end_ms <= segment.start_ms:
                    continue
                conflict = (
                    (index > 0 and self.segments[index - 1].end_ms > segment.start_ms)
                    or (index + 1 < len(self.segments) and segment.end_ms > self.segments[index + 1].start_ms)
                )
                colour = self._cue_colour(index)
                item = pg.LinearRegionItem(
                    values=(segment.start_ms / 1000, segment.end_ms / 1000), movable=False,
                    brush=pg.mkBrush(*colour),
                    pen=pg.mkPen((235, 65, 80, 220) if conflict else (80, 160, 245, 180), width=1),
                )
                item.lines[0].setPen(pg.mkPen("#42d36b", width=2))
                item.lines[1].setPen(pg.mkPen("#ff6659", width=2))
                item.setZValue(5)
                pane.plot.addItem(item)
                pane.cues.append(item)
                pane.cue_regions[index] = item
                if pane is self.wave_pane or self.mode == AudioVisualizationMode.SPECTROGRAM:
                    label = pg.TextItem(
                        text=segment.text.replace("\n", " ")[:42], color=(235, 240, 250, 225),
                        anchor=(0, 0), fill=pg.mkBrush(8, 12, 22, 135),
                    )
                    label.setPos(segment.start_ms / 1000, top)
                    label.setZValue(7)
                    pane.plot.addItem(label)
                    pane.cues.append(label)
                    pane.cue_labels[index] = label

    def _cue_colour(self, index: int) -> tuple[int, int, int, int]:
        if not 0 <= index < len(self.segments):
            return (72, 125, 230, 68)
        segment = self.segments[index]
        conflict = (
            (index > 0 and self.segments[index - 1].end_ms > segment.start_ms)
            or (index + 1 < len(self.segments) and segment.end_ms > self.segments[index + 1].start_ms)
        )
        if index in self.selected_segments:
            return (75, 180, 255, 125)
        if conflict:
            return (230, 55, 70, 105)
        if segment.status == SegmentStatus.LOW_CONFIDENCE:
            return (242, 172, 48, 82)
        if segment.status == SegmentStatus.UNMATCHED:
            return (220, 65, 90, 82)
        return (72, 125, 230, 68)

    def preview_segment(self, index: int) -> None:
        """Update only the edited cue and its conflict neighbours during a drag."""
        if not 0 <= index < len(self.segments):
            return
        segment = self.segments[index]
        for pane in self._panes:
            region = pane.cue_regions.get(index)
            if region is not None:
                region.setRegion((segment.start_ms / 1000, segment.end_ms / 1000))
            label = pane.cue_labels.get(index)
            if label is not None:
                label.setPos(segment.start_ms / 1000, 0.92 if pane.kind == "waveform" else 7600)
        for changed_index in range(max(0, index - 1), min(len(self.segments), index + 2)):
            colour = self._cue_colour(changed_index)
            for pane in self._panes:
                region = pane.cue_regions.get(changed_index)
                if region is not None:
                    region.setBrush(pg.mkBrush(*colour))

    def _render_silences(self) -> None:
        for pane in self._panes:
            self._clear_items(pane.plot, pane.silences)
            for silence in self.silences:
                if silence.start_ms is None or silence.end_ms is None:
                    continue
                item = pg.LinearRegionItem(
                    values=(silence.start_ms / 1000, silence.end_ms / 1000), movable=False,
                    brush=pg.mkBrush(135, 170, 205, 68), pen=pg.mkPen(130, 180, 220, 110),
                )
                item.setZValue(2)
                pane.plot.addItem(item)
                item.setVisible(self.silences_visible)
                pane.silences.append(item)
                line = pg.InfiniteLine(pos=silence.time_ms / 1000, angle=90, movable=False,
                                       pen=pg.mkPen(185, 215, 240, 160, width=1))
                line.setZValue(3)
                pane.plot.addItem(line)
                line.setVisible(self.silences_visible)
                pane.silences.append(line)

    def _nearest_target(self, milliseconds: int) -> _DragTarget:
        tolerance = max(60, int((self.view_end - self.view_start) / max(1, self.width()) * 8))
        for index, segment in enumerate(self.segments):
            if abs(segment.start_ms - milliseconds) <= tolerance:
                return _DragTarget("start", index)
            if abs(segment.end_ms - milliseconds) <= tolerance:
                return _DragTarget("end", index)
        index = self._segment_at(milliseconds)
        return _DragTarget("body", index) if index >= 0 else _DragTarget("selection")

    def _hover(self, seconds: float, view: InteractiveViewBox) -> None:
        if self._drag.kind != "none":
            return
        milliseconds = max(0, min(self.duration_ms, round(seconds * 1000)))
        target = self._nearest_target(milliseconds)
        self._hover_target = target
        self._hover_position = milliseconds
        cursors = {
            "start": Qt.CursorShape.SizeHorCursor,
            "end": Qt.CursorShape.SizeHorCursor,
            "body": Qt.CursorShape.ArrowCursor,
            "selection": Qt.CursorShape.CrossCursor,
        }
        view.setCursor(QCursor(cursors[target.kind]))

    def _hover_left(self, view: InteractiveViewBox) -> None:
        view.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        if self._drag.kind == "none":
            self._hover_target = _DragTarget("none")
            self._hover_position = -1

    def _clear_hover_state(self) -> None:
        self._hover_target = _DragTarget("none")
        self._hover_position = -1
        for pane in self._panes:
            pane.view.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _highlight_drag_boundary(self, target: _DragTarget, active: bool) -> None:
        if target.kind not in {"start", "end"}:
            return
        line_index = 0 if target.kind == "start" else 1
        normal = "#42d36b" if target.kind == "start" else "#ff6659"
        for pane in self._panes:
            region = pane.cue_regions.get(target.segment_index)
            if region is not None:
                region.lines[line_index].setPen(
                    pg.mkPen("#ffe36e" if active else normal, width=4 if active else 2)
                )

    def _drag_start(self, seconds: float, _modifiers: int) -> None:
        if not self.has_audio:
            return
        milliseconds = max(0, min(self.duration_ms, round(seconds * 1000)))
        self._drag_origin = milliseconds
        target = self._nearest_target(milliseconds)
        tolerance = max(60, int((self.view_end - self.view_start) / max(1, self.width()) * 8))
        if (
            self._hover_target.kind in {"start", "end"}
            and abs(self._hover_position - milliseconds) <= tolerance
        ):
            target = self._hover_target
        self._drag = target
        self._drag_changed = False
        self.suspend_follow("edit")
        if self._drag.kind in {"start", "end", "body"}:
            self.boundaryDragStarted.emit()
            for pane in self._panes:
                pane.view.setCursor(QCursor(
                    Qt.CursorShape.SizeHorCursor if self._drag.kind in {"start", "end"}
                    else Qt.CursorShape.ClosedHandCursor
                ))
            self._highlight_drag_boundary(self._drag, True)
            if self._drag.kind == "body":
                self._segment_drag_offset = milliseconds - self.segments[self._drag.segment_index].start_ms
        else:
            self.set_selection(milliseconds, milliseconds)

    def _drag_move(self, seconds: float, modifiers: int) -> None:
        milliseconds = max(0, min(self.duration_ms, round(seconds * 1000)))
        if self._drag.kind == "selection":
            self.set_selection(self._drag_origin, milliseconds)
        elif self._drag.kind in {"start", "end"}:
            shift = bool(modifiers & int(Qt.KeyboardModifier.ShiftModifier.value))
            # Precise dragging is the default.  Hold Shift only when the user
            # explicitly wants to snap the boundary to a nearby silence.
            self.boundaryMoved.emit(self._drag.segment_index, self._drag.kind, milliseconds, shift)
            self._drag_changed = True
        elif self._drag.kind == "body":
            delta = milliseconds - self._drag_origin
            if delta:
                self.segmentShiftRequested.emit(self._drag.segment_index, delta)
                self._drag_changed = True
            self._drag_origin = milliseconds

    def _drag_finish(self, seconds: float, modifiers: int) -> None:
        target = self._drag
        self._drag_move(seconds, modifiers)
        self._drag = _DragTarget("none")
        self._highlight_drag_boundary(target, False)
        for pane in self._panes:
            pane.view.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        if self._drag_changed and target.kind in {"start", "end", "body"}:
            self.segmentEditFinished.emit(target.segment_index)

    def _segment_at(self, milliseconds: int) -> int:
        candidates = [(s.end_ms - s.start_ms, i) for i, s in enumerate(self.segments)
                      if s.start_ms <= milliseconds <= s.end_ms]
        return min(candidates)[1] if candidates else -1

    def _click(self, seconds: float, modifiers: int) -> None:
        if not self.has_audio:
            return
        milliseconds = max(0, min(self.duration_ms, round(seconds * 1000)))
        index = self._segment_at(milliseconds)
        self.timeActivated.emit(milliseconds, index, modifiers)
        if index >= 0:
            control = bool(modifiers & int(Qt.KeyboardModifier.ControlModifier.value))
            shift = bool(modifiers & int(Qt.KeyboardModifier.ShiftModifier.value))
            self.select_segment(index, additive=control or shift, toggle=control)
            self.segmentSelected.emit(index, modifiers)

    def _double_click(self, seconds: float) -> None:
        if not self.has_audio:
            return
        milliseconds = max(0, min(self.duration_ms, round(seconds * 1000)))
        index = self._segment_at(milliseconds)
        if index >= 0:
            self.select_segment(index)
            self.segmentSelected.emit(index, 0)
            self.segmentDoubleClickedAt.emit(index, milliseconds)
        else:
            self.seekRequested.emit(milliseconds)

    def _wheel(self, pane_name: str, centre_seconds: float, delta: float, modifiers: int) -> None:
        span = self.view_end - self.view_start
        alt = bool(modifiers & int(Qt.KeyboardModifier.AltModifier.value))
        control = bool(modifiers & int(Qt.KeyboardModifier.ControlModifier.value))
        shift = bool(modifiers & int(Qt.KeyboardModifier.ShiftModifier.value))
        pane = self.wave_pane if pane_name == "waveform" else self.spectrum_pane
        if not self.has_audio and not shift:
            return
        if shift:
            current = pane.view.viewRange()[1]
            factor = 0.82 if delta > 0 else 1.22
            centre = sum(current) / 2
            if pane_name == "waveform":
                half = min(8.0, max(0.15, (current[1] - current[0]) * factor / 2))
                pane.plot.setYRange(centre - half, centre + half, padding=0)
            else:
                half = min(3990, max(200, (current[1] - current[0]) * factor / 2))
                pane.plot.setYRange(max(20, centre - half), min(8000, centre + half), padding=0)
            return
        if control or alt:
            factor = 0.75 if delta > 0 else 1.34
            new_span = max(500, min(max(self.duration_ms * 2, 500), int(span * factor)))
            if control:
                # Ctrl+wheel is mouse-anchored zoom, matching common audio
                # editors.  It is an explicit browsing action, so pause the
                # automatic centred follow until the user restores it.
                centre = int(centre_seconds * 1000)
                ratio = max(0.0, min(1.0, (centre - self.view_start) / max(1, span)))
                new_start = centre - round(new_span * ratio)
                self.suspend_follow("ctrl-wheel")
            else:
                centre = self.playhead if self.follow_state == PlaybackFollowState.FOLLOWING else int(centre_seconds * 1000)
                new_start = centre - new_span // 2
        else:
            self.suspend_follow("wheel")
            new_span = span
            new_start = self.view_start + int(span * (-0.12 if delta > 0 else 0.12))
        self.set_time_range(new_start, new_start + new_span)

    def _range_changed(self, pane_name: str, ranges) -> None:
        if self._syncing:
            return
        if not self.has_audio:
            self._syncing = True
            try:
                for pane in self._panes:
                    pane.plot.setXRange(0, EMPTY_TIMELINE_MS / 1000, padding=0)
            finally:
                self._syncing = False
            self.view_start, self.view_end = 0, EMPTY_TIMELINE_MS
            self._update_time_grid()
            return
        start = round(ranges[0][0] * 1000)
        end = max(start + 1, round(ranges[0][1] * 1000))
        self._syncing = True
        try:
            other = self.wave_pane if pane_name == "spectrogram" else self.spectrum_pane
            other.plot.setXRange(start / 1000, end / 1000, padding=0)
        finally:
            self._syncing = False
        self.view_start, self.view_end = start, end
        self._update_time_grid()
        self._render_visible()
        self.viewChanged.emit(start, end)

    def _context_menu(self, source: pg.PlotWidget, position: QPoint) -> None:
        scene_position = source.mapToScene(position)
        point = source.plotItem.vb.mapSceneToView(scene_position)
        milliseconds = max(0, min(self.duration_ms, round(point.x() * 1000)))
        index = self._segment_at(milliseconds)
        if index >= 0:
            self.select_segment(index)
            self.segmentSelected.emit(index, 0)
        menu = QMenu(self)
        actions = []
        if index >= 0:
            actions.extend([
                ("播放当前句", self.playCurrentRequested),
                ("编辑文本", self.editTextRequested),
                ("在当前句前插入", self.insertBeforeRequested),
                ("在当前句后插入", self.insertAfterRequested),
            ])
        if self.selection and self.selection.end_ms > self.selection.start_ms:
            actions.extend([("将选区绑定到当前句", self.bindSelectionRequested),
                            ("用选区新建句段", self.newSegmentRequested)])
        actions.extend([
            ("在播放头处拆分", self.splitRequested), ("按标点拆成多句", self.splitPunctuationRequested),
            ("与前一句合并", self.mergePreviousRequested),
            ("与后一句合并", self.mergeNextRequested), ("清除时间对应", self.clearTimingRequested),
            ("锁定/解锁", self.lockRequested), ("恢复原始段落", self.restoreSourceRequested),
            ("寻找下一处静音", self.nextSilenceRequested), ("删除句段", self.deleteRequested),
        ])
        for label, signal in actions:
            action = QAction(label, menu)
            action.triggered.connect(signal.emit)
            menu.addAction(action)
        menu.addSeparator()
        for label, callback in (("重置视图", self.reset_view), ("显示整章", self.show_entire_chapter)):
            action = QAction(label, menu)
            action.triggered.connect(callback)
            menu.addAction(action)
        menu.exec(source.mapToGlobal(position))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.clear_selection()
            event.accept()
        elif event.key() == Qt.Key.Key_Delete:
            self.deleteRequested.emit()
            event.accept()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.selection:
                if self.selected_segments:
                    self.bindSelectionRequested.emit()
                else:
                    self.newSegmentRequested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class AudioVisualizerOverview(pg.PlotWidget):
    seekRequested = Signal(int)
    windowRequested = Signal(int, int)
    interactionStarted = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setMinimumHeight(64)
        self.setMaximumHeight(82)
        self.setBackground((7, 10, 20))
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.hideButtons()
        self.setMouseEnabled(x=False, y=False)
        self.getAxis("left").setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.getAxis("bottom").setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.has_audio = False
        self.duration_ms = EMPTY_TIMELINE_MS
        self.mode = AudioVisualizationMode.COMBINED
        self.cache: AudioVisualizationCache | None = None
        self.setToolTip(
            "总览图：黄色=低置信度句段，红色=未匹配句段，灰蓝=静音区，"
            "蓝色边框及淡蓝填充=主视图窗口，青色竖线=当前播放位置"
        )
        self.image = pg.ImageItem(axisOrder="col-major")
        self.image.setLookupTable(_colour_map())
        self.image.setLevels((0, 255))
        self.addItem(self.image)
        self.wave_min = pg.PlotDataItem(pen=pg.mkPen("#55b6ff", width=1))
        self.wave_max = pg.PlotDataItem(pen=pg.mkPen("#8dd4ff", width=1))
        self.wave_fill = pg.FillBetweenItem(self.wave_min, self.wave_max, brush=pg.mkBrush(70, 155, 235, 105))
        self.addItem(self.wave_fill)
        self.addItem(self.wave_min)
        self.addItem(self.wave_max)
        self.play_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#35e7ef", width=2),
        )
        self.play_line.setZValue(12)
        self.addItem(self.play_line)
        self.window = pg.LinearRegionItem((0, 1), movable=True, brush=pg.mkBrush(90, 160, 255, 42))
        # PyQtGraph's default region handles are yellow, which is already used
        # for low-confidence cues in this overview. Give the viewport its own
        # unambiguous blue visual language.
        for line in self.window.lines:
            line.setPen(pg.mkPen("#4da3ff", width=2))
            line.setHoverPen(pg.mkPen("#d8efff", width=3))
        self.window.setZValue(8)
        self.addItem(self.window)
        self.window.sigRegionChanged.connect(self.interactionStarted.emit)
        self.window.sigRegionChangeFinished.connect(self._window_moved)
        self.scene().sigMouseClicked.connect(self._scene_clicked)
        self._markers: list = []
        self._silences: list = []
        self.silences_visible = True
        self.set_cache(None)

    def set_playhead(self, milliseconds: int) -> None:
        value = max(0, min(self.duration_ms, int(milliseconds)))
        self.play_line.setValue(value / 1000)
        self.play_line.setVisible(self.has_audio)

    def set_mode(self, mode: AudioVisualizationMode | str) -> None:
        self.mode = AudioVisualizationMode(mode)
        self._render()

    def set_cache(self, cache: AudioVisualizationCache | None, duration_ms: int | None = None) -> None:
        self.cache = cache
        self.has_audio = cache is not None or (duration_ms is not None and duration_ms > 0)
        self.duration_ms = (
            max(1, duration_ms or (cache.metadata.duration_ms if cache else 1))
            if self.has_audio else EMPTY_TIMELINE_MS
        )
        self.setXRange(0, self.duration_ms / 1000, padding=0)
        self.setYRange(-1.05, 1.05, padding=0)
        self.window.blockSignals(True)
        self.window.setBounds((0, self.duration_ms / 1000))
        self.window.blockSignals(False)
        self.set_playhead(0)
        self._render()

    def _render(self) -> None:
        if not self.cache:
            self.image.clear(); self.wave_min.clear(); self.wave_max.clear(); return
        width = max(500, self.width())
        show_spectrum = self.mode in {AudioVisualizationMode.SPECTROGRAM, AudioVisualizationMode.COMBINED}
        show_wave = self.mode in {AudioVisualizationMode.WAVEFORM, AudioVisualizationMode.COMBINED}
        if show_spectrum:
            data, _, _ = self.cache.spectrogram_slice(0, self.duration_ms, width)
            self.image.setImage(data.T, autoLevels=False)
            self.image.setRect(0, -1, self.duration_ms / 1000, 2)
            self.image.setVisible(True)
        else:
            self.image.setVisible(False)
        if show_wave:
            minimum, maximum, _, _ = self.cache.waveform_slice(0, self.duration_ms, width)
            x = np.linspace(0, self.duration_ms / 1000, minimum.size, endpoint=False)
            self.wave_min.setData(x, minimum); self.wave_max.setData(x, maximum)
            self.wave_min.setVisible(True); self.wave_max.setVisible(True); self.wave_fill.setVisible(True)
        else:
            self.wave_min.setVisible(False); self.wave_max.setVisible(False); self.wave_fill.setVisible(False)

    def set_segments(self, segments: list[TextSegment]) -> None:
        for marker in self._markers:
            self.removeItem(marker)
        self._markers.clear()
        for segment in segments:
            if segment.status not in {SegmentStatus.LOW_CONFIDENCE, SegmentStatus.UNMATCHED}:
                continue
            marker = pg.LinearRegionItem(
                (segment.start_ms / 1000, max(segment.start_ms + 40, segment.end_ms) / 1000), movable=False,
                brush=pg.mkBrush(245, 90 if segment.status == SegmentStatus.UNMATCHED else 175, 60, 120),
            )
            marker.setZValue(6); self.addItem(marker); self._markers.append(marker)

    def set_silences(self, silences: list[BoundaryCandidate]) -> None:
        for item in self._silences:
            self.removeItem(item)
        self._silences.clear()
        for silence in silences:
            if silence.start_ms is None or silence.end_ms is None:
                continue
            item = pg.LinearRegionItem((silence.start_ms / 1000, silence.end_ms / 1000), movable=False,
                                       brush=pg.mkBrush(160, 195, 220, 65))
            item.setZValue(4); self.addItem(item); item.setVisible(self.silences_visible); self._silences.append(item)

    def set_silences_visible(self, visible: bool) -> None:
        self.silences_visible = bool(visible)
        for item in self._silences:
            item.setVisible(self.silences_visible)

    def set_window(self, start_ms: int, end_ms: int) -> None:
        start = max(0, start_ms)
        end = min(self.duration_ms, max(start, end_ms))
        self.window.blockSignals(True)
        self.window.setRegion((start / 1000, end / 1000))
        self.window.blockSignals(False)

    def _window_moved(self) -> None:
        if not self.has_audio:
            return
        start, end = self.window.getRegion()
        self.windowRequested.emit(round(start * 1000), round(end * 1000))

    def _scene_clicked(self, event) -> None:
        if (
            not self.has_audio
            or event.button() != Qt.MouseButton.LeftButton
            or self.window.sceneBoundingRect().contains(event.scenePos())
        ):
            return
        point = self.plotItem.vb.mapSceneToView(event.scenePos())
        self.seekRequested.emit(max(0, min(self.duration_ms, round(point.x() * 1000))))


# Old import names remain valid for callers while the implementation is now
# a dual waveform/spectrogram editor.
SpectrogramEditor = AudioVisualizerEditor
SpectrogramOverview = AudioVisualizerOverview

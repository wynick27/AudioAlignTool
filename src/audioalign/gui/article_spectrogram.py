from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QKeyEvent, QPainter, QTextCharFormat, QTextLayout
from PySide6.QtWidgets import QAbstractScrollArea, QButtonGroup, QHBoxLayout, QToolButton, QVBoxLayout, QWidget

from audioalign.core.models import AudioVisualizationMode, TextAudioAnchor, TextSegment
from audioalign.core.spectrogram import AudioVisualizationCache
from audioalign.core.text import is_cjk_text


LOGGER = logging.getLogger(__name__)

_ARTICLE_LUT = np.column_stack(
    [
        np.interp(np.arange(256), [0, 70, 150, 220, 255], channel)
        for channel in ([6, 20, 25, 240, 255], [8, 45, 145, 210, 250],
                        [22, 95, 145, 65, 220], [255, 255, 255, 255, 255])
    ]
).astype(np.uint8)


@dataclass(slots=True)
class ArticleVisualLine:
    text: str
    display_start: int
    display_end: int
    source_start: int
    source_end: int
    segment_indices: tuple[int, ...]
    start_ms: int
    end_ms: int
    estimated: bool
    y: int
    height: int = 86


class ArticleCanvas(QAbstractScrollArea):
    """Virtual, continuous, read-only article with an audio strip per visual line."""

    segmentActivated = Signal(int)
    segmentDoubleClicked = Signal(int)
    rangeSelected = Signal(int, int)
    rangeSelectionFinished = Signal(int, int)
    seekRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.segments: list[TextSegment] = []
        self.anchors: list[TextAudioAnchor] = []
        self.cache: AudioVisualizationCache | None = None
        self.lines: list[ArticleVisualLine] = []
        self.document_text = ""
        self.display_to_source: list[int] = []
        self.segment_display_ranges: list[tuple[int, int]] = []
        self.current_segment = -1
        self.mode = AudioVisualizationMode.SPECTROGRAM
        self.audio_visible = True
        self._image_cache: OrderedDict[tuple, QImage] = OrderedDict()
        self._image_cache_limit = 160
        self._audio_drag_start: int | None = None
        self._text_dragging = False
        self._selection_anchor: int | None = None
        self._selection_cursor: int | None = None
        self._caret_position = 0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.verticalScrollBar().setSingleStep(48)

    def rowCount(self) -> int:
        return len(self.lines)

    def set_content(self, segments: list[TextSegment], anchors: list[TextAudioAnchor],
                    cache: AudioVisualizationCache | None) -> None:
        self.segments = segments
        self.anchors = anchors
        self.cache = cache
        self._image_cache.clear()
        self.clear_text_selection()
        self._build_document()
        self._reflow()

    def set_mode(self, mode: AudioVisualizationMode | str) -> None:
        mode = AudioVisualizationMode(mode)
        if mode == AudioVisualizationMode.COMBINED:
            mode = AudioVisualizationMode.SPECTROGRAM
        if self.mode == mode:
            return
        visibility_changed = self.audio_visible != (mode != AudioVisualizationMode.NONE)
        self.audio_visible = mode != AudioVisualizationMode.NONE
        self.mode = mode
        self._image_cache.clear()
        if visibility_changed:
            self._reflow()
        else:
            self.viewport().update()

    def set_audio_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self.audio_visible == visible:
            return
        self.audio_visible = visible
        self._image_cache.clear()
        self._reflow()

    def selected_text(self) -> str:
        bounds = self._selection_bounds()
        return self.document_text[bounds[0]:bounds[1]] if bounds else ""

    def clear_text_selection(self) -> None:
        self._selection_anchor = None
        self._selection_cursor = None
        self._text_dragging = False
        self.viewport().update()

    def _selection_bounds(self) -> tuple[int, int] | None:
        if self._selection_anchor is None or self._selection_cursor is None:
            return None
        start, end = sorted((self._selection_anchor, self._selection_cursor))
        return (start, end) if end > start else None

    def _build_document(self) -> None:
        text_parts: list[str] = []
        mapping: list[int] = []
        ranges: list[tuple[int, int]] = []
        source_offset = 0
        display_offset = 0
        cjk = is_cjk_text("".join(segment.text for segment in self.segments[:20]))
        for index, segment in enumerate(self.segments):
            text = segment.text.replace("\r\n", "\n").replace("\r", "\n")
            start = display_offset
            text_parts.append(text)
            mapping.extend(range(source_offset, source_offset + len(text)))
            source_offset += len(text)
            display_offset += len(text)
            ranges.append((start, display_offset))
            if index + 1 < len(self.segments):
                separator = "" if cjk or text.endswith(("\n", " ")) else " "
                text_parts.append(separator)
                mapping.extend([-1] * len(separator))
                display_offset += len(separator)
        self.document_text = "".join(text_parts)
        self.display_to_source = mapping
        self.segment_display_ranges = ranges

    def _reflow(self) -> None:
        width = max(180, self.viewport().width() - 36)
        layout = QTextLayout(self.document_text or " ", self.font())
        layout.beginLayout()
        visual_ranges: list[tuple[int, int]] = []
        try:
            while True:
                text_line = layout.createLine()
                if not text_line.isValid():
                    break
                text_line.setLineWidth(width)
                visual_ranges.append((text_line.textStart(), text_line.textStart() + text_line.textLength()))
        finally:
            layout.endLayout()
        lines: list[ArticleVisualLine] = []
        y = 10
        for display_start, display_end in visual_ranges or [(0, len(self.document_text))]:
            segment_indices = tuple(
                index for index, (start, end) in enumerate(self.segment_display_ranges)
                if end > display_start and start < display_end
            )
            source_positions = [value for value in self.display_to_source[display_start:display_end] if value >= 0]
            source_start = min(source_positions) if source_positions else 0
            source_end = max(source_positions) + 1 if source_positions else source_start
            matching = [anchor for anchor in self.anchors
                        if anchor.source_end_char > source_start and anchor.source_start_char < source_end]
            if matching:
                start_ms = min(anchor.start_ms for anchor in matching)
                end_ms = max(anchor.end_ms for anchor in matching)
                estimated = False
            else:
                estimates: list[tuple[int, int]] = []
                for index in segment_indices:
                    segment = self.segments[index]
                    segment_start, segment_end = self.segment_display_ranges[index]
                    overlap_start = max(display_start, segment_start)
                    overlap_end = min(display_end, segment_end)
                    duration = max(0, segment.end_ms - segment.start_ms)
                    start_ratio = (overlap_start - segment_start) / max(1, segment_end - segment_start)
                    end_ratio = (overlap_end - segment_start) / max(1, segment_end - segment_start)
                    estimates.append((segment.start_ms + round(duration * start_ratio),
                                      segment.start_ms + round(duration * end_ratio)))
                start_ms = min((value[0] for value in estimates), default=0)
                end_ms = max((value[1] for value in estimates), default=start_ms)
                estimated = True
            line_height = 86 if self.audio_visible else 38
            lines.append(ArticleVisualLine(
                self.document_text[display_start:display_end], display_start, display_end,
                source_start, source_end, segment_indices, start_ms, max(start_ms, end_ms), estimated, y,
                line_height,
            ))
            y += line_height
        self.lines = lines
        self.verticalScrollBar().setRange(0, max(0, y + 10 - self.viewport().height()))
        self.verticalScrollBar().setPageStep(self.viewport().height())
        self.viewport().update()

    def focus_segment(self, segment_index: int, *, ensure_visible: bool = True) -> None:
        self.current_segment = segment_index
        line = next((item for item in self.lines if segment_index in item.segment_indices), None)
        if line and ensure_visible:
            self.verticalScrollBar().setValue(max(0, line.y - self.viewport().height() // 2))
        self.viewport().update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._image_cache.clear()
        self._reflow()

    def _visible_lines(self) -> list[ArticleVisualLine]:
        offset = self.verticalScrollBar().value()
        top, bottom = offset - 90, offset + self.viewport().height() + 90
        return [line for line in self.lines if line.y + line.height >= top and line.y <= bottom]

    def _audio_image(self, line: ArticleVisualLine, width: int, height: int) -> QImage:
        key = (self.mode.value, line.start_ms, line.end_ms, width, height, id(self.cache))
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached
        image = QImage(max(1, width), max(1, height), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(15, 20, 34))
        painter = QPainter()
        if not painter.begin(image):
            return image
        try:
            valid = self.cache is not None and line.end_ms > line.start_ms
            if valid and self.mode == AudioVisualizationMode.WAVEFORM:
                minimum, maximum, _, _ = self.cache.waveform_slice(line.start_ms, line.end_ms, width)
                valid = minimum.size > 0 and maximum.size > 0
                if valid:
                    painter.setPen(QColor(90, 185, 245, 210))
                    centre = height / 2
                    half = height / 2
                    for x in range(width):
                        index = min(minimum.size - 1, int(x / max(1, width) * minimum.size))
                        painter.drawLine(x, round(centre - maximum[index] * half),
                                         x, round(centre - minimum[index] * half))
            elif valid:
                data, _, _ = self.cache.spectrogram_slice(line.start_ms, line.end_ms, width)
                valid = data.size > 0 and data.shape[1] > 0
                if valid:
                    rgba = np.ascontiguousarray(_ARTICLE_LUT[np.flipud(data)])
                    source = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                                    QImage.Format.Format_RGBA8888).copy()
                    painter.drawImage(QRect(0, 0, width, height), source)
            if not valid:
                painter.setPen(QColor(135, 150, 170))
                painter.drawText(image.rect(), Qt.AlignmentFlag.AlignCenter, "无音频映射")
            painter.setPen(QColor(80, 170, 210, 140))
            painter.drawRect(image.rect().adjusted(0, 0, -1, -1))
        except Exception:
            LOGGER.exception("Failed to render article audio line %s-%s", line.start_ms, line.end_ms)
        finally:
            if painter.isActive():
                painter.end()
        self._image_cache[key] = image
        self._image_cache.move_to_end(key)
        while len(self._image_cache) > self._image_cache_limit:
            self._image_cache.popitem(last=False)
        return image

    def paintEvent(self, _event) -> None:
        visible = self._visible_lines()
        audio_images: dict[int, QImage] = {}
        audio_width = max(1, self.viewport().width() - 36)
        for line in visible:
            if not self.audio_visible:
                continue
            try:
                audio_images[id(line)] = self._audio_image(line, audio_width, 44)
            except Exception:
                LOGGER.exception("Unable to prepare article line")

        painter = QPainter()
        if not painter.begin(self.viewport()):
            return
        try:
            painter.fillRect(self.viewport().rect(), self.palette().base())
            offset = self.verticalScrollBar().value()
            for line in visible:
                y = line.y - offset
                text_rect = QRect(18, y + 2, audio_width, 30)
                audio_rect = QRect(18, y + 34, audio_width, 44)
                self._draw_text(painter, line, text_rect)
                image = audio_images.get(id(line)) if self.audio_visible else None
                if image is not None:
                    painter.drawImage(audio_rect, image)
                if self.audio_visible and line.estimated:
                    painter.setPen(QColor(225, 165, 65))
                    painter.drawText(audio_rect.adjusted(4, 2, -4, -2),
                                     Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, "估算时间")
        except Exception:
            LOGGER.exception("Article viewport paint failed")
        finally:
            if painter.isActive():
                painter.end()

    def _draw_text(self, painter: QPainter, line: ArticleVisualLine, rect: QRect) -> None:
        layout = QTextLayout(line.text, self.font())
        formats: list[QTextLayout.FormatRange] = []

        def format_range(start: int, length: int, char_format: QTextCharFormat) -> QTextLayout.FormatRange:
            # PySide6 exposes FormatRange as a value type.  Unlike PyQt it does
            # not accept (start, length, format) constructor arguments.
            result = QTextLayout.FormatRange()
            result.start = start
            result.length = length
            result.format = char_format
            return result

        if 0 <= self.current_segment < len(self.segment_display_ranges):
            segment_start, segment_end = self.segment_display_ranges[self.current_segment]
            start = max(line.display_start, segment_start) - line.display_start
            end = min(line.display_end, segment_end) - line.display_start
            if end > start:
                current_format = QTextCharFormat()
                current_format.setBackground(QColor(235, 196, 75, 70))
                formats.append(format_range(start, end - start, current_format))
        selection = self._selection_bounds()
        if selection:
            start = max(line.display_start, selection[0]) - line.display_start
            end = min(line.display_end, selection[1]) - line.display_start
            if end > start:
                selection_format = QTextCharFormat()
                selection_format.setBackground(self.palette().highlight())
                selection_format.setForeground(self.palette().highlightedText())
                formats.append(format_range(start, end - start, selection_format))
        layout.setFormats(formats)
        layout.beginLayout()
        try:
            text_line = layout.createLine()
            if text_line.isValid():
                text_line.setLineWidth(rect.width())
        finally:
            layout.endLayout()
        painter.setPen(self.palette().text().color())
        layout.draw(painter, rect.topLeft())

    def _line_at(self, point) -> ArticleVisualLine | None:
        y = point.y() + self.verticalScrollBar().value()
        return next((line for line in self.lines if line.y <= y < line.y + line.height), None)

    def _relative_y(self, line: ArticleVisualLine, point) -> float:
        return point.y() + self.verticalScrollBar().value() - line.y

    def _display_position_at(self, line: ArticleVisualLine, x: float) -> int:
        layout = QTextLayout(line.text, self.font())
        layout.beginLayout()
        try:
            text_line = layout.createLine()
            if not text_line.isValid():
                return line.display_start
            text_line.setLineWidth(max(1, self.viewport().width() - 36))
            local = text_line.xToCursor(max(0.0, x - 18))
            return max(line.display_start, min(line.display_end, line.display_start + local))
        finally:
            layout.endLayout()

    def _segment_at_position(self, position: int, fallback: tuple[int, ...] = ()) -> int:
        for index, (start, end) in enumerate(self.segment_display_ranges):
            if start <= position < end or (position == len(self.document_text) and end == position):
                return index
        return fallback[0] if fallback else -1

    def _segment_at(self, line: ArticleVisualLine, x: float) -> int:
        return self._segment_at_position(self._display_position_at(line, x), line.segment_indices)

    def _time_at(self, line: ArticleVisualLine, x: float) -> int:
        ratio = max(0.0, min(1.0, (x - 18) / max(1, self.viewport().width() - 36)))
        return round(line.start_ms + (line.end_ms - line.start_ms) * ratio)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        line = self._line_at(event.position())
        if not line:
            self.clear_text_selection()
            return
        if not self.audio_visible or self._relative_y(line, event.position()) < 32:
            position = self._display_position_at(line, event.position().x())
            if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) or self._selection_anchor is None:
                self._selection_anchor = position
            self._selection_cursor = position
            self._caret_position = position
            self._text_dragging = True
            index = self._segment_at_position(position, line.segment_indices)
            if index >= 0:
                self.segmentActivated.emit(index)
            self.viewport().update()
            event.accept()
            return
        self._audio_drag_start = self._time_at(line, event.position().x())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        line = self._line_at(event.position())
        if self._text_dragging:
            if line:
                self._selection_cursor = self._display_position_at(line, event.position().x())
                self._caret_position = self._selection_cursor
                self.viewport().update()
            event.accept()
            return
        if self._audio_drag_start is not None:
            if line:
                current = self._time_at(line, event.position().x())
                self.rangeSelected.emit(min(self._audio_drag_start, current), max(self._audio_drag_start, current))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._text_dragging:
            self._text_dragging = False
            event.accept()
            return
        if self._audio_drag_start is not None:
            line = self._line_at(event.position())
            if line:
                current = self._time_at(line, event.position().x())
                if abs(current - self._audio_drag_start) < 5:
                    self.seekRequested.emit(current)
                else:
                    start, end = sorted((self._audio_drag_start, current))
                    self.rangeSelected.emit(start, end)
                    self.rangeSelectionFinished.emit(start, end)
            self._audio_drag_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        line = self._line_at(event.position())
        if not line:
            return super().mouseDoubleClickEvent(event)
        if self.audio_visible and self._relative_y(line, event.position()) >= 32:
            index = self._segment_at(line, event.position().x())
            if index >= 0:
                self.segmentDoubleClicked.emit(index)
                event.accept()
                return
        else:
            position = self._display_position_at(line, event.position().x())
            if self.document_text and position >= len(self.document_text):
                position = len(self.document_text) - 1
            if position >= 0:
                if is_cjk_text(self.document_text[position:position + 1]):
                    start, end = position, min(len(self.document_text), position + 1)
                else:
                    start, end = position, position
                    while start > 0 and (self.document_text[start - 1].isalnum() or self.document_text[start - 1] in "'_-"):
                        start -= 1
                    while end < len(self.document_text) and (self.document_text[end].isalnum() or self.document_text[end] in "'_-"):
                        end += 1
                    if end == start:
                        end = min(len(self.document_text), start + 1)
                self._selection_anchor, self._selection_cursor = start, end
                self.viewport().update()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if control and event.key() == Qt.Key.Key_C:
            text = self.selected_text()
            if text:
                QGuiApplication.clipboard().setText(text)
            event.accept()
            return
        if control and event.key() == Qt.Key.Key_A:
            self._selection_anchor = 0
            self._selection_cursor = len(self.document_text)
            self.viewport().update()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._selection_bounds():
            self.clear_text_selection()
            event.accept()
            return
        super().keyPressEvent(event)


class ArticleSpectrogramView(QWidget):
    segmentActivated = Signal(int)
    segmentDoubleClicked = Signal(int)
    rangeSelected = Signal(int, int)
    rangeSelectionFinished = Signal(int, int)
    modeChanged = Signal(str)
    seekRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        mode_bar = QHBoxLayout()
        mode_bar.addStretch()
        self.none_button = self._mode_button("×", "文章行不显示音频图", AudioVisualizationMode.NONE)
        self.wave_button = self._mode_button("∿", "文章行显示波形", AudioVisualizationMode.WAVEFORM)
        self.spectrum_button = self._mode_button("▥", "文章行显示频谱", AudioVisualizationMode.SPECTROGRAM)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.none_button)
        self.mode_group.addButton(self.wave_button)
        self.mode_group.addButton(self.spectrum_button)
        self.spectrum_button.setChecked(True)
        mode_bar.addWidget(self.none_button)
        mode_bar.addWidget(self.wave_button)
        mode_bar.addWidget(self.spectrum_button)
        layout.addLayout(mode_bar)
        self.canvas = ArticleCanvas(self)
        self.line_model = self.canvas
        layout.addWidget(self.canvas)
        self.canvas.segmentActivated.connect(self.segmentActivated)
        self.canvas.segmentDoubleClicked.connect(self.segmentDoubleClicked)
        self.canvas.rangeSelected.connect(self.rangeSelected)
        self.canvas.rangeSelectionFinished.connect(self.rangeSelectionFinished)
        self.canvas.seekRequested.connect(self.seekRequested)

    def _mode_button(self, text: str, tooltip: str, mode: AudioVisualizationMode) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setStatusTip(tooltip)
        button.setCheckable(True)
        button.setFixedSize(30, 30)
        button.setStyleSheet(
            "QToolButton:checked { background:#2f6fa8; color:white; border:1px solid #77b8ef; }"
        )
        button.clicked.connect(lambda _checked=False, value=mode: self.set_mode(value))
        return button

    def set_mode(self, mode: AudioVisualizationMode | str) -> None:
        mode = AudioVisualizationMode(mode)
        if mode == AudioVisualizationMode.COMBINED:
            mode = AudioVisualizationMode.SPECTROGRAM
        changed = self.canvas.mode != mode
        self.none_button.setChecked(mode == AudioVisualizationMode.NONE)
        self.wave_button.setChecked(mode == AudioVisualizationMode.WAVEFORM)
        self.spectrum_button.setChecked(mode == AudioVisualizationMode.SPECTROGRAM)
        self.canvas.set_mode(mode)
        if changed:
            self.modeChanged.emit(mode.value)

    def set_content(self, segments: list[TextSegment], anchors: list[TextAudioAnchor],
                    cache: AudioVisualizationCache | None) -> None:
        self.canvas.set_content(segments, anchors, cache)

    def set_audio_visible(self, visible: bool) -> None:
        self.set_mode(
            AudioVisualizationMode.SPECTROGRAM
            if visible and self.canvas.mode == AudioVisualizationMode.NONE
            else AudioVisualizationMode.NONE if not visible else self.canvas.mode
        )

    def focus_segment(self, segment_index: int, *, ensure_visible: bool = True) -> None:
        self.canvas.focus_segment(segment_index, ensure_visible=ensure_visible)

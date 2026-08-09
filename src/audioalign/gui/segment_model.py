from __future__ import annotations

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from audioalign.core.models import AudioVisualizationMode, SegmentStatus, TextSegment
from audioalign.core.spectrogram import AudioVisualizationCache


class SegmentTableModel(QAbstractTableModel):
    beforeEdit = Signal()
    segmentEdited = Signal(object)
    HEADERS = ["时间", "正文", "对应频谱", "状态"]

    def __init__(self) -> None:
        super().__init__()
        self.HEADERS = list(type(self).HEADERS)
        self.HEADERS[2] = "对应音频图"
        self.segments: list[TextSegment] = []

    def set_segments(self, segments: list[TextSegment]) -> None:
        self.beginResetModel()
        self.segments = segments
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.segments)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.segments):
            return None
        segment = self.segments[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if index.column() == 0:
                return f"{segment.start_ms / 1000:.3f} – {segment.end_ms / 1000:.3f}"
            if index.column() == 1:
                return segment.text
            if index.column() == 2:
                return ""
            return {
                SegmentStatus.AUTO: "自动",
                SegmentStatus.MANUAL: "人工",
                SegmentStatus.LOCKED: "锁定",
                SegmentStatus.LOW_CONFIDENCE: "待检查",
                SegmentStatus.UNMATCHED: "未匹配",
            }[segment.status]
        if role == Qt.ItemDataRole.UserRole:
            return segment
        if role == Qt.ItemDataRole.BackgroundRole:
            row = index.row()
            previous = self.segments[row - 1] if row > 0 else None
            following = self.segments[row + 1] if row + 1 < len(self.segments) else None
            if (previous and previous.end_ms > segment.start_ms) or (following and segment.end_ms > following.start_ms):
                return QColor(230, 55, 70, 65)
            if segment.status == SegmentStatus.LOW_CONFIDENCE:
                return QColor(255, 190, 60, 45)
            if segment.status == SegmentStatus.UNMATCHED:
                return QColor(220, 60, 90, 40)
            if segment.locked:
                return QColor(70, 120, 220, 35)
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"置信度 {segment.confidence:.0%}" + (" · 已锁定" if segment.locked else "")
        return None

    def flags(self, index: QModelIndex):
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 1:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole or index.column() != 1 or not index.isValid():
            return False
        self.beforeEdit.emit()
        segment = self.segments[index.row()]
        segment.text = str(value).replace("\r\n", "\n").replace("\r", "\n")
        segment.status = SegmentStatus.LOCKED if segment.locked else SegmentStatus.MANUAL
        self.dataChanged.emit(index, index)
        self.segmentEdited.emit(segment)
        return True

    def row_for_time(self, time_ms: int) -> int:
        for index, segment in enumerate(self.segments):
            if segment.start_ms <= time_ms < segment.end_ms:
                return index
        return -1


_LUT = np.column_stack(
    [
        np.interp(np.arange(256), [0, 80, 160, 225, 255], channel)
        for channel in ([7, 18, 35, 245, 255], [10, 55, 165, 205, 250], [25, 105, 145, 55, 220], [255] * 5)
    ]
).astype(np.uint8)


class MiniSpectrogramDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cache: AudioVisualizationCache | None = None
        self.mode = AudioVisualizationMode.SPECTROGRAM
        self.images: dict[tuple, QImage] = {}

    def set_cache(self, cache: AudioVisualizationCache | None) -> None:
        self.cache = cache
        self.images.clear()
        self._refresh_column()

    def set_mode(self, mode: AudioVisualizationMode | str) -> None:
        mode = AudioVisualizationMode(mode)
        if mode == AudioVisualizationMode.COMBINED:
            mode = AudioVisualizationMode.SPECTROGRAM
        if self.mode == mode:
            return
        self.mode = mode
        self.images.clear()
        self._refresh_column()

    def _refresh_column(self) -> None:
        view = self.parent()
        model = view.model() if view else None
        if model and model.rowCount():
            model.dataChanged.emit(model.index(0, 2), model.index(model.rowCount() - 1, 2))

    def _image(self, segment: TextSegment, width: int, height: int) -> QImage | None:
        key = (self.mode.value, segment.start_ms, segment.end_ms, width, height)
        if key in self.images:
            return self.images[key]
        if not self.cache or segment.end_ms <= segment.start_ms or self.mode == AudioVisualizationMode.NONE:
            return None
        if self.mode == AudioVisualizationMode.WAVEFORM:
            minimum, maximum, _, _ = self.cache.waveform_slice(segment.start_ms, segment.end_ms, width)
            if not minimum.size:
                return None
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, :] = (15, 20, 34, 255)
            rgba[height // 2:height // 2 + 1, :, :] = (70, 90, 115, 255)
            xs = np.rint(np.linspace(0, width - 1, minimum.size)).astype(int)
            top = np.clip(np.rint((1.0 - maximum) * (height - 1) / 2), 0, height - 1).astype(int)
            bottom = np.clip(np.rint((1.0 - minimum) * (height - 1) / 2), 0, height - 1).astype(int)
            for x, y1, y2 in zip(xs, top, bottom):
                rgba[min(y1, y2):max(y1, y2) + 1, x, :] = (82, 167, 255, 255)
        else:
            data, _, _ = self.cache.spectrogram_slice(segment.start_ms, segment.end_ms, width)
            if not data.size or data.shape[1] <= 0:
                return None
            rgba = np.ascontiguousarray(_LUT[np.flipud(data)])
        image = QImage(
            rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888,
        ).copy()
        self.images[key] = image
        if len(self.images) > 500:
            self.images.clear()
            self.images[key] = image
        return image

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if index.column() != 2:
            return super().paint(painter, option, index)
        segment: TextSegment = index.data(Qt.ItemDataRole.UserRole)
        painter.save()
        try:
            if option.state & QStyle.StateFlag.State_Selected:
                painter.fillRect(option.rect, option.palette.highlight())
                colour = option.palette.highlightedText().color()
            else:
                colour = option.palette.color(QPalette.ColorRole.Highlight)
            rect = option.rect.adjusted(6, 6, -6, -6)
            painter.fillRect(rect, QColor(15, 20, 34))
            image = self._image(segment, max(24, rect.width()), max(8, rect.height()))
            if image is not None:
                painter.drawImage(QRect(rect.left(), rect.top(), rect.width(), rect.height()), image)
            painter.setPen(colour)
            painter.drawRect(rect)
        finally:
            painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(54, size.height()))
        return size

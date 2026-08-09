from __future__ import annotations

from collections import defaultdict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from audioalign.core.mapping import AudioSlice, audio_slices, automatic_links
from audioalign.core.models import Chapter, ChapterAudioLink
from audioalign.core.storage import ProjectSession


def _clock(milliseconds: int) -> str:
    seconds = max(0, milliseconds) // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class ChapterAudioMappingDialog(QDialog):
    """Transactional one-to-many chapter/audio-slice mapping editor."""

    def __init__(self, session: ProjectSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("章节与音频配对管理器")
        self.resize(1120, 700)
        self.chapters = session.repository.chapters()
        self.chapter_by_id = {chapter.id or 0: chapter for chapter in self.chapters}
        assets = session.repository.all_audio()
        markers = {asset.id or 0: session.repository.audio_chapters(asset.id or 0) for asset in assets}
        self.choices = audio_slices(assets, markers)

        layout = QVBoxLayout(self)
        help_label = QLabel(
            "更改在点击“应用”前不会写入项目。同一文本章节可以添加多个音频切片；M4B 内嵌章节保留原文件时间。"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        tools = QHBoxLayout()
        buttons = [
            ("自动匹配", self.auto_match),
            ("整体偏移 -1", lambda: self.shift(-1)),
            ("整体偏移 +1", lambda: self.shift(1)),
            ("为选中章节添加切片", self.add_slice_row),
            ("片段上移", lambda: self.move_slice(-1)),
            ("片段下移", lambda: self.move_slice(1)),
            ("取消选中配对", self.clear_selected),
        ]
        for label, callback in buttons:
            button = QPushButton(label)
            button.clicked.connect(callback)
            tools.addWidget(button)
        tools.addStretch()
        layout.addLayout(tools)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["文本章节", "文本预览", "顺序", "音频 / M4B 章节", "切片时间", "状态", "置信度"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        header = self.table.horizontalHeader()
        for column in (0, 2, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        links_by_chapter: dict[int, list[ChapterAudioLink]] = defaultdict(list)
        for link in session.repository.all_links():
            links_by_chapter[link.chapter_id].append(link)
        rows: list[tuple[Chapter, ChapterAudioLink | None]] = []
        for chapter in self.chapters:
            links = links_by_chapter.get(chapter.id or 0) or [None]
            rows.extend((chapter, link) for link in links)
        self._populate(rows)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        apply_button = dialog_buttons.button(QDialogButtonBox.StandardButton.Apply)
        cancel_button = dialog_buttons.button(QDialogButtonBox.StandardButton.Cancel)
        apply_button.setText("应用")
        cancel_button.setText("取消")
        apply_button.clicked.connect(self.apply)
        dialog_buttons.rejected.connect(self.reject)
        self.apply_button = apply_button
        self.cancel_button = cancel_button
        layout.addWidget(dialog_buttons)

    def _choice_index(self, link: ChapterAudioLink | None) -> int:
        if not link:
            return -1
        for index, choice in enumerate(self.choices):
            if (choice.audio_id, choice.start_ms, choice.end_ms) == (link.audio_id, link.source_start_ms, link.source_end_ms):
                return index
        return -1

    def _populate(self, rows: list[tuple[Chapter, ChapterAudioLink | None]]) -> None:
        self.table.setRowCount(0)
        positions: dict[int, int] = defaultdict(int)
        for chapter, link in rows:
            position = positions[chapter.id or 0]
            positions[chapter.id or 0] += 1
            self._insert_row(self.table.rowCount(), chapter, link, position)

    def _insert_row(self, row: int, chapter: Chapter, link: ChapterAudioLink | None, position: int) -> None:
        self.table.insertRow(row)
        segments = self.session.repository.segments(chapter.id or 0)
        preview = "".join(segment.text for segment in segments)[:100]
        chapter_item = QTableWidgetItem(chapter.title)
        chapter_item.setData(Qt.ItemDataRole.UserRole, chapter.id)
        self.table.setItem(row, 0, chapter_item)
        self.table.setItem(row, 1, QTableWidgetItem(preview))
        self.table.setItem(row, 2, QTableWidgetItem(str(position + 1)))
        combo = QComboBox()
        combo.addItem("（未配对）", None)
        for index, choice in enumerate(self.choices):
            combo.addItem(f"{choice.path_label}  ·  {choice.title}", index)
        choice_index = self._choice_index(link)
        combo.setCurrentIndex(choice_index + 1)
        combo.currentIndexChanged.connect(lambda _index, target=combo: self._refresh_combo_row(target))
        self.table.setCellWidget(row, 3, combo)
        self.table.setItem(row, 6, QTableWidgetItem(f"{link.confidence:.0%}" if link else "—"))
        self._refresh_row(row)

    def _row_for_combo(self, combo: QComboBox) -> int:
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, 3) is combo:
                return row
        return -1

    def _refresh_combo_row(self, combo: QComboBox) -> None:
        row = self._row_for_combo(combo)
        if row >= 0:
            self._refresh_row(row)

    def _choice(self, row: int) -> AudioSlice | None:
        combo = self.table.cellWidget(row, 3)
        index = combo.currentData() if isinstance(combo, QComboBox) else None
        return self.choices[index] if index is not None else None

    def _refresh_row(self, row: int) -> None:
        choice = self._choice(row)
        self.table.setItem(row, 4, QTableWidgetItem(f"{_clock(choice.start_ms)} – {_clock(choice.end_ms)}" if choice else ""))
        self.table.setItem(row, 5, QTableWidgetItem("已配对" if choice else "未配对"))

    def _chapter_id(self, row: int) -> int:
        return int(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))

    def _current_rows(self) -> list[tuple[Chapter, AudioSlice | None, float]]:
        result = []
        for row in range(self.table.rowCount()):
            chapter = self.chapter_by_id[self._chapter_id(row)]
            confidence_text = self.table.item(row, 6).text().rstrip("%").strip()
            confidence = float(confidence_text) / 100 if confidence_text not in {"", "—"} else 1.0
            result.append((chapter, self._choice(row), confidence))
        return result

    def auto_match(self) -> None:
        text_lengths = {
            chapter.id or 0: sum(len(segment.text) for segment in self.session.repository.segments(chapter.id or 0))
            for chapter in self.chapters
        }
        links = {link.chapter_id: link for link in automatic_links(self.chapters, self.choices, text_lengths)}
        self._populate([(chapter, links.get(chapter.id or 0)) for chapter in self.chapters])

    def shift(self, amount: int) -> None:
        grouped: dict[int, list[AudioSlice]] = defaultdict(list)
        for chapter, choice, _confidence in self._current_rows():
            if choice:
                grouped[chapter.id or 0].append(choice)
        chapter_ids = [chapter.id or 0 for chapter in self.chapters]
        shifted: dict[int, list[AudioSlice]] = defaultdict(list)
        for index, chapter_id in enumerate(chapter_ids):
            destination = index + amount
            if 0 <= destination < len(chapter_ids):
                shifted[chapter_ids[destination]].extend(grouped.get(chapter_id, []))
        rows: list[tuple[Chapter, ChapterAudioLink | None]] = []
        for chapter in self.chapters:
            choices = shifted.get(chapter.id or 0)
            if not choices:
                rows.append((chapter, None))
                continue
            for position, choice in enumerate(choices):
                rows.append((chapter, ChapterAudioLink(None, chapter.id or 0, choice.audio_id, position, choice.start_ms, choice.end_ms, 1.0)))
        self._populate(rows)

    def add_slice_row(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        chapter_id = self._chapter_id(row)
        insert = row + 1
        while insert < self.table.rowCount() and self._chapter_id(insert) == chapter_id:
            insert += 1
        self._insert_row(insert, self.chapter_by_id[chapter_id], None, insert - row)
        self._renumber(chapter_id)
        self.table.selectRow(insert)

    def move_slice(self, amount: int) -> None:
        row = self.table.currentRow()
        destination = row + amount
        if row < 0 or not 0 <= destination < self.table.rowCount() or self._chapter_id(row) != self._chapter_id(destination):
            return
        left, right = self.table.cellWidget(row, 3), self.table.cellWidget(destination, 3)
        left_index, right_index = left.currentIndex(), right.currentIndex()
        left.setCurrentIndex(right_index)
        right.setCurrentIndex(left_index)
        self.table.selectRow(destination)

    def clear_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            chapter_id = self._chapter_id(row)
            siblings = [candidate for candidate in range(self.table.rowCount()) if self._chapter_id(candidate) == chapter_id]
            if len(siblings) > 1:
                self.table.removeRow(row)
            else:
                self.table.cellWidget(row, 3).setCurrentIndex(0)
            self._renumber(chapter_id)

    def _renumber(self, chapter_id: int) -> None:
        position = 1
        for row in range(self.table.rowCount()):
            if self._chapter_id(row) == chapter_id:
                self.table.setItem(row, 2, QTableWidgetItem(str(position)))
                position += 1

    def apply(self) -> None:
        links: list[ChapterAudioLink] = []
        positions: dict[int, int] = defaultdict(int)
        for row in range(self.table.rowCount()):
            chapter_id = self._chapter_id(row)
            choice = self._choice(row)
            if not choice:
                continue
            position = positions[chapter_id]
            positions[chapter_id] += 1
            links.append(ChapterAudioLink(None, chapter_id, choice.audio_id, position, choice.start_ms, choice.end_ms, 1.0))
        self.session.repository.replace_all_chapter_links(links)
        self.session.mark_dirty()
        self.accept()

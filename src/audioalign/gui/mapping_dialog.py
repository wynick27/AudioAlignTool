from __future__ import annotations

import copy
import os
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from audioalign.core.audio import probe_audio
from audioalign.core.mapping import AudioSlice, audio_slices, automatic_links
from audioalign.core.models import AudioAsset, AudioChapterMarker, Chapter, ChapterAudioLink
from audioalign.core.storage import ProjectSession, fingerprint_file, full_fingerprint_file


MEDIA_FILTER = (
    "媒体文件 (*.mp3 *.m4a *.m4b *.aac *.wav *.flac *.ogg *.opus "
    "*.mp4 *.mkv *.mov *.webm);;所有文件 (*.*)"
)


def _clock(milliseconds: int) -> str:
    seconds = max(0, milliseconds) // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


class ChapterAudioMappingDialog(QDialog):
    """Staged media library and transactional chapter mapping editor."""

    def __init__(self, session: ProjectSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("媒体资源与章节配对管理器")
        self.resize(1280, 790)
        self.chapters = session.repository.chapters()
        self.chapter_by_id = {chapter.id or 0: chapter for chapter in self.chapters}
        self.assets = copy.deepcopy(session.repository.all_audio())
        self.markers: dict[int, list[AudioChapterMarker]] = {
            asset.id or 0: copy.deepcopy(session.repository.audio_chapters(asset.id or 0))
            for asset in self.assets
        }
        self._next_temporary_id = -1
        self._next_temporary_chapter_id = -1
        self.choices: list[AudioSlice] = []

        layout = QVBoxLayout(self)
        help_label = QLabel(
            "媒体和章节配对仅在点击“应用”后一次写入。移除只删除项目引用，永不删除磁盘文件。"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        splitter = QSplitter(Qt.Orientation.Vertical)
        media_panel = QWidget()
        media_layout = QVBoxLayout(media_panel)
        media_tools = QHBoxLayout()
        for label, callback in (
            ("添加媒体…", self.add_files),
            ("重新定位/替换…", self.relink_selected),
            ("移除项目引用", self.remove_selected_asset),
            ("上移", lambda: self.move_asset(-1)),
            ("下移", lambda: self.move_asset(1)),
            ("检查重复项", self.check_duplicates),
            ("按未配对媒体创建章节", self.create_chapters_for_unpaired),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            media_tools.addWidget(button)
        media_tools.addStretch()
        media_layout.addLayout(media_tools)
        self.media_table = QTableWidget(0, 7)
        self.media_table.setHorizontalHeaderLabels(
            ["顺序", "标题", "文件", "时长", "格式", "使用章节", "状态"]
        )
        self.media_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.media_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        media_header = self.media_table.horizontalHeader()
        for column in (0, 3, 4, 5, 6):
            media_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        media_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        media_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        media_layout.addWidget(self.media_table)
        splitter.addWidget(media_panel)

        mapping_panel = QWidget()
        mapping_layout = QVBoxLayout(mapping_panel)
        tools = QHBoxLayout()
        for label, callback in (
            ("自动匹配", self.auto_match),
            ("复用到所选章节", self.reuse_for_selected_chapters),
            ("整体偏移 -1", lambda: self.shift(-1)),
            ("整体偏移 +1", lambda: self.shift(1)),
            ("为选中章节添加切片", self.add_slice_row),
            ("片段上移", lambda: self.move_slice(-1)),
            ("片段下移", lambda: self.move_slice(1)),
            ("取消选中配对", self.clear_selected),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            tools.addWidget(button)
        tools.addStretch()
        mapping_layout.addLayout(tools)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["文本章节", "文本预览", "顺序", "媒体 / M4B章节", "切片时间", "状态", "置信度"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.table.horizontalHeader()
        for column in (0, 2, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        mapping_layout.addWidget(self.table)
        splitter.addWidget(mapping_panel)
        splitter.setSizes([290, 450])
        layout.addWidget(splitter)

        self._rebuild_choices()
        links_by_chapter: dict[int, list[ChapterAudioLink]] = defaultdict(list)
        for link in session.repository.all_links():
            links_by_chapter[link.chapter_id].append(link)
        rows: list[tuple[Chapter, ChapterAudioLink | None]] = []
        for chapter in self.chapters:
            links = links_by_chapter.get(chapter.id or 0) or [None]
            rows.extend((chapter, link) for link in links)
        self._populate(rows)
        self._refresh_media_table()

        dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        self.apply_button = dialog_buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.cancel_button = dialog_buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self.apply_button.setText("应用")
        self.cancel_button.setText("取消")
        self.apply_button.clicked.connect(self.apply)
        dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(dialog_buttons)

    def _sync_asset_titles(self) -> None:
        for row, asset in enumerate(self.assets):
            item = self.media_table.item(row, 1)
            if item:
                asset.title = item.text().strip() or Path(asset.absolute_path).stem

    def _mapping_links(self) -> list[ChapterAudioLink]:
        links: list[ChapterAudioLink] = []
        positions: dict[int, int] = defaultdict(int)
        for row in range(self.table.rowCount()):
            chapter_id = self._chapter_id(row)
            choice = self._choice(row)
            if not choice:
                continue
            position = positions[chapter_id]
            positions[chapter_id] += 1
            links.append(ChapterAudioLink(
                None, chapter_id, choice.audio_id, position,
                choice.start_ms, choice.end_ms, self._confidence(row),
            ))
        return links

    def _rebuild_choices(self) -> None:
        self.choices = audio_slices(self.assets, self.markers)

    def _rebuild_mapping_table(self, links: list[ChapterAudioLink]) -> None:
        self._rebuild_choices()
        links_by_chapter: dict[int, list[ChapterAudioLink]] = defaultdict(list)
        for link in links:
            if any(asset.id == link.audio_id for asset in self.assets):
                links_by_chapter[link.chapter_id].append(link)
        rows: list[tuple[Chapter, ChapterAudioLink | None]] = []
        for chapter in self.chapters:
            chapter_links = links_by_chapter.get(chapter.id or 0) or [None]
            rows.extend((chapter, link) for link in chapter_links)
        self._populate(rows)

    def _refresh_media_table(self) -> None:
        links = self._mapping_links() if self.table.rowCount() else []
        usage: dict[int, set[int]] = defaultdict(set)
        for link in links:
            usage[link.audio_id].add(link.chapter_id)
        self.media_table.setRowCount(len(self.assets))
        for row, asset in enumerate(self.assets):
            order = QTableWidgetItem(str(row + 1))
            order.setData(Qt.ItemDataRole.UserRole, asset.id)
            self.media_table.setItem(row, 0, order)
            self.media_table.setItem(row, 1, QTableWidgetItem(asset.title or Path(asset.absolute_path).stem))
            self.media_table.setItem(row, 2, QTableWidgetItem(asset.absolute_path))
            self.media_table.setItem(row, 3, QTableWidgetItem(_clock(asset.duration_ms)))
            self.media_table.setItem(row, 4, QTableWidgetItem(asset.format))
            self.media_table.setItem(row, 5, QTableWidgetItem(str(len(usage.get(asset.id or 0, set())))))
            self.media_table.setItem(
                row, 6, QTableWidgetItem("可用" if Path(asset.absolute_path).exists() else "文件缺失")
            )
            for column in (0, 2, 3, 4, 5, 6):
                self.media_table.item(row, column).setFlags(
                    self.media_table.item(row, column).flags() & ~Qt.ItemFlag.ItemIsEditable
                )

    def add_files(self, checked=False, files: list[str] | None = None) -> None:
        if files is None:
            files, _ = QFileDialog.getOpenFileNames(self, "添加音频或视频", "", MEDIA_FILTER)
        if not files:
            return
        self._sync_asset_titles()
        links = self._mapping_links()
        for file_name in files:
            path = Path(file_name).resolve()
            existing_path = next(
                (asset for asset in self.assets if _path_key(asset.absolute_path) == _path_key(path)), None
            )
            if existing_path:
                QMessageBox.information(self, "媒体已存在", f"该文件已经在项目中：\n{path}")
                continue
            probe = probe_audio(path)
            quick = fingerprint_file(path)
            same_content = None
            for asset in self.assets:
                if asset.fingerprint != quick:
                    continue
                existing = self.session.resolve_audio(asset) if asset.id and asset.id > 0 else Path(asset.absolute_path)
                if existing and existing.exists() and full_fingerprint_file(existing) == full_fingerprint_file(path):
                    same_content = asset
                    break
            if same_content:
                box = QMessageBox(self)
                box.setWindowTitle("发现相同内容")
                box.setText(f"“{path.name}”与已有媒体“{same_content.title}”内容相同。")
                reuse = box.addButton("复用已有资源", QMessageBox.ButtonRole.AcceptRole)
                keep = box.addButton("保留独立副本", QMessageBox.ButtonRole.ActionRole)
                box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
                box.exec()
                if box.clickedButton() is reuse or box.clickedButton() is not keep:
                    continue
            temporary_id = self._next_temporary_id
            self._next_temporary_id -= 1
            asset = AudioAsset(
                temporary_id, str(path), None, quick, probe.duration_ms, probe.sample_rate,
                probe.channels, probe.format, probe.title or path.stem, len(self.assets),
            )
            self.assets.append(asset)
            self.markers[temporary_id] = [
                AudioChapterMarker(None, temporary_id, index, title, start, end)
                for index, (title, start, end) in enumerate(probe.chapters)
            ]
        self._rebuild_mapping_table(links)
        self._refresh_media_table()

    def _selected_asset_row(self) -> int:
        return self.media_table.currentRow()

    def relink_selected(self) -> None:
        row = self._selected_asset_row()
        if not 0 <= row < len(self.assets):
            return
        path, _ = QFileDialog.getOpenFileName(self, "重新定位或替换媒体", "", MEDIA_FILTER)
        if not path:
            return
        target = Path(path).resolve()
        asset = self.assets[row]
        duplicate_path = next((
            candidate for candidate in self.assets
            if candidate.id != asset.id
            and _path_key(candidate.absolute_path) == _path_key(target)
        ), None)
        if duplicate_path is not None:
            answer = QMessageBox.question(
                self, "复用已有媒体",
                "该路径已经在媒体库中。是否把当前资源的所有草稿配对改为复用已有记录，"
                "并移除当前项目引用？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            links = self._mapping_links()
            for link in links:
                if link.audio_id == asset.id:
                    link.audio_id = duplicate_path.id or 0
            unique: dict[tuple[int, int, int, int], ChapterAudioLink] = {}
            for link in links:
                key = (link.chapter_id, link.audio_id, link.source_start_ms, link.source_end_ms)
                unique.setdefault(key, link)
            self.markers.pop(asset.id or 0, None)
            self.assets.pop(row)
            self._rebuild_mapping_table(list(unique.values()))
            self._refresh_media_table()
            return
        quick = fingerprint_file(target)
        same_content = quick == asset.fingerprint
        if same_content and Path(asset.absolute_path).exists():
            same_content = full_fingerprint_file(target) == full_fingerprint_file(asset.absolute_path)
        if not same_content:
            answer = QMessageBox.question(
                self, "媒体内容不同",
                "新文件内容与原资源不同。继续会保留配对但使已有时间轴需要复核，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        probe = probe_audio(target)
        asset.absolute_path = str(target)
        asset.relative_path = None
        asset.fingerprint = quick
        asset.duration_ms = probe.duration_ms
        asset.sample_rate = probe.sample_rate
        asset.channels = probe.channels
        asset.format = probe.format
        asset.title = probe.title or asset.title or target.stem
        self.markers[asset.id or 0] = [
            AudioChapterMarker(None, asset.id or 0, index, title, start, end)
            for index, (title, start, end) in enumerate(probe.chapters)
        ]
        links = self._mapping_links()
        if not same_content:
            for link in links:
                if link.audio_id == asset.id:
                    link.source_end_ms = min(link.source_end_ms, probe.duration_ms)
        self._rebuild_mapping_table(links)
        self._refresh_media_table()
        self.media_table.selectRow(row)

    def remove_selected_asset(self) -> None:
        row = self._selected_asset_row()
        if not 0 <= row < len(self.assets):
            return
        asset = self.assets[row]
        links = self._mapping_links()
        used = [link for link in links if link.audio_id == asset.id]
        if used:
            answer = QMessageBox.question(
                self, "解除媒体配对",
                f"该媒体仍被 {len({link.chapter_id for link in used})} 个章节使用。\n"
                "移除项目引用会解除这些配对，但保留所有文本和时间，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        links = [link for link in links if link.audio_id != asset.id]
        self.markers.pop(asset.id or 0, None)
        self.assets.pop(row)
        self._rebuild_mapping_table(links)
        self._refresh_media_table()

    def move_asset(self, amount: int) -> None:
        row = self._selected_asset_row()
        destination = row + amount
        if row < 0 or not 0 <= destination < len(self.assets):
            return
        self._sync_asset_titles()
        self.assets[row], self.assets[destination] = self.assets[destination], self.assets[row]
        self._refresh_media_table()
        self.media_table.selectRow(destination)

    def check_duplicates(self) -> None:
        groups: dict[str, list[AudioAsset]] = defaultdict(list)
        for asset in self.assets:
            if asset.fingerprint:
                groups[asset.fingerprint].append(asset)
        duplicates = [items for items in groups.values() if len(items) > 1]
        if not duplicates:
            QMessageBox.information(self, "检查重复项", "没有发现相同快速指纹的媒体资源。")
            return
        text = "\n\n".join("\n".join(item.absolute_path for item in items) for items in duplicates)
        QMessageBox.information(
            self, "发现可能的重复项",
            "以下资源需要人工确认；本次检查不会自动合并或删除：\n\n" + text,
        )

    def create_chapters_for_unpaired(self) -> None:
        links = self._mapping_links()
        used = {(link.audio_id, link.source_start_ms, link.source_end_ms) for link in links}
        created = 0
        for choice in self.choices:
            key = (choice.audio_id, choice.start_ms, choice.end_ms)
            if key in used:
                continue
            chapter_id = self._next_temporary_chapter_id
            self._next_temporary_chapter_id -= 1
            chapter = Chapter(chapter_id, choice.title or Path(choice.path_label).stem, len(self.chapters))
            self.chapters.append(chapter)
            self.chapter_by_id[chapter_id] = chapter
            links.append(ChapterAudioLink(
                None, chapter_id, choice.audio_id, 0, choice.start_ms, choice.end_ms, 1.0,
            ))
            created += 1
        if not created:
            QMessageBox.information(self, "创建章节", "没有未配对的媒体或 M4B 章节。")
            return
        self._rebuild_mapping_table(links)
        self._refresh_media_table()

    def _choice_index(self, link: ChapterAudioLink | None) -> int:
        if not link:
            return -1
        for index, choice in enumerate(self.choices):
            if (choice.audio_id, choice.start_ms, choice.end_ms) == (
                link.audio_id, link.source_start_ms, link.source_end_ms,
            ):
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
            combo.addItem(f"{choice.path_label} · {choice.title}", index)
        combo.setCurrentIndex(self._choice_index(link) + 1)
        combo.currentIndexChanged.connect(lambda _index, target=combo: self._refresh_combo_row(target))
        self.table.setCellWidget(row, 3, combo)
        self.table.setItem(row, 6, QTableWidgetItem(f"{link.confidence:.0%}" if link else "—"))
        self._refresh_row(row)

    def _row_for_combo(self, combo: QComboBox) -> int:
        return next((row for row in range(self.table.rowCount()) if self.table.cellWidget(row, 3) is combo), -1)

    def _refresh_combo_row(self, combo: QComboBox) -> None:
        row = self._row_for_combo(combo)
        if row >= 0:
            self._refresh_row(row)
            self._refresh_media_table()

    def _choice(self, row: int) -> AudioSlice | None:
        combo = self.table.cellWidget(row, 3)
        index = combo.currentData() if isinstance(combo, QComboBox) else None
        return self.choices[index] if index is not None and 0 <= index < len(self.choices) else None

    def _refresh_row(self, row: int) -> None:
        choice = self._choice(row)
        self.table.setItem(
            row, 4, QTableWidgetItem(f"{_clock(choice.start_ms)} – {_clock(choice.end_ms)}" if choice else "")
        )
        self.table.setItem(row, 5, QTableWidgetItem("已配对" if choice else "未配对"))

    def _chapter_id(self, row: int) -> int:
        return int(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))

    def _confidence(self, row: int) -> float:
        text = self.table.item(row, 6).text().rstrip("%").strip()
        return float(text) / 100 if text not in {"", "—"} else 1.0

    def _current_rows(self) -> list[tuple[Chapter, AudioSlice | None, float]]:
        return [
            (self.chapter_by_id[self._chapter_id(row)], self._choice(row), self._confidence(row))
            for row in range(self.table.rowCount())
        ]

    def auto_match(self) -> None:
        text_lengths = {
            chapter.id or 0: sum(len(segment.text) for segment in self.session.repository.segments(chapter.id or 0))
            for chapter in self.chapters
        }
        links = {link.chapter_id: link for link in automatic_links(self.chapters, self.choices, text_lengths)}
        self._populate([(chapter, links.get(chapter.id or 0)) for chapter in self.chapters])
        self._refresh_media_table()

    def reuse_for_selected_chapters(self) -> None:
        """Assign one media slice to multiple text chapters without duplicating the asset."""
        source_row = self.table.currentRow()
        choice = self._choice(source_row) if source_row >= 0 else None
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if choice is None or not rows:
            QMessageBox.information(self, "一对多配对", "请先选择一个已配对行，并选中目标章节行。")
            return
        choice_index = next((
            index for index, candidate in enumerate(self.choices)
            if candidate == choice
        ), -1)
        if choice_index < 0:
            return
        chapter_rows: dict[int, int] = {}
        for row in rows:
            chapter_rows.setdefault(self._chapter_id(row), row)
        for row in chapter_rows.values():
            combo = self.table.cellWidget(row, 3)
            if isinstance(combo, QComboBox):
                combo.setCurrentIndex(choice_index + 1)
        self._refresh_media_table()

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
            else:
                rows.extend((chapter, ChapterAudioLink(
                    None, chapter.id or 0, choice.audio_id, position,
                    choice.start_ms, choice.end_ms, 1.0,
                )) for position, choice in enumerate(choices))
        self._populate(rows)
        self._refresh_media_table()

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
        left = self.table.cellWidget(row, 3)
        right = self.table.cellWidget(destination, 3)
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
        self._refresh_media_table()

    def _renumber(self, chapter_id: int) -> None:
        position = 1
        for row in range(self.table.rowCount()):
            if self._chapter_id(row) == chapter_id:
                self.table.setItem(row, 2, QTableWidgetItem(str(position)))
                position += 1

    def apply(self) -> None:
        self._sync_asset_titles()
        self.session.repository.replace_media_library(
            self.assets, self.markers, self._mapping_links(),
            [chapter for chapter in self.chapters if chapter.id is not None and chapter.id < 0],
        )
        self.session.mark_dirty()
        self.accept()

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


DEFAULT_SHORTCUTS = {
    "play_pause": "Space", "play_current": "F5", "loop_current": "Shift+F5",
    "set_start": "F11", "set_end": "F12", "set_end_next": "F10",
    "previous": "Up", "next": "Down", "split": "Ctrl+Alt+V",
    "seek_back": "Left", "seek_forward": "Right", "fine_back": "Alt+Left", "fine_forward": "Alt+Right",
    "new_segment": "Insert", "delete_segment": "Delete",
    "merge": "Ctrl+Shift+M", "lock": "Ctrl+L", "detect_silence": "Ctrl+Alt+D",
    "auto_silence": "Ctrl+Alt+G", "waveform": "Ctrl+1", "spectrogram": "Ctrl+2",
    "combined": "Ctrl+3", "speed_down": "Ctrl+[", "speed_up": "Ctrl+]",
    "speed_reset": "Ctrl+\\", "follow": "Ctrl+Alt+C", "return_playhead": "Ctrl+Alt+Home",
}


class ShortcutDialog(QDialog):
    def __init__(self, actions: dict[str, QAction], _configured: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("快捷键")
        self.resize(620, 520)
        self._actions = actions
        layout = QVBoxLayout(self)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("搜索"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("输入命令名称")
        self.search.textChanged.connect(self._filter)
        search_row.addWidget(self.search)
        layout.addLayout(search_row)
        self.table = QTableWidget(len(actions), 3)
        self.table.setHorizontalHeaderLabels(["分类", "命令", "快捷键"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        for row, (command_id, action) in enumerate(actions.items()):
            category = QTableWidgetItem(self._category(command_id))
            category.setFlags(category.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name = QTableWidgetItem(action.text())
            name.setData(Qt.ItemDataRole.UserRole, command_id)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            shortcut = QTableWidgetItem(action.shortcut().toString(QKeySequence.SequenceFormat.PortableText))
            self.table.setItem(row, 0, category)
            self.table.setItem(row, 1, name)
            self.table.setItem(row, 2, shortcut)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        reset = QPushButton("恢复默认")
        reset.clicked.connect(self._reset)
        apply_button = QPushButton("应用")
        apply_button.clicked.connect(self._accept_checked)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(reset)
        buttons.addStretch()
        buttons.addWidget(apply_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)

    def _filter(self, text: str) -> None:
        needle = text.casefold().strip()
        for row in range(self.table.rowCount()):
            haystack = f"{self.table.item(row, 0).text()} {self.table.item(row, 1).text()}".casefold()
            self.table.setRowHidden(row, needle not in haystack)

    @staticmethod
    def _category(command_id: str) -> str:
        if command_id in {"play_pause", "play_current", "loop_current", "speed_down", "speed_up", "speed_reset",
                          "seek_back", "seek_forward", "fine_back", "fine_forward"}:
            return "播放"
        if command_id in {"waveform", "spectrogram", "combined", "follow", "return_playhead"}:
            return "音频视图"
        if command_id in {"detect_silence", "auto_silence"}:
            return "静音"
        return "句段编辑"

    def _reset(self) -> None:
        for row in range(self.table.rowCount()):
            command_id = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
            self.table.item(row, 2).setText(DEFAULT_SHORTCUTS.get(command_id, ""))

    def _accept_checked(self) -> None:
        seen: dict[str, str] = {}
        for command_id, sequence in self.shortcuts().items():
            normalized = QKeySequence(sequence).toString(QKeySequence.SequenceFormat.PortableText)
            if not normalized:
                continue
            if normalized in seen:
                QMessageBox.warning(
                    self, "快捷键冲突",
                    f"{self._actions[seen[normalized]].text()} 与 {self._actions[command_id].text()} 都使用 {normalized}",
                )
                return
            seen[normalized] = command_id
        self.accept()

    def shortcuts(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in range(self.table.rowCount()):
            command_id = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
            result[command_id] = QKeySequence(self.table.item(row, 2).text()).toString(
                QKeySequence.SequenceFormat.PortableText
            )
        return result

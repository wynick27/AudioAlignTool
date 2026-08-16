from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class FindDialog(QDialog):
    """Small non-modal subtitle search window with regular-expression support."""

    findRequested = Signal(str, bool, bool, bool, str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("查找")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("查找内容"))
        self.query_edit = QLineEdit()
        self.query_edit.setClearButtonEnabled(True)
        self.query_edit.returnPressed.connect(lambda: self.request_find(False))
        query_row.addWidget(self.query_edit, 1)
        layout.addLayout(query_row)

        options = QHBoxLayout()
        self.regex_check = QCheckBox("正则表达式")
        self.case_check = QCheckBox("区分大小写")
        self.whole_word_check = QCheckBox("全词匹配")
        options.addWidget(self.regex_check)
        options.addWidget(self.case_check)
        options.addWidget(self.whole_word_check)
        options.addStretch()
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("当前章节", "chapter")
        self.scope_combo.addItem("整个项目", "project")
        options.addWidget(self.scope_combo)
        layout.addLayout(options)

        buttons = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons.addWidget(self.status_label, 1)
        previous = QPushButton("查找上一个")
        previous.clicked.connect(lambda: self.request_find(True))
        buttons.addWidget(previous)
        next_button = QPushButton("查找下一个")
        next_button.setDefault(True)
        next_button.clicked.connect(lambda: self.request_find(False))
        buttons.addWidget(next_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.hide)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def request_find(self, backwards: bool = False) -> None:
        self.findRequested.emit(
            self.query_edit.text(),
            self.regex_check.isChecked(),
            self.case_check.isChecked(),
            self.whole_word_check.isChecked(),
            str(self.scope_combo.currentData()),
            bool(backwards),
        )

    def set_status(self, message: str, *, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color:#c63b3b;" if error else "")

    def show_for_search(self, selected_text: str = "") -> None:
        if selected_text:
            self.query_edit.setText(selected_text)
        self.show()
        self.raise_()
        self.activateWindow()
        self.query_edit.setFocus()
        self.query_edit.selectAll()


from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from .main_window import MainWindow
from .original_book_view import register_book_scheme


def run(argv: list[str]) -> int:
    QCoreApplication.setOrganizationName("AudioAlignTool")
    QCoreApplication.setApplicationName("AudioAlignTool")
    register_book_scheme()
    app = QApplication(argv)
    app.setStyle("Fusion")
    font = QFont()
    font.setFamilies(["Microsoft YaHei UI", "Segoe UI", "Arial"])
    font.setPointSize(9)
    app.setFont(font)
    try:
        window = MainWindow()
    except PermissionError as exc:
        QMessageBox.critical(None, "AudioAlignTool", str(exc))
        return 2
    window.show()
    return app.exec()

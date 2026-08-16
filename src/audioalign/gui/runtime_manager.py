from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout,
)

from audioalign.core.paths import ApplicationPaths
from audioalign.core.runtime_addons import (
    RuntimeComponent, component_manifest, load_runtime_index,
    install_runtime_component, load_active_runtimes, remove_runtime_component,
)


class _RuntimeSignals(QObject):
    index_ready = Signal(object)
    progress = Signal(float, str)
    installed = Signal(object)
    failed = Signal(str)


def _format_size(size: int) -> str:
    if size <= 0:
        return "安装时计算大小"
    value = float(size)
    for unit in ("B", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return ""


class RuntimeManagerDialog(QDialog):
    """Install optional engines described by the bundled local index."""

    def __init__(self, paths: ApplicationPaths, parent=None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.components: list[RuntimeComponent] = []
        self.signals = _RuntimeSignals(self)
        self.signals.index_ready.connect(self._show_index)
        self.signals.progress.connect(self._show_progress)
        self.signals.installed.connect(self._installed)
        self.signals.failed.connect(self._failed)
        self.setWindowTitle("运行时组件")
        self.resize(720, 470)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "基础便携版已包含 faster-whisper CPU。组件列表来自程序目录中的本地索引；"
            "安装包从 PyPI 或 PyTorch 官方 wheel 源获取，不访问 GitHub。"
            "安装结果保存在 runtimes 中，重启后生效。发布版使用 Python 3.13，"
            "可直接安装 WhisperX CPU 或 GPU 组件。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.list = QListWidget(self)
        self.list.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.list, 1)
        self.detail = QLabel("正在读取本地组件索引…", self)
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("刷新列表", self)
        self.install_button = QPushButton("安装并启用", self)
        self.remove_button = QPushButton("移除", self)
        close_button = QPushButton("关闭", self)
        buttons.addWidget(self.refresh_button)
        buttons.addStretch(1)
        buttons.addWidget(self.install_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self.refresh_button.clicked.connect(self.refresh)
        self.install_button.clicked.connect(self.install_selected)
        self.remove_button.clicked.connect(self.remove_selected)
        close_button.clicked.connect(self.accept)
        self.install_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.refresh()

    def _set_busy(self, busy: bool) -> None:
        self.refresh_button.setEnabled(not busy)
        self.list.setEnabled(not busy)
        component = self._component()
        self.install_button.setEnabled(not busy and component is not None)
        self.progress.setVisible(busy)

    def refresh(self) -> None:
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.detail.setText(f"正在读取本地运行时索引：{self.paths.runtime_index}")

        def job() -> None:
            try:
                self.signals.index_ready.emit(load_runtime_index(self.paths))
            except Exception as exc:
                self.signals.failed.emit(f"无法读取本地运行时索引：{type(exc).__name__}: {exc}")

        threading.Thread(target=job, name="local-runtime-index", daemon=True).start()

    def _show_index(self, components) -> None:
        self.components = list(components)
        active = load_active_runtimes(self.paths)
        self.list.clear()
        for component in self.components:
            installed = component_manifest(component.id, self.paths) is not None
            enabled = active.get(component.group) == component.id
            state = "已启用" if enabled else ("已安装" if installed else "未安装")
            item = QListWidgetItem(f"{component.display_name}  ·  {state}  ·  {_format_size(component.size)}")
            item.setData(Qt.ItemDataRole.UserRole, component.id)
            self.list.addItem(item)
        self._set_busy(False)
        self.progress.setVisible(False)
        self.detail.setText(
            "请选择组件。CPU 与 GPU 版本属于同一功能组，启用一个会替换该组的当前版本。"
            if self.components else
            f"本地索引没有可安装组件：{self.paths.runtime_index}"
        )
        if self.list.count():
            self.list.setCurrentRow(0)

    def _component(self) -> RuntimeComponent | None:
        item = self.list.currentItem()
        if item is None:
            return None
        component_id = item.data(Qt.ItemDataRole.UserRole)
        return next((item for item in self.components if item.id == component_id), None)

    def _selection_changed(self) -> None:
        component = self._component()
        self.install_button.setEnabled(component is not None)
        self.remove_button.setEnabled(bool(component and component_manifest(component.id, self.paths)))
        if component:
            packages = "、".join(component.packages)
            self.detail.setText(
                (component.description or component.display_name)
                + f"\n来源：PyPI / 官方 Python wheel 源\n包：{packages}"
            )

    def install_selected(self) -> None:
        component = self._component()
        if component is None:
            return
        self._set_busy(True)
        self.progress.setRange(0, 1000)

        def report(value: float, message: str) -> None:
            self.signals.progress.emit(value, message)

        def job() -> None:
            try:
                install_runtime_component(component, self.paths, report)
                self.signals.installed.emit(component)
            except Exception as exc:
                self.signals.failed.emit(f"运行时安装失败：{type(exc).__name__}: {exc}")

        threading.Thread(target=job, name="runtime-pip-install", daemon=True).start()

    def _show_progress(self, value: float, message: str) -> None:
        if value < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(value * 1000))
        self.detail.setText(message)

    def _installed(self, component) -> None:
        self._set_busy(False)
        self.progress.setVisible(False)
        self.refresh()
        QMessageBox.information(self, "运行时已安装", f"{component.display_name} 已安装。\n\n请重启程序后使用。")

    def _failed(self, message: str) -> None:
        self._set_busy(False)
        self.progress.setVisible(False)
        self.detail.setText(message)
        QMessageBox.warning(self, "运行时组件", message)

    def remove_selected(self) -> None:
        component = self._component()
        if component is None:
            return
        if QMessageBox.question(
            self, "移除运行时", f"移除 {component.display_name}？模型文件不会删除。",
        ) != QMessageBox.StandardButton.Yes:
            return
        remove_runtime_component(component.id, self.paths)
        self.refresh()

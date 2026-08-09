from __future__ import annotations

import html

from PySide6.QtCore import QUrl, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTextBrowser, QVBoxLayout, QWidget

from audioalign.core.models import ASRToken, SegmentStatus, TextSegment
from audioalign.core.text import normalize_for_match


class ASRComparisonView(QWidget):
    """Selectable source/recognition comparison with audio-position links."""

    seekRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel("尚无识别结果")
        layout.addWidget(self.summary)
        columns = QHBoxLayout()
        self.source = QTextBrowser()
        self.source.setOpenLinks(False)
        self.source.setPlaceholderText("原文")
        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.setPlaceholderText("ASR 识别稿")
        self.transcript.anchorClicked.connect(self._anchor_clicked)
        columns.addWidget(self.source, 1)
        columns.addWidget(self.transcript, 1)
        layout.addLayout(columns)
        self._syncing_scroll = False
        self.source.verticalScrollBar().valueChanged.connect(
            lambda value: self._sync_scroll(self.source.verticalScrollBar(), self.transcript.verticalScrollBar(), value)
        )
        self.transcript.verticalScrollBar().valueChanged.connect(
            lambda value: self._sync_scroll(self.transcript.verticalScrollBar(), self.source.verticalScrollBar(), value)
        )

    def _sync_scroll(self, source, target, value: int) -> None:
        if self._syncing_scroll:
            return
        self._syncing_scroll = True
        try:
            target.setValue(round(value / max(1, source.maximum()) * target.maximum()))
        finally:
            self._syncing_scroll = False

    def set_content(self, segments: list[TextSegment], tokens: list[ASRToken]) -> None:
        source_html = []
        for segment in segments:
            colour = {
                SegmentStatus.UNMATCHED: "#d9576b",
                SegmentStatus.LOW_CONFIDENCE: "#d69a36",
            }.get(segment.status, "inherit")
            source_html.append(
                f'<span style="color:{colour}">{html.escape(segment.text)}</span>'
            )
        self.source.setHtml("<div style='white-space:pre-wrap'>" + "".join(source_html) + "</div>")
        normalized_source = normalize_for_match("".join(segment.text for segment in segments))
        transcript_html = []
        matched = 0
        for token in tokens:
            normalized = normalize_for_match(token.text)
            is_match = bool(normalized and normalized in normalized_source)
            matched += int(is_match)
            colour = "#3aa675" if is_match else "#dc6072"
            transcript_html.append(
                f'<a href="ms:{token.start_ms}" style="color:{colour};text-decoration:none">'
                f'{html.escape(token.text)}</a>'
            )
        self.transcript.setHtml("<div style='white-space:pre-wrap'>" + "".join(transcript_html) + "</div>")
        if tokens:
            self.summary.setText(
                f"识别单元 {len(tokens)} · 可在原文中直接找到 {matched}/{len(tokens)} · 点击识别词跳转音频"
            )
        else:
            self.summary.setText("尚无识别结果")

        self.source.verticalScrollBar().setValue(0)
        self.transcript.verticalScrollBar().setValue(0)

    def _anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() == "ms":
            try:
                self.seekRequested.emit(int(url.path()))
            except ValueError:
                pass

from __future__ import annotations

import html
import unicodedata

from PySide6.QtCore import QUrl, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTextBrowser, QVBoxLayout, QWidget

from audioalign.core.models import ASRToken, SegmentStatus, TextSegment
from audioalign.core.text import normalize_for_match


_STRONG_ENDINGS = frozenset("。！？.!?…")
_CLOSING_PUNCTUATION = frozenset("，。！？；：、,.!?;:%)]}〉》」』】〕〗〙〛’”")
_OPENING_PUNCTUATION = frozenset("([{〈《「『【〔〖〘〚‘“")


def _is_han_or_kana(character: str) -> bool:
    if not character:
        return False
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x9FFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _needs_word_space(previous: str, current: str) -> bool:
    """Restore spaces omitted by word-token ASR without spacing CJK text."""
    if not previous or not current or current[0].isspace() or previous[-1].isspace():
        return False
    left, right = previous[-1], current[0]
    if right in _CLOSING_PUNCTUATION or left in _OPENING_PUNCTUATION:
        return False
    if _is_han_or_kana(left) or _is_han_or_kana(right):
        return False
    if left in _CLOSING_PUNCTUATION:
        return unicodedata.category(right)[0] in "LN"
    return unicodedata.category(left)[0] in "LN" and unicodedata.category(right)[0] in "LN"


def _token_blocks(tokens: list[ASRToken], maximum_tokens: int = 36) -> list[list[ASRToken]]:
    """Create readable ASR paragraphs from pauses, punctuation and a length cap."""
    blocks: list[list[ASRToken]] = []
    current: list[ASRToken] = []
    for token in tokens:
        pause = token.start_ms - current[-1].end_ms if current else 0
        if current and (pause >= 900 or len(current) >= maximum_tokens):
            blocks.append(current)
            current = []
        current.append(token)
        if token.text.rstrip().endswith(tuple(_STRONG_ENDINGS)):
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


class ASRComparisonView(QWidget):
    """Selectable source/recognition comparison with audio-position links."""

    seekRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel("尚无识别结果")
        layout.addWidget(self.summary)
        columns = QHBoxLayout()
        source_column = QVBoxLayout()
        source_column.addWidget(QLabel("原文句段"))
        self.source = QTextBrowser()
        self.source.setOpenLinks(False)
        self.source.setPlaceholderText("原文")
        transcript_column = QVBoxLayout()
        transcript_column.addWidget(QLabel("ASR 识别稿"))
        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.setPlaceholderText("ASR 识别稿")
        self.transcript.anchorClicked.connect(self._anchor_clicked)
        source_column.addWidget(self.source, 1)
        transcript_column.addWidget(self.transcript, 1)
        columns.addLayout(source_column, 1)
        columns.addLayout(transcript_column, 1)
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
        for index, segment in enumerate(segments):
            colour = {
                SegmentStatus.UNMATCHED: "#d9576b",
                SegmentStatus.LOW_CONFIDENCE: "#d69a36",
            }.get(segment.status, "inherit")
            source_html.append(
                '<div style="white-space:pre-wrap;margin:0 0 8px 0;padding:5px 7px;'
                'border-bottom:1px solid #d7dbe0">'
                f'<span style="color:#7b838c;font-size:9pt">{index + 1}.&nbsp;</span>'
                f'<span style="color:{colour}">{html.escape(segment.text)}</span></div>'
            )
        self.source.setHtml("<div>" + "".join(source_html) + "</div>")
        normalized_source = normalize_for_match("".join(segment.text for segment in segments))
        transcript_html: list[str] = []
        matched = 0
        for block_index, block in enumerate(_token_blocks(tokens)):
            rendered: list[str] = []
            previous_text = ""
            for token in block:
                token_text = token.text.strip()
                normalized = normalize_for_match(token_text)
                is_match = bool(normalized and normalized in normalized_source)
                matched += int(is_match)
                colour = "#3aa675" if is_match else "#dc6072"
                separator = " " if _needs_word_space(previous_text, token_text) else ""
                rendered.append(
                    html.escape(separator)
                    + f'<a href="ms:{token.start_ms}" style="color:{colour};text-decoration:none">'
                    + f'{html.escape(token_text)}</a>'
                )
                if token_text:
                    previous_text = token_text
            transcript_html.append(
                '<div style="white-space:pre-wrap;margin:0 0 8px 0;padding:5px 7px;'
                'border-bottom:1px solid #d7dbe0">'
                f'<span style="color:#7b838c;font-size:9pt">{block_index + 1}.&nbsp;</span>'
                + "".join(rendered)
                + "</div>"
            )
        self.transcript.setHtml("<div>" + "".join(transcript_html) + "</div>")
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

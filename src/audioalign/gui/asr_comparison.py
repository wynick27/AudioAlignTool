from __future__ import annotations

import html
import unicodedata

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QTextTable
from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

from audioalign.core.models import ASRToken, SegmentStatus, TextAudioAnchor, TextSegment
from audioalign.core.text import normalize_for_match


_STRONG_ENDINGS = frozenset("。！？.!?…")
_CLOSING_PUNCTUATION = frozenset("，。！？；：、,.!?;:%)]}〉》」』】〗〕’”»…")
_OPENING_PUNCTUATION = frozenset("([{〈《「『【〖〔‘“«¿¡")


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
    """Create readable groups for recognition tokens not assigned to a cue."""
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


class _ComparisonBrowser(QTextBrowser):
    wordClicked = Signal(int)
    wordDoubleClicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._anchor_clicked)

    def _anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() != "ms":
            return
        try:
            self.wordClicked.emit(int(url.path()))
        except ValueError:
            pass

    def mouseDoubleClickEvent(self, event) -> None:
        anchor = self.anchorAt(event.position().toPoint())
        if anchor:
            url = QUrl(anchor)
            if url.scheme() == "ms":
                try:
                    self.wordDoubleClicked.emit(int(url.path()))
                except ValueError:
                    pass
        super().mouseDoubleClickEvent(event)


class ASRComparisonView(QWidget):
    """Source and ASR in paired rows sharing one real scroll position."""

    seekRequested = Signal(int)
    jumpRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel("尚无识别结果")
        layout.addWidget(self.summary)
        self.comparison = _ComparisonBrowser()
        self.comparison.setPlaceholderText("原文与 ASR 识别稿")
        self.comparison.wordClicked.connect(self.seekRequested)
        self.comparison.wordDoubleClicked.connect(self.jumpRequested)
        # Compatibility names for callers that previously addressed the two
        # panes. Both now intentionally share one document and scrollbar.
        self.source = self.comparison
        self.transcript = self.comparison
        layout.addWidget(self.comparison, 1)
        self._table: QTextTable | None = None
        self._current_row = -1
        self._row_count = 0

    def _set_row_background(self, row: int, colour: QColor | None) -> None:
        if self._table is None or not (0 <= row < self._row_count):
            return
        for column in range(2):
            cell = self._table.cellAt(row + 1, column)  # row 0 is the header
            cell_format = cell.format()
            cell_format.setBackground(QBrush(colour) if colour is not None else QBrush())
            cell.setFormat(cell_format)

    def focus_segment(self, row: int, *, ensure_visible: bool = False) -> None:
        """Highlight one paired source/ASR row without rebuilding the document."""
        if row == self._current_row and not ensure_visible:
            return
        self._set_row_background(self._current_row, None)
        self._current_row = row if 0 <= row < self._row_count else -1
        self._set_row_background(self._current_row, QColor("#fff0aa"))
        if ensure_visible and self._table is not None and self._current_row >= 0:
            cursor = self._table.cellAt(self._current_row + 1, 0).firstCursorPosition()
            self.comparison.setTextCursor(cursor)
            self.comparison.ensureCursorVisible()

    @staticmethod
    def _render_tokens(tokens: list[ASRToken], source_text: str) -> tuple[str, int]:
        normalized_source = normalize_for_match(source_text)
        rendered: list[str] = []
        previous_text = ""
        matched = 0
        for token in tokens:
            token_text = token.text.strip()
            normalized = normalize_for_match(token_text)
            is_match = bool(normalized and normalized in normalized_source)
            matched += int(is_match)
            colour = "#3aa675" if is_match else "#dc6072"
            separator = " " if _needs_word_space(previous_text, token_text) else ""
            rendered.append(
                html.escape(separator)
                + f'<a href="ms:{token.start_ms}" style="color:{colour}">'
                + html.escape(token_text) + "</a>"
            )
            if token_text:
                previous_text = token_text
        return "".join(rendered), matched

    def set_content(
        self,
        segments: list[TextSegment],
        tokens: list[ASRToken],
        anchors: list[TextAudioAnchor] | None = None,
    ) -> None:
        token_rows: list[list[ASRToken]] = [[] for _segment in segments]
        unmatched_tokens: list[ASRToken] = []
        row_by_segment_id = {
            segment.id: index for index, segment in enumerate(segments)
            if segment.id is not None
        }
        anchor_rows: dict[tuple[int, int], int] = {}
        for anchor in anchors or []:
            row = row_by_segment_id.get(anchor.segment_id)
            if row is not None:
                anchor_rows[(anchor.start_ms, anchor.end_ms)] = row
        timed = [
            (index, segment) for index, segment in enumerate(segments)
            if segment.end_ms > segment.start_ms
        ]
        for token in tokens:
            anchored_row = anchor_rows.get((token.start_ms, token.end_ms))
            if anchored_row is not None:
                token_rows[anchored_row].append(token)
                continue
            midpoint = (token.start_ms + token.end_ms) / 2
            candidates = [
                (index, segment) for index, segment in timed
                if segment.start_ms <= midpoint <= segment.end_ms
            ]
            if not candidates:
                unmatched_tokens.append(token)
                continue
            index, _segment = min(
                candidates,
                key=lambda item: abs(
                    midpoint - (item[1].start_ms + item[1].end_ms) / 2
                ),
            )
            token_rows[index].append(token)

        rows_html: list[str] = []
        matched = 0
        for index, (segment, row_tokens) in enumerate(zip(segments, token_rows)):
            colour = {
                SegmentStatus.UNMATCHED: "#d9576b",
                SegmentStatus.LOW_CONFIDENCE: "#d69a36",
            }.get(segment.status, "inherit")
            rendered, row_matched = self._render_tokens(row_tokens, segment.text)
            matched += row_matched
            rows_html.append(
                '<tr><td class="source">'
                f'<span class="number">{index + 1}.</span> '
                f'<span style="color:{colour}">{html.escape(segment.text)}</span>'
                '</td><td class="asr">'
                + (rendered or '<span class="empty">— 无对应识别词 —</span>')
                + '</td></tr>'
            )

        for block in _token_blocks(unmatched_tokens):
            rendered, _row_matched = self._render_tokens(block, "")
            rows_html.append(
                '<tr><td class="source empty">— 未匹配原文 —</td>'
                f'<td class="asr">{rendered}</td></tr>'
            )

        document = (
            '<style>table{width:100%;table-layout:fixed;border-collapse:collapse}'
            'th{background:#eef1f4;padding:7px;text-align:left}'
            'td{width:50%;vertical-align:top;padding:7px;white-space:pre-wrap;'
            'border-bottom:1px solid #d7dbe0}td+td,th+th{border-left:1px solid #c8cdd3}'
            '.number{color:#7b838c;font-size:9pt}.empty{color:#9299a1}'
            'a{text-decoration:none}</style>'
            '<table><thead><tr><th>原文句段</th><th>ASR 识别稿（双击词定位）</th>'
            '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table>'
        )
        self.comparison.setHtml(document)
        self._row_count = len(segments)
        root = self.comparison.document().rootFrame()
        self._table = next(
            (frame for frame in root.childFrames() if isinstance(frame, QTextTable)),
            None,
        )
        current_row = self._current_row
        self._current_row = -1
        if 0 <= current_row < self._row_count:
            self.focus_segment(current_row)
        self.summary.setText(
            f"识别单元 {len(tokens)} · 当前句内匹配 {matched}/{len(tokens)} · 双击识别词定位音频"
            if tokens else "尚无识别结果"
        )
        self.comparison.verticalScrollBar().setValue(0)

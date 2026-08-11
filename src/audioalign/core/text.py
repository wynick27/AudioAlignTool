from __future__ import annotations

import html
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree


@dataclass(slots=True)
class ImportedChapter:
    title: str
    text: str
    source_html: str = ""
    fragments: list["ImportedSourceFragment"] = field(default_factory=list)


@dataclass(slots=True)
class ImportedSourceFragment:
    position: int
    kind: str
    text: str
    source_start_char: int
    source_end_char: int


class _HTMLTextExtractor(HTMLParser):
    block_tags = {"p", "div", "h1", "h2", "h3", "h4", "li", "br", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg"}:
            self._ignored += 1
        elif tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg"} and self._ignored:
            self._ignored -= 1
        elif tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        value = re.sub(r"[\t\r\f\v ]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n\n", value)
        return value.strip()


def html_to_text(source: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(source)
    return parser.text()


class _HTMLFragmentExtractor(HTMLParser):
    """Extract authored block elements without flattening them together."""

    block_tags = {"p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.fragments: list[tuple[str, str]] = []
        self._tag = ""
        self._parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "svg"}:
            self._ignored += 1
        elif not self._ignored and tag in self.block_tags and not self._tag:
            self._tag = tag
            self._parts = []
        elif not self._ignored and tag == "br" and self._tag:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "svg"} and self._ignored:
            self._ignored -= 1
        elif tag == self._tag:
            text = html.unescape("".join(self._parts))
            text = re.sub(r"[\t\r\f\v ]+", " ", text)
            text = re.sub(r" *\n *", "\n", text).strip()
            if text:
                self.fragments.append((tag, text))
            self._tag = ""
            self._parts = []

    def handle_data(self, data: str) -> None:
        if not self._ignored and self._tag:
            self._parts.append(data)


def source_fragments(source_html: str, fallback_text: str) -> list[ImportedSourceFragment]:
    blocks: list[tuple[str, str]] = []
    if source_html:
        parser = _HTMLFragmentExtractor()
        parser.feed(source_html)
        blocks = parser.fragments
    if not blocks:
        blocks = [
            ("paragraph", paragraph.strip())
            for paragraph in re.split(r"\n\s*\n+", fallback_text)
            if paragraph.strip()
        ]
    result: list[ImportedSourceFragment] = []
    cursor = 0
    for position, (kind, text) in enumerate(blocks):
        end = cursor + len(text)
        result.append(ImportedSourceFragment(position, kind, text, cursor, end))
        cursor = end
    return result


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "shift_jis", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_GENERIC_CHAPTER_TITLE = re.compile(r"^章节\s*\d+$")


def infer_epub_chapter_title(source_html: str, text: str, fallback: str) -> str:
    """Find a useful title even when an EPUB styles headings as ordinary paragraphs."""
    heading = re.search(r"<h[1-6][^>]*>(.*?)</h[1-6]>", source_html, re.I | re.S)
    if heading:
        value = html_to_text(heading.group(1)).strip()
        if value:
            return value
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line and len(first_line) <= 160:
        return first_line
    document_title = re.search(r"<title[^>]*>(.*?)</title>", source_html, re.I | re.S)
    if document_title:
        value = html_to_text(document_title.group(1)).strip()
        if value:
            return value
    return fallback


def display_chapter_title(title: str, source_html: str) -> str:
    """Improve only generated numeric titles, preserving authored/user-edited names."""
    if not _GENERIC_CHAPTER_TITLE.fullmatch(title.strip()) or not source_html:
        return title
    text = html_to_text(source_html)
    return infer_epub_chapter_title(source_html, text, title)


def import_txt(path: str | Path) -> list[ImportedChapter]:
    source = Path(path)
    text = decode_text(source.read_bytes()).replace("\r\n", "\n")
    text = text.strip()
    return [ImportedChapter(source.stem, text, fragments=source_fragments("", text))]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def import_epub(path: str | Path) -> list[ImportedChapter]:
    """Read EPUB spine using only the standard library.

    Keeping this fallback avoids making ebooklib mandatory for opening projects.
    """
    source = Path(path)
    chapters: list[ImportedChapter] = []
    with zipfile.ZipFile(source) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(
            node.attrib["full-path"]
            for node in container.iter()
            if _local_name(node.tag) == "rootfile"
        )
        opf = ElementTree.fromstring(archive.read(rootfile))
        base = Path(rootfile).parent
        manifest: dict[str, str] = {}
        spine: list[str] = []
        for node in opf.iter():
            name = _local_name(node.tag)
            if name == "item" and "id" in node.attrib and "href" in node.attrib:
                manifest[node.attrib["id"]] = node.attrib["href"]
            elif name == "itemref" and "idref" in node.attrib:
                spine.append(node.attrib["idref"])
        for index, item_id in enumerate(spine, 1):
            href = manifest.get(item_id)
            if not href:
                continue
            member = (base / href).as_posix()
            try:
                raw = archive.read(member)
            except KeyError:
                continue
            source_html = decode_text(raw)
            text = html_to_text(source_html)
            if not text:
                continue
            title = infer_epub_chapter_title(source_html, text, f"章节 {index}")
            chapters.append(ImportedChapter(
                title or f"章节 {index}", text, source_html, source_fragments(source_html, text)
            ))
    merged: list[ImportedChapter] = []
    index = 0
    while index < len(chapters):
        current = chapters[index]
        if current.text.strip().casefold() == "cover":
            index += 1
            continue
        if index + 1 < len(chapters):
            following = chapters[index + 1]
            current_key = normalize_for_match(current.text)
            following_first = next((line.strip() for line in following.text.splitlines() if line.strip()), "")
            if (
                0 < len(current.text) <= 200
                and len(following.text) >= 5 * len(current.text)
                and current_key
                and current_key == normalize_for_match(following_first)
            ):
                following.title = current.title
                merged.append(following)
                index += 2
                continue
        merged.append(current)
        index += 1
    chapters = merged
    if not chapters:
        raise ValueError("EPUB 中没有可读取的正文")
    return chapters


def import_book(path: str | Path) -> list[ImportedChapter]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".txt":
        return import_txt(source)
    if suffix == ".epub":
        return import_epub(source)
    raise ValueError(f"不支持的文本格式：{suffix}")


_STRONG_END = set("。！？!?…⋯")
_WEAK_END = set("，,、；;：:")
_CLOSERS = set("”’」』】》）)]}\"")
_OPENING_QUOTES = set("‘“「『《〈«‹\"")
_CURSOR_TRAILING_PUNCTUATION = _STRONG_END | _WEAK_END | _CLOSERS | {"."}


def _has_text(value: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in value)


def _measure(value: str, cjk: bool) -> int:
    if cjk:
        return sum(1 for char in value if not char.isspace())
    return len(re.findall(r"\b[\w'’-]+\b", value, re.UNICODE))


def _strong_sentence_pieces(paragraph: str) -> list[tuple[str, int, int]]:
    pieces: list[tuple[str, int, int]] = []
    start = 0
    index = 0
    while index < len(paragraph):
        char = paragraph[index]
        boundary = char in _STRONG_END
        if char == ".":
            boundary = (
                index + 1 == len(paragraph)
                or paragraph[index + 1].isspace()
                or paragraph[index + 1] in _OPENING_QUOTES
            )
        if boundary:
            end = index + 1
            while end < len(paragraph) and paragraph[end] in _CLOSERS:
                end += 1
            if (
                char in _STRONG_END
                or end == len(paragraph)
                or paragraph[end].isspace()
                or paragraph[end] in _OPENING_QUOTES
            ):
                raw = paragraph[start:end]
                left = len(raw) - len(raw.lstrip())
                right = len(raw.rstrip())
                if right > left:
                    pieces.append((raw[left:right], start + left, start + right))
                start = end
                while start < len(paragraph) and paragraph[start].isspace():
                    start += 1
                index = start
                continue
        index += 1
    if start < len(paragraph):
        raw = paragraph[start:]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            pieces.append((raw[left:right], start + left, start + right))
    return pieces


def preferred_split_offset(text: str, approximate: int, *, radius: int | None = None) -> int:
    """Choose a readable boundary near an audio-derived character position."""
    if len(text) < 2:
        return 0
    approximate = max(1, min(len(text) - 1, approximate))
    radius = radius or max(12, min(48, len(text) // 3))
    low, high = max(1, approximate - radius), min(len(text) - 1, approximate + radius)
    candidates: list[tuple[int, int]] = []
    for index in range(low, high + 1):
        previous = text[index - 1]
        if previous in _WEAK_END:
            priority = 0
        elif previous in _STRONG_END or previous == "\n":
            priority = 1
        elif previous.isspace():
            priority = 2
        else:
            continue
        if _has_text(text[:index]) and _has_text(text[index:]):
            candidates.append((priority, index))
    if not candidates:
        return approximate
    priority = min(value for value, _index in candidates)
    return min(
        (index for value, index in candidates if value == priority),
        key=lambda index: abs(index - approximate),
    )


def cursor_split_offset(text: str, offset: int) -> int:
    """Keep punctuation at the cursor with the left side of an explicit split."""
    if len(text) < 2:
        return 0
    offset = max(1, min(len(text) - 1, int(offset)))
    while offset < len(text) and text[offset] in _CURSOR_TRAILING_PUNCTUATION:
        offset += 1
    return offset if _has_text(text[:offset]) and _has_text(text[offset:]) else 0


def _balanced_piece(text: str, base: int, target: int, cjk: bool) -> list[tuple[str, int, int]]:
    if _measure(text, cjk) <= target:
        return [(text, base, base + len(text))]
    result: list[tuple[str, int, int]] = []
    remaining = text
    offset = base
    while _measure(remaining, cjk) > target:
        if cjk:
            approximate = min(len(remaining) - 1, target)
        else:
            matches = list(re.finditer(r"\b[\w'’-]+\b", remaining, re.UNICODE))
            approximate = matches[min(target, len(matches)) - 1].end() if matches else min(len(remaining) - 1, target)
        split = preferred_split_offset(remaining, approximate)
        if split <= 0 or split >= len(remaining):
            break
        raw_left = remaining[:split]
        left = raw_left.strip()
        leading = len(raw_left) - len(raw_left.lstrip())
        if not _has_text(left):
            break
        left_start = offset + leading
        result.append((left, left_start, left_start + len(left)))
        consumed = split
        while consumed < len(remaining) and remaining[consumed].isspace():
            consumed += 1
        offset += consumed
        remaining = remaining[consumed:]
    if remaining.strip():
        leading = len(remaining) - len(remaining.lstrip())
        value = remaining.strip()
        result.append((value, offset + leading, offset + leading + len(value)))
    return result


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split prose into balanced readable cues while retaining source offsets."""
    result: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n+|\Z)", text, re.S):
        paragraph = match.group(0)
        cjk = is_cjk_text(paragraph)
        target = 80 if cjk else 35
        for piece, start, _end in _strong_sentence_pieces(paragraph):
            result.extend(_balanced_piece(piece, match.start() + start, target, cjk))
    merged: list[tuple[str, int, int]] = []
    for value, start, end in result:
        if _has_text(value):
            merged.append((value, start, end))
        elif merged:
            previous, previous_start, _previous_end = merged[-1]
            merged[-1] = (previous + value, previous_start, end)
    return merged


def split_sentences(text: str) -> list[str]:
    return [value for value, _start, _end in split_sentences_with_offsets(text)]


def normalize_for_match(text: str, language: str = "auto") -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    chars: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if category.startswith(("P", "Z", "C")):
            continue
        chars.append(char)
    return "".join(chars)


def is_cjk_text(text: str) -> bool:
    if not text:
        return False
    cjk = sum(
        1
        for char in text
        if "\u3400" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff"
    )
    return cjk / max(1, len(text)) > 0.2

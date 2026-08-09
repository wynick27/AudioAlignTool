from __future__ import annotations

import html
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree


@dataclass(slots=True)
class ImportedChapter:
    title: str
    text: str
    source_html: str = ""


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
    return [ImportedChapter(source.stem, text.strip())]


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
            chapters.append(ImportedChapter(title or f"章节 {index}", text, source_html))
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


_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])(?:[”’」』】》）)])?\s*|(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡0-9])")


def split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    result: list[str] = []
    for paragraph in paragraphs:
        pieces = [piece.strip() for piece in _SENTENCE_END.split(paragraph) if piece.strip()]
        result.extend(pieces or [paragraph])
    return result


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

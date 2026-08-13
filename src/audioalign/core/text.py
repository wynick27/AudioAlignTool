from __future__ import annotations

import html
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from bs4 import BeautifulSoup, Tag

from .models import SourceDocumentKind


@dataclass(slots=True)
class ImportedChapter:
    title: str
    text: str
    source_html: str = ""
    fragments: list["ImportedSourceFragment"] = field(default_factory=list)
    source_kind: SourceDocumentKind = SourceDocumentKind.TXT
    entry_path: str = ""
    selector: str = ""
    source_parts: list["ImportedSourcePart"] = field(default_factory=list)


@dataclass(slots=True)
class ImportedSourcePart:
    entry_path: str
    selector: str
    source_html: str


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
        if tag in {"head", "script", "style", "svg"}:
            self._ignored += 1
        elif not self._ignored and tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"head", "script", "style", "svg"} and self._ignored:
            self._ignored -= 1
        elif not self._ignored and tag in self.block_tags:
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
    return [ImportedChapter(
        source.stem, text, fragments=source_fragments("", text),
        source_kind=SourceDocumentKind.TXT, entry_path=source.name,
    )]


def _markdown_renderer():
    try:
        from markdown_it import MarkdownIt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("导入 Markdown 需要 markdown-it-py") from exc
    return MarkdownIt("commonmark", {"html": False})


def _markdown_sections(value: str, fallback_title: str) -> list[tuple[str, str]]:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    headings: list[int] = []
    fenced = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
        if not fenced and re.match(r"^#\s+\S", line):
            headings.append(index)
    if not headings:
        return [(fallback_title, "\n".join(lines).strip())]
    result: list[tuple[str, str]] = []
    if any(line.strip() for line in lines[: headings[0]]):
        result.append((fallback_title, "\n".join(lines[: headings[0]]).strip()))
    for position, start in enumerate(headings):
        end = headings[position + 1] if position + 1 < len(headings) else len(lines)
        title = re.sub(r"^#\s+", "", lines[start]).strip().rstrip("#").strip()
        result.append((title or f"章节 {position + 1}", "\n".join(lines[start:end]).strip()))
    return result


def import_markdown(path: str | Path) -> list[ImportedChapter]:
    source = Path(path)
    value = decode_text(source.read_bytes())
    renderer = _markdown_renderer()
    chapters: list[ImportedChapter] = []
    for index, (title, markdown) in enumerate(_markdown_sections(value, source.stem)):
        rendered = renderer.render(markdown)
        text = html_to_text(rendered)
        if not text:
            continue
        chapters.append(ImportedChapter(
            title, text, rendered, source_fragments(rendered, text),
            SourceDocumentKind.MARKDOWN, source.name, f"markdown-section:{index}",
        ))
    if not chapters:
        raise ValueError("Markdown 中没有可读取的正文")
    return chapters


def import_markdown_as_chapter(path: str | Path) -> ImportedChapter:
    source = Path(path)
    rendered = _markdown_renderer().render(decode_text(source.read_bytes()))
    text = html_to_text(rendered)
    if not text:
        raise ValueError(f"Markdown 中没有可读取的正文：{source.name}")
    return ImportedChapter(
        source.stem, text, rendered, source_fragments(rendered, text),
        SourceDocumentKind.MARKDOWN, source.name, "markdown-document",
    )


def _html_document_for_nodes(soup: BeautifulSoup, nodes: list[Tag]) -> str:
    head = str(soup.head) if soup.head else "<head><meta charset=\"utf-8\"></head>"
    body = "".join(str(node) for node in nodes)
    return f"<!doctype html><html>{head}<body>{body}</body></html>"


def import_html(path: str | Path) -> list[ImportedChapter]:
    source = Path(path)
    raw = decode_text(source.read_bytes())
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all(["script", "noscript"]):
        tag.decompose()
    articles = list(soup.find_all("article"))
    chapters: list[ImportedChapter] = []
    if articles:
        groups = [([article], f"article:nth-of-type({index + 1})") for index, article in enumerate(articles)]
    else:
        body = soup.body or soup
        children = [node for node in body.children if isinstance(node, Tag)]
        heading_positions = [index for index, node in enumerate(children) if node.name == "h1"]
        groups = []
        if heading_positions:
            if heading_positions[0] > 0:
                groups.append((children[: heading_positions[0]], "html-preamble"))
            for group_index, start in enumerate(heading_positions):
                end = heading_positions[group_index + 1] if group_index + 1 < len(heading_positions) else len(children)
                groups.append((children[start:end], f"h1-section:{group_index}"))
        else:
            groups = [(children, "body")]
    for index, (nodes, selector) in enumerate(groups):
        if not nodes:
            continue
        document = _html_document_for_nodes(soup, nodes)
        text = html_to_text(document)
        if not text:
            continue
        heading = next((node.get_text(" ", strip=True) for node in nodes if node.name == "h1"), "")
        if not heading:
            heading = next((node.get_text(" ", strip=True) for node in nodes if node.name in {"h2", "h3"}), "")
        title = heading or (soup.title.get_text(" ", strip=True) if soup.title and index == 0 else "")
        chapters.append(ImportedChapter(
            title or (source.stem if len(groups) == 1 else f"章节 {index + 1}"),
            text, document, source_fragments(document, text),
            SourceDocumentKind.HTML, source.name, selector,
        ))
    if not chapters:
        raise ValueError("HTML 中没有可读取的正文")
    return chapters


def import_html_as_chapter(path: str | Path) -> ImportedChapter:
    source = Path(path)
    raw = decode_text(source.read_bytes())
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all(["script", "noscript"]):
        tag.decompose()
    document = str(soup)
    text = html_to_text(document)
    if not text:
        raise ValueError(f"HTML 中没有可读取的正文：{source.name}")
    title = soup.title.get_text(" ", strip=True) if soup.title else source.stem
    return ImportedChapter(
        title or source.stem, text, document, source_fragments(document, text),
        SourceDocumentKind.HTML, source.name, "body",
    )


_RESOURCE_ATTRS = {"link": "href", "img": "src", "source": "src", "audio": "src", "video": "src"}
_CSS_URL = re.compile(r"(?:url\(\s*|@import\s+)(?:['\"])?([^)'\"\s;]+)", re.I)


def collect_local_html_resources(path: str | Path) -> tuple[list[Path], list[str]]:
    """Return referenced files confined to the HTML document directory."""
    source = Path(path).resolve()
    root = source.parent
    pending: list[Path] = []
    warnings: list[str] = []
    soup = BeautifulSoup(decode_text(source.read_bytes()), "html.parser")
    for tag_name, attribute in _RESOURCE_ATTRS.items():
        for tag in soup.find_all(tag_name):
            value = tag.get(attribute)
            if value:
                pending.append(Path(str(value).split("#", 1)[0].split("?", 1)[0]))
    copied: dict[Path, None] = {}
    index = 0
    while index < len(pending):
        reference = pending[index]
        index += 1
        if not reference or reference.as_posix().startswith(("data:", "http:", "https:", "//")):
            continue
        candidate = (root / reference).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            warnings.append(f"资源越出 HTML 目录，已跳过：{reference}")
            continue
        if not candidate.is_file() or candidate in copied or candidate == source:
            continue
        copied[candidate] = None
        if candidate.suffix.lower() == ".css":
            css = decode_text(candidate.read_bytes())
            css_root = candidate.parent
            for match in _CSS_URL.finditer(css):
                value = match.group(1)
                if value.startswith(("data:", "http:", "https:", "//")):
                    continue
                nested = (css_root / value).resolve()
                try:
                    pending.append(nested.relative_to(root))
                except ValueError:
                    warnings.append(f"CSS 资源越出 HTML 目录，已跳过：{value}")
    return list(copied), warnings


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")


def collect_local_markdown_resources(path: str | Path) -> tuple[list[Path], list[str]]:
    source = Path(path).resolve()
    root = source.parent
    resources: list[Path] = []
    warnings: list[str] = []
    for match in _MARKDOWN_LINK.finditer(decode_text(source.read_bytes())):
        value = match.group(1).strip("<>").split("#", 1)[0].split("?", 1)[0]
        if not value or value.startswith(("data:", "http:", "https:", "//", "mailto:")):
            continue
        candidate = (root / value).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            warnings.append(f"Markdown 资源越出文档目录，已跳过：{value}")
            continue
        if candidate.is_file() and candidate not in resources:
            resources.append(candidate)
    return resources, warnings


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
                title or f"章节 {index}", text, source_html, source_fragments(source_html, text),
                SourceDocumentKind.EPUB, member, f"spine:{index - 1}",
                [ImportedSourcePart(member, f"spine:{index - 1}", source_html)],
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
            duplicate_title_page = (
                0 < len(current.text) <= 200
                and len(following.text) >= 5 * len(current.text)
                and current_key
                and current_key == normalize_for_match(following_first)
            )
            chapter_label_page = (
                0 < len(current.text) <= 120
                and len(following.text) >= 5 * len(current.text)
                and current_key.startswith("chapter")
                and len(current.fragments) <= 3
                and all(fragment.kind.startswith("h") for fragment in current.fragments)
            )
            if duplicate_title_page or chapter_label_page:
                if chapter_label_page:
                    separator = "\n\n"
                    offset = len(current.text) + len(separator)
                    following_fragments = [
                        ImportedSourceFragment(
                            fragment.position + len(current.fragments), fragment.kind, fragment.text,
                            fragment.source_start_char + offset, fragment.source_end_char + offset,
                        )
                        for fragment in following.fragments
                    ]
                    combined_title = f"{current.title} — {following.title}"
                    combined_html = _combine_html_documents(current.source_html, following.source_html)
                    merged.append(ImportedChapter(
                        combined_title, current.text + separator + following.text, combined_html,
                        [*current.fragments, *following_fragments], SourceDocumentKind.EPUB,
                        current.entry_path, current.selector,
                        [*current.source_parts, *following.source_parts],
                    ))
                else:
                    following.title = current.title
                    following.source_parts = [*current.source_parts, *following.source_parts]
                    merged.append(following)
                index += 2
                continue
        merged.append(current)
        index += 1
    chapters = merged
    if not chapters:
        raise ValueError("EPUB 中没有可读取的正文")
    return chapters


def _combine_html_documents(left: str, right: str) -> str:
    """Create one display document while retaining both original source pages."""
    left_soup = BeautifulSoup(left, "html.parser")
    right_soup = BeautifulSoup(right, "html.parser")
    output = BeautifulSoup(str(right_soup), "html.parser")
    body = output.body or output
    body.clear()
    for source, name in ((left_soup, "0"), (right_soup, "1")):
        section = output.new_tag("section")
        section["data-aat-source-part"] = name
        source_body = source.body or source
        for child in list(source_body.contents):
            section.append(BeautifulSoup(str(child), "html.parser"))
        body.append(section)
    return str(output)


def import_book(path: str | Path, *, one_chapter: bool = False) -> list[ImportedChapter]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".txt":
        return import_txt(source)
    if suffix in {".md", ".markdown"}:
        return [import_markdown_as_chapter(source)] if one_chapter else import_markdown(source)
    if suffix in {".html", ".htm"}:
        return [import_html_as_chapter(source)] if one_chapter else import_html(source)
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

from __future__ import annotations

import mimetypes
import html
import shutil
import zipfile
from bisect import bisect_left
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineUrlRequestInterceptor,
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
    QWebEngineSettings,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from audioalign.core.models import SourceDocumentKind, TextSegment
from audioalign.core.epub_media_overlay import _fuzzy_projection_match, _match_projection
from audioalign.core.storage import ProjectSession
from audioalign.core.text import html_to_text, normalize_for_match


_SCHEME_REGISTERED = False


def register_book_scheme() -> None:
    global _SCHEME_REGISTERED
    if _SCHEME_REGISTERED:
        return
    scheme = QWebEngineUrlScheme(b"aatbook")
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.HostAndPort)
    scheme.setDefaultPort(0)
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.LocalScheme
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.ContentSecurityPolicyIgnored
    )
    QWebEngineUrlScheme.registerScheme(scheme)
    _SCHEME_REGISTERED = True


class _BookRequestInterceptor(QWebEngineUrlRequestInterceptor):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root: Path | None = None

    def set_root(self, root: Path | None) -> None:
        self.root = root.resolve() if root else None

    def interceptRequest(self, info) -> None:  # noqa: N802 - Qt API
        url = info.requestUrl()
        if url.scheme() == "file":
            if self.root is None:
                info.block(True)
                return
            candidate = Path(url.toLocalFile()).resolve()
            try:
                candidate.relative_to(self.root)
            except ValueError:
                info.block(True)
            return
        if url.scheme() not in {"aatbook", "qrc", "data", "about"}:
            info.block(True)


class _BookSchemeHandler(QWebEngineUrlSchemeHandler):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.root: Path | None = None

    def set_root(self, root: Path | None) -> None:
        self.root = root.resolve() if root else None

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802 - Qt API
        if self.root is None:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        relative = unquote(job.requestUrl().path()).lstrip("/")
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
            return
        if not candidate.is_file():
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        buffer = QBuffer(job)
        buffer.setData(QByteArray(candidate.read_bytes()))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        job.reply(mime.encode("ascii", errors="ignore"), buffer)


class _BookBridge(QObject):
    activated = Signal(int, bool)

    @Slot(str, bool)
    def activateSegment(self, segment_id: str, jump: bool = False) -> None:  # noqa: N802 - JS API
        try:
            self.activated.emit(int(segment_id), bool(jump))
        except ValueError:
            return


def _safe_extract_epub(source: Path, target: Path) -> None:
    marker = target / ".source-fingerprint"
    signature = f"{source.stat().st_size}:{source.stat().st_mtime_ns}"
    if marker.exists() and marker.read_text("utf-8") == signature:
        return
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            destination = (target / item.filename).resolve()
            try:
                destination.relative_to(target.resolve())
            except ValueError as exc:
                raise ValueError(f"EPUB contains an unsafe path: {item.filename}") from exc
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as input_handle, destination.open("wb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle)
    marker.write_text(signature, encoding="utf-8")


def _normalized_with_mapping(nodes: list[NavigableString]) -> tuple[str, list[tuple[int, int] | None]]:
    characters: list[str] = []
    mapping: list[tuple[int, int] | None] = []
    whitespace = False
    for node_index, node in enumerate(nodes):
        for offset, character in enumerate(str(node)):
            if character.isspace():
                whitespace = True
                continue
            if whitespace and characters:
                characters.append(" ")
                mapping.append(None)
            whitespace = False
            characters.append(character)
            mapping.append((node_index, offset))
    return "".join(characters), mapping


def annotate_html(source_html: str, segments: list[TextSegment], base_href: str) -> str:
    soup = BeautifulSoup(source_html, "html.parser")
    for tag in soup.find_all(["script", "noscript"]):
        tag.decompose()
    for meta in soup.find_all("meta"):
        if str(meta.get("http-equiv", "")).casefold() == "content-security-policy":
            meta.decompose()
    if soup.head is None:
        head = soup.new_tag("head")
        if soup.html:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)
    base = soup.new_tag("base", href=base_href)
    soup.head.insert(0, base)
    style = soup.new_tag("style")
    style.string = (
        "[data-aat-segment]{cursor:pointer;}"
        "[data-aat-segment].aat-active{background:rgba(255,214,64,.52)!important;"
        "box-shadow:inset 0 -2px rgba(226,145,0,.7);border-radius:.15em;}"
    )
    soup.head.append(style)

    nodes = [
        node for node in soup.find_all(string=True)
        if node.parent and node.parent.name not in {"style", "script", "title", "svg"}
    ]
    document, mapping = _normalized_with_mapping(nodes)
    projected_document, projected_offsets = _match_projection(document)
    cursor = 0
    ranges: dict[int, list[tuple[int, int, str]]] = {}
    for segment in segments:
        if segment.id is None:
            continue
        target = " ".join(segment.text.split())
        if not target:
            continue
        start = document.find(target, cursor)
        if start < 0:
            projected_target, _unused = _match_projection(target)
            projected_cursor = bisect_left(projected_offsets, cursor)
            projected_start = (
                projected_document.find(projected_target, projected_cursor)
                if projected_target else -1
            )
            if projected_start < 0 and projected_target:
                fuzzy = _fuzzy_projection_match(
                    projected_document, projected_target, projected_cursor,
                )
                if fuzzy:
                    projected_start, projected_end = fuzzy
                else:
                    projected_end = -1
            else:
                projected_end = projected_start + len(projected_target)
            if projected_start < 0 or projected_end <= projected_start:
                continue
            start = projected_offsets[projected_start]
            end = projected_offsets[projected_end - 1] + 1
        else:
            end = start + len(target)
        cursor = end
        per_node: dict[int, list[int]] = {}
        for item in mapping[start:end]:
            if item is not None:
                per_node.setdefault(item[0], []).append(item[1])
        for node_index, offsets in per_node.items():
            ranges.setdefault(node_index, []).append((min(offsets), max(offsets) + 1, str(segment.id)))

    for node_index in sorted(ranges, reverse=True):
        node = nodes[node_index]
        value = str(node)
        intervals = sorted(ranges[node_index], key=lambda item: item[0])
        replacements = []
        offset = 0
        for start, end, segment_id in intervals:
            if start > offset:
                replacements.append(NavigableString(value[offset:start]))
            span = soup.new_tag("span")
            span["data-aat-segment"] = segment_id
            span.string = value[start:end]
            replacements.append(span)
            offset = end
        if offset < len(value):
            replacements.append(NavigableString(value[offset:]))
        node.replace_with(*replacements)

    script = soup.new_tag("script", src="qrc:///qtwebchannel/qwebchannel.js")
    soup.head.append(script)
    bridge_script = soup.new_tag("script")
    bridge_script.string = r"""
      let aatBridge=null,aatFollow=true;
      new QWebChannel(qt.webChannelTransport,function(channel){aatBridge=channel.objects.aatBridge;});
      document.addEventListener('click',function(event){
        const selection=window.getSelection();
        if(selection && !selection.isCollapsed)return;
        const target=event.target.closest('[data-aat-segment]');
        if(target && aatBridge)aatBridge.activateSegment(target.dataset.aatSegment,false);
      });
      document.addEventListener('dblclick',function(event){
        const target=event.target.closest('[data-aat-segment]');
        if(target && aatBridge)aatBridge.activateSegment(target.dataset.aatSegment,true);
      });
      document.addEventListener('wheel',function(){aatFollow=false;},{passive:true});
      window.aatSetActive=function(id,follow){
        document.querySelectorAll('.aat-active').forEach(e=>e.classList.remove('aat-active'));
        const targets=document.querySelectorAll('[data-aat-segment="'+id+'"]');
        targets.forEach(target=>target.classList.add('aat-active'));
        if(targets.length&&follow&&aatFollow)targets[0].scrollIntoView({block:'center'});
      };
      window.aatResumeFollow=function(){aatFollow=true;};
    """
    soup.head.append(bridge_script)
    return str(soup)


class OriginalBookView(QWebEngineView):
    segmentActivated = Signal(int, int, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._current_segment_id: int | None = None
        self._current_ensure_visible = True
        self._page_ready = False
        self._fallback_html = ""
        self._fallback_path: Path | None = None
        self._fallback_attempted = False
        self._handler = _BookSchemeHandler(self)
        profile = QWebEngineProfile(self)
        profile.installUrlSchemeHandler(b"aatbook", self._handler)
        self._interceptor = _BookRequestInterceptor(profile)
        profile.setUrlRequestInterceptor(self._interceptor)
        self.setPage(QWebEnginePage(profile, self))
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True,
        )
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False,
        )
        self._bridge = _BookBridge(self)
        self._bridge.activated.connect(
            lambda segment_id, jump: self.segmentActivated.emit(
                segment_id, self._generation, jump,
            )
        )
        self._channel = QWebChannel(self.page())
        self._channel.registerObject("aatBridge", self._bridge)
        self.page().setWebChannel(self._channel)
        self.loadFinished.connect(self._load_finished)

    def clear_book(self, generation: int = 0) -> None:
        self._generation = generation
        self._handler.set_root(None)
        self._interceptor.set_root(None)
        self._fallback_html = ""
        self._fallback_path = None
        self._fallback_attempted = False
        self._current_segment_id = None
        self._page_ready = False
        self.setHtml("<html><body></body></html>")

    def _load_finished(self, ok: bool) -> None:
        if ok:
            self._page_ready = True
            self._apply_focus()
            return
        self._page_ready = False
        if self._fallback_attempted or self._fallback_path is None:
            return
        self._fallback_attempted = True
        self.setUrl(QUrl.fromLocalFile(str(self._fallback_path.resolve())))

    def set_chapter(
        self, session: ProjectSession, chapter_id: int, segments: list[TextSegment], generation: int,
    ) -> None:
        self._generation = generation
        self._page_ready = False
        self._handler.set_root(session.root)
        self._interceptor.set_root(session.root)
        chapter = next((item for item in session.repository.chapters() if item.id == chapter_id), None)
        if chapter is None:
            self.clear_book(generation)
            return
        source_html = chapter.source_html
        base_relative = Path("cache")
        source_parts = session.repository.chapter_source_parts(chapter_id)
        source_info = session.repository.chapter_source_document(chapter_id)
        if source_parts:
            source_info = source_parts[0]
        if source_info:
            document, entry_path, _selector = source_info
            stored = session.root / document.stored_path
            if document.kind == SourceDocumentKind.EPUB and stored.exists():
                extracted = session.root / "cache" / "book-view" / document.fingerprint[:16]
                _safe_extract_epub(stored, extracted)
                entry = extracted / entry_path
                if len(source_parts) <= 1 and entry.is_file():
                    source_html = entry.read_text("utf-8", errors="replace")
                if entry.is_file():
                    base_relative = entry.parent.relative_to(session.root)
            elif document.resource_root:
                base_relative = Path(document.resource_root)
        elif session.manifest.source_name:
            candidates = [
                session.root / session.manifest.source_name,
                session.root / "source" / session.manifest.source_name,
            ]
            candidates.extend(session.root.glob(f"source/**/{Path(session.manifest.source_name).name}"))
            epub = next((item for item in candidates if item.is_file() and item.suffix.lower() == ".epub"), None)
            if epub:
                extracted = session.root / "cache" / "book-view" / f"legacy-{epub.stat().st_size:x}"
                _safe_extract_epub(epub, extracted)
                target = normalize_for_match(html_to_text(chapter.source_html))[:1200]
                scored: list[tuple[float, Path]] = []
                for entry in list(extracted.rglob("*.xhtml")) + list(extracted.rglob("*.html")):
                    candidate_html = entry.read_text("utf-8", errors="replace")
                    candidate = normalize_for_match(html_to_text(candidate_html))[:1200]
                    scored.append((SequenceMatcher(None, target, candidate).ratio(), entry))
                if scored:
                    score, entry = max(scored, key=lambda item: item[0])
                    if score >= 0.45:
                        source_html = entry.read_text("utf-8", errors="replace")
                        base_relative = entry.parent.relative_to(session.root)
        if not source_html:
            body = "".join(f"<p>{html.escape(segment.text)}</p>" for segment in segments)
            source_html = (
                "<!doctype html><html><head><meta charset='utf-8'><style>"
                "body{max-width:52rem;margin:2rem auto;padding:0 1.5rem;font:18px/1.9 serif}"
                "</style></head><body>" + body + "</body></html>"
            )
        base_path = (session.root / base_relative).resolve()
        try:
            base_path.relative_to(session.root.resolve())
        except ValueError:
            base_path = session.root.resolve()
        # The rendered page is itself a project-local file. Keeping CSS,
        # fonts and images on the same restricted file origin is reliable on
        # Qt WebEngine builds that reject file -> custom-scheme subresources.
        base_url = QUrl.fromLocalFile(base_path.as_posix().rstrip("/") + "/").toString()
        rendered = annotate_html(source_html, segments, base_url)
        rendered_soup = BeautifulSoup(rendered, "html.parser")
        rendered_body = rendered_soup.body
        has_visible_source = bool(
            rendered_body
            and (
                rendered_body.get_text(strip=True)
                or rendered_body.find(["img", "svg", "image", "video", "canvas"])
            )
        )
        if not has_visible_source:
            body = "".join(f"<p>{html.escape(segment.text)}</p>" for segment in segments)
            rendered = annotate_html(
                "<!doctype html><html><head><meta charset='utf-8'><style>"
                "body{max-width:52rem;margin:2rem auto;padding:0 1.5rem;font:18px/1.9 serif}"
                "</style></head><body>" + body + "</body></html>",
                segments,
                base_url,
            )
        output = session.root / "cache" / "book-view" / "rendered" / f"chapter-{chapter_id}.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".html.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
        fallback_body = "".join(
            f"<p>{html.escape(segment.text)}</p>" for segment in segments
        ) or "<p>本章节没有可显示的正文。</p>"
        self._fallback_html = (
            "<!doctype html><html><head><meta charset='utf-8'><style>"
            "body{max-width:52rem;margin:2rem auto;padding:0 1.5rem;font:18px/1.9 serif}"
            "</style></head><body>" + fallback_body + "</body></html>"
        )
        self._fallback_path = output.with_name(f"chapter-{chapter_id}-fallback.html")
        self._fallback_path.write_text(self._fallback_html, encoding="utf-8")
        self._fallback_attempted = False
        # Some Qt WebEngine builds leave a custom-scheme top-level navigation
        # at about:blank.  Load the generated page as a project-local file;
        # the interceptor restricts file access to this project, while EPUB
        # CSS/images still resolve through the restricted aatbook scheme.
        self.setUrl(QUrl.fromLocalFile(str(output.resolve())))

    def focus_segment(self, segment_id: int | None, *, ensure_visible: bool = True) -> None:
        if segment_id is None:
            return
        self._current_segment_id = segment_id
        self._current_ensure_visible = bool(ensure_visible)
        if self._page_ready:
            self._apply_focus()

    def _apply_focus(self) -> None:
        if self._current_segment_id is None:
            return
        self.page().runJavaScript(
            "window.aatSetActive && window.aatSetActive("
            f"{self._current_segment_id!r},{str(self._current_ensure_visible).lower()});"
        )

    def resume_follow(self) -> None:
        self.page().runJavaScript("window.aatResumeFollow && window.aatResumeFollow();")

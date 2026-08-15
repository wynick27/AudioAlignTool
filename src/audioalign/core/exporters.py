from __future__ import annotations

import html
import json
import re
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from .alignment import enforce_monotonic
from .models import SegmentOverlapPolicy, SegmentStatus, SourceDocumentKind, TextSegment
from .storage import ProjectSession
from .timecode import format_srt_time, format_time_ms


def _safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value).strip(" .")
    return name[:80] or fallback


def _srt_time(milliseconds: int) -> str:
    return format_srt_time(milliseconds)


def _vtt_time(milliseconds: int) -> str:
    return format_time_ms(milliseconds)


def export_subtitles(session: ProjectSession, output: str | Path, kind: str) -> list[Path]:
    kind = kind.lower()
    if kind not in {"srt", "vtt"}:
        raise ValueError("Subtitle kind must be srt or vtt")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for chapter in session.repository.chapters():
        source_segments = [s for s in session.repository.segments(chapter.id or 0) if s.end_ms > s.start_ms]
        conflicts = [
            index for index in range(len(source_segments) - 1)
            if source_segments[index].end_ms > source_segments[index + 1].start_ms
        ]
        if conflicts and session.manifest.segment_overlap_policy == SegmentOverlapPolicy.ALLOW_OVERLAP:
            first = conflicts[0] + 1
            raise ValueError(f"章节“{chapter.title}”的第 {first}/{first + 1} 句存在时间重叠，请先修正")
        segments = enforce_monotonic(source_segments)
        lines: list[str] = ["WEBVTT", ""] if kind == "vtt" else []
        for index, segment in enumerate(segments, 1):
            if kind == "srt":
                lines.extend([str(index), f"{_srt_time(segment.start_ms)} --> {_srt_time(segment.end_ms)}", segment.text, ""])
            else:
                lines.extend([f"{_vtt_time(segment.start_ms)} --> {_vtt_time(segment.end_ms)}", segment.text, ""])
        path = target / f"{chapter.position + 1:03d}-{_safe_name(chapter.title, 'chapter')}.{kind}"
        path.write_text("\n".join(lines), encoding="utf-8-sig" if kind == "srt" else "utf-8")
        written.append(path)
    return written


def project_alignment_dict(session: ProjectSession) -> dict:
    chapters: list[dict] = []
    for chapter in session.repository.chapters():
        linked_audio: list[dict] = []
        for link in session.repository.chapter_links(chapter.id or 0):
            asset = session.repository.audio(link.audio_id)
            if not asset:
                continue
            linked_audio.append(
                {
                    "audio_id": asset.id,
                    "path": asset.relative_path or asset.absolute_path,
                    "duration_ms": asset.duration_ms,
                    "fingerprint": asset.fingerprint,
                    "source_start_ms": link.source_start_ms,
                    "source_end_ms": link.source_end_ms,
                    "position": link.position,
                }
            )
        chapters.append(
            {
                "id": chapter.id,
                "title": chapter.title,
                "position": chapter.position,
                "audio": linked_audio[0] if linked_audio else None,
                "audio_links": linked_audio,
                "segments": [
                    {
                        "id": segment.id,
                        "position": segment.position,
                        "text": segment.text,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "confidence": round(segment.confidence, 4),
                        "status": segment.status.value,
                        "locked": segment.locked,
                        "origin": segment.origin.value,
                        "source_fragment_id": segment.source_fragment_id,
                        "source_start_char": segment.source_start_char,
                        "source_end_char": segment.source_end_char,
                    }
                    for segment in session.repository.segments(chapter.id or 0)
                ],
            }
        )
    return {
        "schema_version": 2,
        "project_id": session.manifest.project_id,
        "title": session.manifest.title,
        "language": session.manifest.language,
        "alignment_mode": session.manifest.alignment_mode.value,
        "segment_overlap_policy": session.manifest.segment_overlap_policy.value,
        "chapters": chapters,
    }


def export_json(session: ProjectSession, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(project_alignment_dict(session), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _normalized_html_nodes(nodes: list[NavigableString]):
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


def _annotate_html_export_page(
    source_html: str, segments: list[TextSegment], base_href: str,
) -> str:
    """Preserve source markup while adding reader synchronization anchors."""
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
    if base_href:
        soup.head.insert(0, soup.new_tag("base", href=base_href))
    style = soup.new_tag("style")
    style.string = (
        "[data-aat-index]{cursor:pointer;}"
        "[data-aat-index].aat-active{background:rgba(255,214,64,.52)!important;"
        "box-shadow:inset 0 -2px rgba(226,145,0,.7);border-radius:.15em;}"
    )
    soup.head.append(style)

    nodes = [
        node for node in soup.find_all(string=True)
        if node.parent and node.parent.name not in {"style", "script", "title", "svg"}
    ]
    document, mapping = _normalized_html_nodes(nodes)
    cursor = 0
    ranges: dict[int, list[tuple[int, int, str]]] = {}
    for index, segment in enumerate(segments):
        target = " ".join(segment.text.split())
        if not target:
            continue
        start = document.find(target, cursor)
        if start < 0:
            start = document.find(target)
        if start < 0:
            continue
        end = start + len(target)
        cursor = end
        per_node: dict[int, list[int]] = {}
        for item in mapping[start:end]:
            if item is not None:
                per_node.setdefault(item[0], []).append(item[1])
        for node_index, offsets in per_node.items():
            ranges.setdefault(node_index, []).append(
                (min(offsets), max(offsets) + 1, str(index))
            )
    for node_index in sorted(ranges, reverse=True):
        node = nodes[node_index]
        value = str(node)
        replacements = []
        offset = 0
        for start, end, segment_index in sorted(ranges[node_index], key=lambda item: item[0]):
            if start > offset:
                replacements.append(NavigableString(value[offset:start]))
            span = soup.new_tag("span")
            span["data-aat-index"] = segment_index
            span.string = value[start:end]
            replacements.append(span)
            offset = end
        if offset < len(value):
            replacements.append(NavigableString(value[offset:]))
        node.replace_with(*replacements)

    script = soup.new_tag("script")
    script.string = r"""
      document.addEventListener('click',function(event){
        const selection=window.getSelection();
        if(selection && !selection.isCollapsed)return;
        const target=event.target.closest('[data-aat-index]');
        if(target)parent.postMessage({type:'aat-segment',index:Number(target.dataset.aatIndex)},'*');
      });
      addEventListener('message',function(event){
        if(!event.data || event.data.type!=='aat-active')return;
        document.querySelectorAll('.aat-active').forEach(e=>e.classList.remove('aat-active'));
        const targets=document.querySelectorAll('[data-aat-index="'+event.data.index+'"]');
        targets.forEach(target=>target.classList.add('aat-active'));
        if(targets.length&&event.data.follow)targets[0].scrollIntoView({block:'center',behavior:'smooth'});
      });
    """
    soup.head.append(script)
    return str(soup)


def _extract_epub_assets(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            destination = (target / item.filename).resolve()
            try:
                destination.relative_to(target.resolve())
            except ValueError as exc:
                raise ValueError(f"EPUB 包含不安全路径：{item.filename}") from exc
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as input_handle, destination.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle)


_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#e9e8e3;color:#242424;overflow:hidden}
header{height:74px;background:#fff;border-bottom:1px solid #ccc;padding:.45rem .75rem;display:flex;gap:.55rem;align-items:center;flex-wrap:wrap}
#book{display:block;width:100%;height:calc(100vh - 74px);border:0;background:white}#player{width:min(430px,38vw);height:38px}select,button,label{font:inherit}select,button{padding:.3rem .5rem}.time{min-width:5.5rem;font-variant-numeric:tabular-nums;color:#666}.grow{flex:1;min-width:.5rem}
@media(max-width:900px){header{height:118px}#book{height:calc(100vh - 118px)}#player{width:55vw}}
@media(prefers-color-scheme:dark){body{background:#171717;color:#eee}header{background:#222;border-color:#444}.time{color:#bbb}}
</style></head><body><header><strong>__TITLE__</strong><select id="chapters"></select><button id="previous" title="上一句">⏮</button><button id="play" title="播放/暂停">▶</button><button id="next" title="下一句">⏭</button><audio id="player" controls></audio><label>倍速 <select id="speed"><option value="0.25">0.25</option><option value="0.5">0.50</option><option value="0.75">0.75</option><option value="1" selected>1.00</option><option value="1.25">1.25</option><option value="1.5">1.50</option><option value="2">2.00</option><option value="2.5">2.50</option><option value="3">3.00</option></select>×</label><label><input id="loop" type="checkbox"> 单句循环</label><span class="time" id="clock">00:00:00</span><span class="grow"></span></header><iframe id="book" title="原书正文"></iframe>
<script id="alignment" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('alignment').textContent),select=document.getElementById('chapters'),player=document.getElementById('player'),playButton=document.getElementById('play'),book=document.getElementById('book'),clock=document.getElementById('clock'),speed=document.getElementById('speed'),loop=document.getElementById('loop');let active=-1,segments=[],links=[],linkIndex=0,lastSaved=0;const storageKey='AudioAlignTool:'+data.project_id;
data.chapters.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.title;select.appendChild(o)});
function localStart(i){return links.slice(0,i).reduce((n,x)=>n+x.source_end_ms-x.source_start_ms,0)}
function currentMs(){const x=links[linkIndex];return x?Math.max(0,localStart(linkIndex)+player.currentTime*1000-x.source_start_ms):0}
function save(force=false){const now=Date.now();if(!force&&now-lastSaved<500)return;lastSaved=now;try{localStorage.setItem(storageKey,JSON.stringify({chapter:+select.value,position_ms:Math.round(currentMs()),rate:player.playbackRate,loop:loop.checked}))}catch(_){}}
function syncPlayButton(){const playing=!player.paused&&!player.ended;playButton.textContent=playing?'⏸':'▶';playButton.title=playing?'暂停':'播放';playButton.setAttribute('aria-label',playButton.title)}
function postActive(follow=true){if(active>=0&&book.contentWindow)book.contentWindow.postMessage({type:'aat-active',index:active,follow},'*')}
function playAt(ms,autoplay=true){if(!links.length)return;let i=links.findIndex((x,n)=>ms<localStart(n)+x.source_end_ms-x.source_start_ms);if(i<0)i=links.length-1;const x=links[i],position=Math.max(x.source_start_ms,(x.source_start_ms+ms-localStart(i)))/1000,src=x.export_path||x.path;const apply=()=>{player.currentTime=position;player.playbackRate=Number(speed.value);if(autoplay)player.play().catch(()=>{});tick()};if(i!==linkIndex||player.getAttribute('src')!==src){linkIndex=i;player.src=src;player.onloadedmetadata=apply}else apply()}
function load(i,position=0){const c=data.chapters[i];select.value=i;segments=c.segments;links=c.audio_links||(c.audio?[c.audio]:[]);linkIndex=0;active=-1;book.src=c.page_path;player.src=links[0]?(links[0].export_path||links[0].path):'';player.onloadedmetadata=()=>playAt(position,false);if(!links.length)save(true)}
function activate(i,autoplay=true){if(!segments[i])return;active=i;postActive(true);playAt(segments[i].start_ms,autoplay)}
function tick(){const x=links[linkIndex],ms=currentMs();clock.textContent=new Date(Math.max(0,ms)).toISOString().slice(11,19);if(loop.checked&&active>=0&&segments[active]&&ms>=segments[active].end_ms-25){playAt(segments[active].start_ms,true);return}if(x&&player.currentTime*1000>=x.source_end_ms){if(linkIndex+1<links.length)playAt(localStart(linkIndex+1),!player.paused);else player.pause()}const i=segments.findIndex(s=>s.end_ms>s.start_ms&&ms>=s.start_ms&&ms<s.end_ms);if(i!==active){active=i;postActive(true)}save()}
addEventListener('message',e=>{if(e.data&&e.data.type==='aat-segment')activate(Number(e.data.index),true)});book.addEventListener('load',()=>postActive(false));select.onchange=()=>load(+select.value,0);speed.onchange=()=>{player.playbackRate=Number(speed.value);save(true)};loop.onchange=()=>save(true);player.ontimeupdate=tick;player.onplay=syncPlayButton;player.onpause=()=>{syncPlayButton();save(true)};player.onended=syncPlayButton;player.onemptied=syncPlayButton;playButton.onclick=()=>player.paused?player.play():player.pause();document.getElementById('previous').onclick=()=>activate(Math.max(0,(active<0?0:active)-1),true);document.getElementById('next').onclick=()=>activate(Math.min(segments.length-1,(active<0?-1:active)+1),true);addEventListener('beforeunload',()=>save(true));let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(_){};speed.value=String(saved.rate||1);if(!speed.value)speed.value='1';player.playbackRate=Number(speed.value);loop.checked=!!saved.loop;syncPlayButton();load(Math.max(0,Math.min(data.chapters.length-1,Number(saved.chapter)||0)),Math.max(0,Number(saved.position_ms)||0));
</script></body></html>"""


def export_html(session: ProjectSession, output: str | Path) -> Path:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    media_dir = target / "media"
    media_dir.mkdir(exist_ok=True)
    pages_dir = target / "pages"
    pages_dir.mkdir(exist_ok=True)
    book_dir = target / "book"
    book_dir.mkdir(exist_ok=True)
    data = project_alignment_dict(session)
    exported_assets: dict[int, str] = {}
    for chapter in data["chapters"]:
        for audio_info in chapter.get("audio_links", []):
            audio_id = int(audio_info.get("audio_id") or 0)
            if audio_id in exported_assets:
                audio_info["export_path"] = exported_assets[audio_id]
                continue
            asset = session.repository.audio(audio_id)
            source = session.resolve_audio(asset) if asset else None
            if source and source.exists():
                name = f"audio-{audio_id:03d}-{_safe_name(source.name, 'audio')}"
                destination = media_dir / name
                shutil.copy2(source, destination)
                exported_assets[audio_id] = f"media/{name}"
                audio_info["export_path"] = exported_assets[audio_id]
        chapter["audio"] = chapter.get("audio_links", [None])[0] if chapter.get("audio_links") else None

    document_roots: dict[int, Path] = {}

    def prepare_document(document) -> Path:
        document_id = int(document.id or 0)
        if document_id in document_roots:
            return document_roots[document_id]
        destination = book_dir / f"source-{document_id}"
        stored = session.root / document.stored_path
        if document.kind == SourceDocumentKind.EPUB and stored.is_file():
            _extract_epub_assets(stored, destination)
        else:
            resource_root = session.root / document.resource_root if document.resource_root else stored.parent
            if resource_root.is_dir():
                shutil.copytree(resource_root, destination, dirs_exist_ok=True)
            elif stored.is_file():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stored, destination / stored.name)
        document_roots[document_id] = destination
        return destination

    source_chapters = session.repository.chapters()
    for chapter, chapter_data in zip(source_chapters, data["chapters"]):
        chapter_id = chapter.id or 0
        segments = session.repository.segments(chapter_id)
        source_html = chapter.source_html
        source_parts = session.repository.chapter_source_parts(chapter_id)
        source_info = source_parts[0] if source_parts else session.repository.chapter_source_document(chapter_id)
        base_href = ""
        if source_info:
            document, entry_path, _selector = source_info
            document_root = prepare_document(document)
            entry = document_root / entry_path
            if len(source_parts) <= 1 and entry.is_file():
                source_html = entry.read_text("utf-8", errors="replace")
            relative_base = Path("..") / "book" / document_root.name / Path(entry_path).parent
            base_href = relative_base.as_posix().rstrip("/") + "/"
        if not source_html:
            body = "".join(
                f"<p>{html.escape(segment.text).replace(chr(10), '<br>')}</p>"
                for segment in segments
            )
            source_html = (
                "<!doctype html><html><head><meta charset='utf-8'><style>"
                "body{max-width:52rem;margin:2rem auto;padding:0 1.5rem;font:18px/1.9 serif}"
                "</style></head><body>" + body + "</body></html>"
            )
        page_name = f"chapter-{chapter.position + 1:03d}.html"
        page = _annotate_html_export_page(source_html, segments, base_href)
        (pages_dir / page_name).write_text(page, encoding="utf-8")
        chapter_data["page_path"] = f"pages/{page_name}"

    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(session.manifest.title)
    document = _HTML.replace("__TITLE__", title).replace("__DATA__", encoded)
    index = target / "index.html"
    index.write_text(document, encoding="utf-8")
    return index

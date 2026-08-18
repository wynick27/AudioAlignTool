from __future__ import annotations

import html
import json
import posixpath
import re
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

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


def _infer_html_resource_directory(document_root: Path, source_html: str) -> Path:
    """Infer the source page directory when an imported cover/TOC lacks a locator."""
    soup = BeautifulSoup(source_html, "html.parser")
    references: list[str] = []
    for tag, attribute in (
        ("link", "href"), ("a", "href"), ("img", "src"), ("image", "href"),
        ("source", "src"), ("audio", "src"), ("video", "src"),
    ):
        for element in soup.find_all(tag):
            value = str(element.get(attribute, "")).strip()
            parsed = urlsplit(value)
            if value and not parsed.scheme and not value.startswith(("#", "//")):
                references.append(unquote(parsed.path))
    if not references:
        return document_root
    directories = [document_root]
    directories.extend(path for path in document_root.rglob("*") if path.is_dir())
    best = document_root
    best_score = -1
    asset_directories = {
        "image", "images", "img", "style", "styles", "css", "font", "fonts",
        "media", "audio", "video",
    }
    for directory in directories:
        matched = sum(1 for value in references if (directory / value).resolve().is_file())
        has_markup = any(
            child.is_file() and child.suffix.casefold() in {".html", ".htm", ".xhtml"}
            for child in directory.iterdir()
        )
        score = matched * 10 + int(has_markup) * 2
        if directory.name.casefold() in asset_directories:
            score -= 2
        if score > best_score:
            best, best_score = directory, score
    return best


def _annotate_html_export_page(
    source_html: str,
    segments: list[TextSegment],
    base_href: str,
    *,
    current_entry_path: str = "",
    chapter_targets: dict[str, tuple[int, str]] | None = None,
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
    targets = chapter_targets or {}
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", "")).strip()
        parsed = urlsplit(href)
        if not href or parsed.scheme or href.startswith("//"):
            continue
        target = targets.get(f"#{parsed.fragment}") if parsed.fragment else None
        if target is None and parsed.path:
            resolved = posixpath.normpath(posixpath.join(
                posixpath.dirname(current_entry_path), unquote(parsed.path),
            ))
            target = targets.get(resolved) or targets.get("@" + posixpath.basename(resolved))
        if target is None:
            continue
        target_index, target_page = target
        link["data-aat-chapter-index"] = str(target_index)
        link["data-aat-chapter-page"] = target_page
        if parsed.fragment:
            link["data-aat-fragment"] = parsed.fragment
    style = soup.new_tag("style")
    style.string = (
        "[data-aat-index]{cursor:pointer;}"
        "[data-aat-index].aat-active{background:rgba(255,214,64,.52)!important;"
        "box-shadow:inset 0 -2px rgba(226,145,0,.7);border-radius:.15em;}"
    )
    soup.head.append(style)

    nodes = [
        node for node in soup.find_all(string=True)
        if node.parent and node.parent.name not in {"style", "script", "title", "svg", "rt", "rp"}
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
        const chapterLink=event.target.closest('[data-aat-chapter-index]');
        if(chapterLink){
          event.preventDefault();
          const detail={index:Number(chapterLink.dataset.aatChapterIndex),
            page:chapterLink.dataset.aatChapterPage||'',fragment:chapterLink.dataset.aatFragment||''};
          if(typeof window.aatOpenChapter==='function')window.aatOpenChapter(detail);
          else parent.postMessage({type:'aat-chapter',...detail},'*');
          return;
        }
        const target=event.target.closest('[data-aat-index]');
        if(!target)return;
        const index=Number(target.dataset.aatIndex);
        if(typeof window.aatActivateSegment==='function')window.aatActivateSegment(index);
        else parent.postMessage({type:'aat-segment',index},'*');
      });
      window.aatSetActive=function(index,follow){
        document.querySelectorAll('.aat-active').forEach(e=>e.classList.remove('aat-active'));
        const targets=document.querySelectorAll('[data-aat-index="'+index+'"]');
        targets.forEach(target=>target.classList.add('aat-active'));
        // Keep playback readable without continually pulling the active sentence
        // into the middle of the page.  `nearest` is a no-op while it is already
        // visible and otherwise performs only the smallest necessary scroll.
        if(targets.length&&follow)targets[0].scrollIntoView({block:'nearest',inline:'nearest'});
      };
      addEventListener('message',function(event){
        if(!event.data || event.data.type!=='aat-active')return;
        window.aatSetActive(event.data.index,event.data.follow);
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
</style></head><body><header><strong>__TITLE__</strong><select id="chapters"></select><button id="previous" title="上一句">⏮</button><button id="play" title="播放/暂停">▶</button><button id="next" title="下一句">⏭</button><audio id="player" controls></audio><label>倍速 <select id="speed"><option value="0.25">0.25</option><option value="0.5">0.50</option><option value="0.75">0.75</option><option value="1" selected>1.00</option><option value="1.25">1.25</option><option value="1.5">1.50</option><option value="2">2.00</option><option value="2.5">2.50</option><option value="3">3.00</option></select>×</label><label><input id="loop" type="checkbox"> 单句循环</label><label><input id="follow" type="checkbox" checked> 跟随当前句</label><span class="time" id="clock">00:00:00</span><span class="grow"></span></header><iframe id="book" title="原书正文"></iframe>
<script id="alignment" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('alignment').textContent),select=document.getElementById('chapters'),player=document.getElementById('player'),playButton=document.getElementById('play'),book=document.getElementById('book'),clock=document.getElementById('clock'),speed=document.getElementById('speed'),loop=document.getElementById('loop'),follow=document.getElementById('follow');let active=-1,segments=[],links=[],linkIndex=0,lastSaved=0;const storageKey='AudioAlignTool:'+data.project_id;
data.chapters.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.title;select.appendChild(o)});
function localStart(i){return links.slice(0,i).reduce((n,x)=>n+x.source_end_ms-x.source_start_ms,0)}
function currentMs(){const x=links[linkIndex];return x?Math.max(0,localStart(linkIndex)+player.currentTime*1000-x.source_start_ms):0}
function save(force=false){const now=Date.now();if(!force&&now-lastSaved<500)return;lastSaved=now;try{localStorage.setItem(storageKey,JSON.stringify({chapter:+select.value,position_ms:Math.round(currentMs()),rate:player.playbackRate,loop:loop.checked,follow:follow.checked}))}catch(_){}}
function syncPlayButton(){const playing=!player.paused&&!player.ended;playButton.textContent=playing?'⏸':'▶';playButton.title=playing?'暂停':'播放';playButton.setAttribute('aria-label',playButton.title)}
function postActive(ensureVisible=true){if(active>=0&&book.contentWindow)book.contentWindow.postMessage({type:'aat-active',index:active,follow:ensureVisible&&follow.checked},'*')}
function playAt(ms,autoplay=true){if(!links.length)return;let i=links.findIndex((x,n)=>ms<localStart(n)+x.source_end_ms-x.source_start_ms);if(i<0)i=links.length-1;const x=links[i],position=Math.max(x.source_start_ms,(x.source_start_ms+ms-localStart(i)))/1000,src=x.export_path||x.path;const apply=()=>{player.currentTime=position;player.playbackRate=Number(speed.value);if(autoplay)player.play().catch(()=>{});tick()};if(i!==linkIndex||player.getAttribute('src')!==src){linkIndex=i;player.src=src;player.onloadedmetadata=apply}else apply()}
function load(i,position=0,fragment=''){const c=data.chapters[i];select.value=i;segments=c.segments;links=c.audio_links||(c.audio?[c.audio]:[]);linkIndex=0;active=-1;book.src=c.page_path+(fragment?'#'+encodeURIComponent(fragment):'');player.src=links[0]?(links[0].export_path||links[0].path):'';player.onloadedmetadata=()=>playAt(position,false);if(!links.length)save(true)}
function activate(i,autoplay=true){if(!segments[i])return;active=i;postActive(true);playAt(segments[i].start_ms,autoplay)}
function tick(){const x=links[linkIndex],ms=currentMs();clock.textContent=new Date(Math.max(0,ms)).toISOString().slice(11,19);if(loop.checked&&active>=0&&segments[active]&&ms>=segments[active].end_ms-25){playAt(segments[active].start_ms,true);return}if(x&&player.currentTime*1000>=x.source_end_ms){if(linkIndex+1<links.length)playAt(localStart(linkIndex+1),!player.paused);else player.pause()}const i=segments.findIndex(s=>s.end_ms>s.start_ms&&ms>=s.start_ms&&ms<s.end_ms);if(i!==active){active=i;postActive(true)}save()}
addEventListener('message',e=>{if(!e.data)return;if(e.data.type==='aat-segment')activate(Number(e.data.index),true);else if(e.data.type==='aat-chapter')load(Number(e.data.index),0,String(e.data.fragment||''))});book.addEventListener('load',()=>postActive(false));select.onchange=()=>load(+select.value,0);speed.onchange=()=>{player.playbackRate=Number(speed.value);save(true)};loop.onchange=()=>save(true);follow.onchange=()=>save(true);player.ontimeupdate=tick;player.onplay=syncPlayButton;player.onpause=()=>{syncPlayButton();save(true)};player.onended=syncPlayButton;player.onemptied=syncPlayButton;playButton.onclick=()=>player.paused?player.play():player.pause();document.getElementById('previous').onclick=()=>activate(Math.max(0,(active<0?0:active)-1),true);document.getElementById('next').onclick=()=>activate(Math.min(segments.length-1,(active<0?-1:active)+1),true);addEventListener('beforeunload',()=>save(true));let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(_){};speed.value=String(saved.rate||1);if(!speed.value)speed.value='1';player.playbackRate=Number(speed.value);loop.checked=!!saved.loop;follow.checked=saved.follow!==false;syncPlayButton();load(Math.max(0,Math.min(data.chapters.length-1,Number(saved.chapter)||0)),Math.max(0,Number(saved.position_ms)||0));
</script></body></html>"""


_STANDALONE_READER_SCRIPT = r"""
const data=JSON.parse(document.getElementById('aat-chapter-data').textContent),
player=document.getElementById('aat-player'),playButton=document.getElementById('aat-play'),
clock=document.getElementById('aat-clock'),speed=document.getElementById('aat-speed'),
loop=document.getElementById('aat-loop'),follow=document.getElementById('aat-follow');
const segments=data.segments||[],links=data.audio_links||[];
let active=-1,linkIndex=0,lastSaved=0;
const storageKey='AudioAlignTool:chapter:'+data.project_id+':'+data.chapter_id;
function localStart(i){return links.slice(0,i).reduce((n,x)=>n+x.source_end_ms-x.source_start_ms,0)}
function currentMs(){const x=links[linkIndex];return x?Math.max(0,localStart(linkIndex)+player.currentTime*1000-x.source_start_ms):0}
function save(force=false){const now=Date.now();if(!force&&now-lastSaved<500)return;lastSaved=now;try{localStorage.setItem(storageKey,JSON.stringify({position_ms:Math.round(currentMs()),rate:player.playbackRate,loop:loop.checked,follow:follow.checked}))}catch(_){}}
function syncPlayButton(){const playing=!player.paused&&!player.ended;playButton.textContent=playing?'⏸':'▶';playButton.title=playing?'暂停':'播放';playButton.setAttribute('aria-label',playButton.title)}
function playAt(ms,autoplay=true){
  if(!links.length)return;
  let i=links.findIndex((x,n)=>ms<localStart(n)+x.source_end_ms-x.source_start_ms);
  if(i<0)i=links.length-1;
  const x=links[i],position=Math.max(x.source_start_ms,x.source_start_ms+ms-localStart(i))/1000,src=x.export_path||x.path;
  const apply=()=>{player.currentTime=position;player.playbackRate=Number(speed.value);if(autoplay)player.play().catch(()=>{});tick()};
  if(i!==linkIndex||player.getAttribute('src')!==src){linkIndex=i;player.src=src;player.onloadedmetadata=apply}else apply();
}
function activate(i,autoplay=true){if(!segments[i])return;active=i;window.aatSetActive(i,follow.checked);playAt(segments[i].start_ms,autoplay)}
window.aatActivateSegment=i=>activate(Number(i),true);
window.aatOpenChapter=detail=>{
  const target=new URL(detail.page,window.location.href);
  if(detail.fragment)target.hash=detail.fragment;
  window.location.href=target.href;
};
function tick(){
  const x=links[linkIndex],ms=currentMs();clock.textContent=new Date(Math.max(0,ms)).toISOString().slice(11,19);
  if(loop.checked&&active>=0&&segments[active]&&ms>=segments[active].end_ms-25){playAt(segments[active].start_ms,true);return}
  if(x&&player.currentTime*1000>=x.source_end_ms){if(linkIndex+1<links.length)playAt(localStart(linkIndex+1),!player.paused);else player.pause()}
  const i=segments.findIndex(s=>s.end_ms>s.start_ms&&ms>=s.start_ms&&ms<s.end_ms);
  if(i!==active){active=i;window.aatSetActive(i,follow.checked)}
  save();
}
speed.onchange=()=>{player.playbackRate=Number(speed.value);save(true)};
loop.onchange=()=>save(true);follow.onchange=()=>save(true);player.ontimeupdate=tick;player.onplay=syncPlayButton;
player.onpause=()=>{syncPlayButton();save(true)};player.onended=syncPlayButton;player.onemptied=syncPlayButton;
playButton.onclick=()=>player.paused?player.play():player.pause();
document.getElementById('aat-previous').onclick=()=>activate(Math.max(0,(active<0?0:active)-1),true);
document.getElementById('aat-next').onclick=()=>activate(Math.min(segments.length-1,(active<0?-1:active)+1),true);
addEventListener('beforeunload',()=>save(true));
let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(_){}
speed.value=String(saved.rate||1);if(!speed.value)speed.value='1';player.playbackRate=Number(speed.value);loop.checked=!!saved.loop;follow.checked=saved.follow!==false;syncPlayButton();
if(links.length){player.src=links[0].export_path||links[0].path;player.onloadedmetadata=()=>playAt(Math.max(0,Number(saved.position_ms)||0),false)}
"""


def _standalone_chapter_page(
    source_html: str,
    segments: list[TextSegment],
    chapter_data: dict,
    project_id: str,
    base_href: str,
    current_entry_path: str = "",
    chapter_targets: dict[str, tuple[int, str]] | None = None,
) -> str:
    soup = BeautifulSoup(
        _annotate_html_export_page(
            source_html, segments, base_href,
            current_entry_path=current_entry_path,
            chapter_targets=chapter_targets,
        ),
        "html.parser",
    )
    if soup.body is None:
        body = soup.new_tag("body")
        soup.append(body)
    style = soup.new_tag("style")
    style.string = (
        "#aat-reader-toolbar{position:sticky;top:0;z-index:2147483647;display:flex;"
        "align-items:center;gap:.45rem;flex-wrap:wrap;padding:.5rem .7rem;background:#fff;"
        "color:#222;border-bottom:1px solid #bbb;box-shadow:0 1px 5px #0003;font:14px/1.4 "
        "system-ui,sans-serif}#aat-reader-toolbar button,#aat-reader-toolbar select{font:inherit;"
        "padding:.25rem .5rem}#aat-player{width:min(430px,55vw);height:36px}"
        "#aat-clock{min-width:5.2rem;font-variant-numeric:tabular-nums;color:#666}"
    )
    soup.head.append(style)
    toolbar = BeautifulSoup(
        '<div id="aat-reader-toolbar">'
        '<button id="aat-previous" title="上一句">⏮</button>'
        '<button id="aat-play" title="播放" aria-label="播放">▶</button>'
        '<button id="aat-next" title="下一句">⏭</button>'
        '<audio id="aat-player" controls></audio>'
        '<label>倍速 <select id="aat-speed">'
        '<option value="0.25">0.25</option><option value="0.5">0.50</option>'
        '<option value="0.75">0.75</option><option value="1" selected>1.00</option>'
        '<option value="1.25">1.25</option><option value="1.5">1.50</option>'
        '<option value="2">2.00</option><option value="2.5">2.50</option>'
        '<option value="3">3.00</option></select>×</label>'
        '<label><input id="aat-loop" type="checkbox"> 单句循环</label>'
        '<label><input id="aat-follow" type="checkbox" checked> 跟随当前句</label>'
        '<span id="aat-clock">00:00:00</span></div>',
        "html.parser",
    ).div
    soup.body.insert(0, toolbar)
    payload = {
        "project_id": project_id,
        "chapter_id": chapter_data.get("id"),
        "segments": chapter_data.get("segments", []),
        "audio_links": chapter_data.get("audio_links", []),
    }
    data_script = soup.new_tag("script", id="aat-chapter-data", type="application/json")
    data_script.string = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    soup.body.append(data_script)
    reader_script = soup.new_tag("script")
    reader_script.string = _STANDALONE_READER_SCRIPT
    soup.body.append(reader_script)
    return str(soup)


def export_html(
    session: ProjectSession,
    output: str | Path,
    *,
    standalone_chapters: bool = False,
) -> Path:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    media_dir = target / "media"
    media_dir.mkdir(exist_ok=True)
    pages_dir = target / "pages"
    if not standalone_chapters:
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
    page_names = [
        (
            f"{chapter.position + 1:03d}-{_safe_name(chapter.title, 'chapter')}.html"
            if standalone_chapters else f"chapter-{chapter.position + 1:03d}.html"
        )
        for chapter in source_chapters
    ]
    target_candidates: dict[str, set[tuple[int, str]]] = {}

    def add_target(key: str, target: tuple[int, str]) -> None:
        if key:
            target_candidates.setdefault(key, set()).add(target)

    fallback_document = None
    for index, chapter in enumerate(source_chapters):
        chapter_id = chapter.id or 0
        navigation_target = (index, page_names[index])
        parts = session.repository.chapter_source_parts(chapter_id)
        source_info = parts[0] if parts else session.repository.chapter_source_document(chapter_id)
        if source_info and fallback_document is None:
            fallback_document = source_info[0]
        for _document, entry_path, _selector in (
            parts or ([source_info] if source_info else [])
        ):
            normalized = posixpath.normpath(entry_path)
            add_target(normalized, navigation_target)
            add_target("@" + posixpath.basename(normalized), navigation_target)
        chapter_soup = BeautifulSoup(chapter.source_html or "", "html.parser")
        for element in chapter_soup.find_all(attrs={"id": True}):
            add_target("#" + str(element.get("id")), navigation_target)
        for element in chapter_soup.find_all("a", attrs={"name": True}):
            add_target("#" + str(element.get("name")), navigation_target)
    chapter_targets = {
        key: next(iter(candidates))
        for key, candidates in target_candidates.items() if len(candidates) == 1
    }

    for index, (chapter, chapter_data) in enumerate(zip(source_chapters, data["chapters"])):
        chapter_id = chapter.id or 0
        segments = session.repository.segments(chapter_id)
        source_html = chapter.source_html
        source_parts = session.repository.chapter_source_parts(chapter_id)
        source_info = source_parts[0] if source_parts else session.repository.chapter_source_document(chapter_id)
        base_href = ""
        current_entry_path = ""
        if source_info:
            document, entry_path, _selector = source_info
            current_entry_path = entry_path
            document_root = prepare_document(document)
            entry = document_root / entry_path
            if len(source_parts) <= 1 and entry.is_file():
                source_html = entry.read_text("utf-8", errors="replace")
            relative_base = (
                Path("book") if standalone_chapters else Path("..") / "book"
            ) / document_root.name / Path(entry_path).parent
            base_href = relative_base.as_posix().rstrip("/") + "/"
        elif fallback_document is not None:
            document_root = prepare_document(fallback_document)
            inferred_directory = _infer_html_resource_directory(document_root, source_html)
            inferred_relative = inferred_directory.relative_to(document_root)
            current_entry_path = (inferred_relative / "__inferred__.html").as_posix()
            relative_base = (
                Path("book") if standalone_chapters else Path("..") / "book"
            ) / document_root.name / inferred_relative
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
        if standalone_chapters:
            page_name = page_names[index]
            page = _standalone_chapter_page(
                source_html, segments, chapter_data,
                session.manifest.project_id, base_href,
                current_entry_path, chapter_targets,
            )
            (target / page_name).write_text(page, encoding="utf-8")
        else:
            page_name = page_names[index]
            page = _annotate_html_export_page(
                source_html, segments, base_href,
                current_entry_path=current_entry_path,
                chapter_targets=chapter_targets,
            )
            (pages_dir / page_name).write_text(page, encoding="utf-8")
            chapter_data["page_path"] = f"pages/{page_name}"

    if standalone_chapters:
        return target

    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(session.manifest.title)
    document = _HTML.replace("__TITLE__", title).replace("__DATA__", encoded)
    index = target / "index.html"
    index.write_text(document, encoding="utf-8")
    return index

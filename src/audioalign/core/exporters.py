from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from .alignment import enforce_monotonic
from .models import SegmentStatus, TextSegment
from .storage import ProjectSession


def _safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value).strip(" .")
    return name[:80] or fallback


def _srt_time(milliseconds: int) -> str:
    hours, remainder = divmod(max(0, milliseconds), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _vtt_time(milliseconds: int) -> str:
    return _srt_time(milliseconds).replace(",", ".")


def export_subtitles(session: ProjectSession, output: str | Path, kind: str) -> list[Path]:
    kind = kind.lower()
    if kind not in {"srt", "vtt"}:
        raise ValueError("Subtitle kind must be srt or vtt")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for chapter in session.repository.chapters():
        segments = [s for s in enforce_monotonic(session.repository.segments(chapter.id or 0)) if s.end_ms > s.start_ms]
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
        "chapters": chapters,
    }


def export_json(session: ProjectSession, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(project_alignment_dict(session), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif;--accent:#4f7cff}body{margin:0;background:#f4f3ef;color:#242424}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:.75rem 1rem;z-index:2;display:flex;gap:.75rem;align-items:center}
main{max-width:850px;margin:0 auto;padding:2rem 1.25rem 8rem;background:#fff;min-height:100vh}.segment{font-size:1.12rem;line-height:2;padding:.1rem .2rem;border-radius:.25rem;cursor:pointer}.segment.active{background:#dce6ff}.segment.unmatched{border-bottom:2px dotted #c04}
#player{width:min(520px,55vw)}select{padding:.35rem}.time{font-variant-numeric:tabular-nums;color:#666}@media(prefers-color-scheme:dark){body,main{background:#171717;color:#eee}header{background:#222;border-color:#444}.segment.active{background:#29406d}.time{color:#aaa}}
</style></head><body><header><strong>__TITLE__</strong><select id="chapters"></select><audio id="player" controls></audio><span class="time" id="clock">00:00</span></header><main><h1 id="chapter-title"></h1><article id="text"></article></main>
<script id="alignment" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('alignment').textContent),select=document.getElementById('chapters'),player=document.getElementById('player'),text=document.getElementById('text'),clock=document.getElementById('clock');let active=-1,segments=[],links=[],linkIndex=0;
data.chapters.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.title;select.appendChild(o)});
function localStart(i){return links.slice(0,i).reduce((n,x)=>n+x.source_end_ms-x.source_start_ms,0)}
function playAt(ms){let i=Math.max(0,links.findIndex((x,n)=>ms<localStart(n)+x.source_end_ms-x.source_start_ms));if(i<0)i=links.length-1;const x=links[i],position=(x.source_start_ms+ms-localStart(i))/1000,src=x.export_path||x.path;if(i!==linkIndex||player.getAttribute('src')!==src){linkIndex=i;player.src=src;player.onloadedmetadata=()=>{player.currentTime=position;player.play()}}else{player.currentTime=position;player.play()}}
function load(i){const c=data.chapters[i];document.getElementById('chapter-title').textContent=c.title;segments=c.segments;links=c.audio_links||(c.audio?[c.audio]:[]);linkIndex=0;text.replaceChildren();segments.forEach((s,j)=>{const e=document.createElement('span');e.className='segment '+s.status;e.textContent=s.text+' ';e.dataset.i=j;e.onclick=()=>playAt(s.start_ms);text.appendChild(e)});player.src=links[0]?(links[0].export_path||links[0].path):'';active=-1}
function tick(){const x=links[linkIndex],ms=x?localStart(linkIndex)+player.currentTime*1000-x.source_start_ms:0;clock.textContent=new Date(Math.max(0,ms)).toISOString().slice(14,19);if(x&&player.currentTime*1000>=x.source_end_ms){if(linkIndex+1<links.length)playAt(localStart(linkIndex+1));else player.pause()}const i=segments.findIndex(s=>ms>=s.start_ms&&ms<s.end_ms);if(i!==active){text.querySelector('.active')?.classList.remove('active');active=i;if(i>=0){const e=text.querySelector(`[data-i="${i}"]`);e.classList.add('active');e.scrollIntoView({block:'center',behavior:'smooth'})}}}
select.onchange=()=>load(+select.value);player.ontimeupdate=tick;load(0);
</script></body></html>"""


def export_html(session: ProjectSession, output: str | Path) -> Path:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    media_dir = target / "media"
    media_dir.mkdir(exist_ok=True)
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
    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = session.manifest.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    document = _HTML.replace("__TITLE__", title).replace("__DATA__", encoded)
    index = target / "index.html"
    index.write_text(document, encoding="utf-8")
    return index

from __future__ import annotations

import html
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from bisect import bisect_left
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
from threading import Lock
import unicodedata
from enum import StrEnum
from urllib.parse import unquote
from xml.etree import ElementTree

from bs4 import BeautifulSoup, NavigableString

from .audio import create_m4a_proxy, probe_audio
from .models import AudioConversionPolicy, Chapter, TextSegment
from .storage import ProjectSession, fingerprint_file
from .timecode import format_time_ms
from .text import html_to_text, normalize_for_match


Progress = Callable[[float, str], None]


class EpubTextPolicy(StrEnum):
    PRESERVE_SOURCE = "preserve_source"
    APPLY_EDITS = "apply_edits"


@dataclass(slots=True)
class EpubMediaOverlayOptions:
    audio_policy: AudioConversionPolicy = AudioConversionPolicy.AUTO_COMPATIBLE
    mono_bitrate: int = 64_000
    stereo_bitrate: int = 128_000
    extend_segment_ends: bool = False
    max_end_extension_ms: int = 2_000
    text_policy: EpubTextPolicy = EpubTextPolicy.PRESERVE_SOURCE
    warnings: list[str] = field(default_factory=list, repr=False)


@dataclass(slots=True)
class _OverlayUnit:
    text_id: str
    audio_id: int
    clip_start_ms: int
    clip_end_ms: int


def _source_epub(session: ProjectSession) -> Path | None:
    name = session.manifest.source_name
    candidates = [session.root / name, session.root / "source" / name]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".epub":
            return candidate
    basename = Path(name).name if name else ""
    if basename:
        candidate = next(session.root.glob(f"source/**/{basename}"), None)
        if candidate and candidate.suffix.lower() == ".epub":
            return candidate
    return next(session.root.glob("source/**/*.epub"), None)


def _copy_epub_tree(source: Path, target: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            destination = (target / item.filename).resolve()
            try:
                destination.relative_to(target.resolve())
            except ValueError as exc:
                raise ValueError(f"EPUB 包含不安全路径：{item.filename}") from exc
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def _write_template_epub(session: ProjectSession, root: Path) -> None:
    (root / "META-INF").mkdir(parents=True)
    content = root / "OEBPS"
    content.mkdir()
    (root / "mimetype").write_text("application/epub+zip", encoding="ascii")
    (root / "META-INF" / "container.xml").write_text(
        "<?xml version='1.0' encoding='utf-8'?>"
        "<container version='1.0' xmlns='urn:oasis:names:tc:opendocument:xmlns:container'>"
        "<rootfiles><rootfile full-path='OEBPS/content.opf' media-type='application/oebps-package+xml'/>"
        "</rootfiles></container>", encoding="utf-8",
    )
    manifest_items: list[str] = []
    spine_items: list[str] = []
    nav_links: list[str] = []
    for index, chapter in enumerate(session.repository.chapters()):
        filename = f"chapter-{index + 1:03d}.xhtml"
        body = "".join(
            f"<p>{html.escape(segment.text).replace(chr(10), '<br/>')}</p>"
            for segment in session.repository.segments(chapter.id or 0)
        )
        (content / filename).write_text(
            "<?xml version='1.0' encoding='utf-8'?>"
            "<html xmlns='http://www.w3.org/1999/xhtml'><head><meta charset='utf-8'/>"
            f"<title>{html.escape(chapter.title)}</title><link rel='stylesheet' href='book.css'/></head>"
            f"<body><h1>{html.escape(chapter.title)}</h1>{body}</body></html>", encoding="utf-8",
        )
        item_id = f"chapter-{index + 1}"
        manifest_items.append(
            f"<item id='{item_id}' href='{filename}' media-type='application/xhtml+xml'/>"
        )
        spine_items.append(f"<itemref idref='{item_id}'/>")
        nav_links.append(f"<li><a href='{filename}'>{html.escape(chapter.title)}</a></li>")
    (content / "book.css").write_text(
        "body{max-width:52rem;margin:2rem auto;padding:0 1.5rem;font:1em/1.8 serif}h1{line-height:1.3}",
        encoding="utf-8",
    )
    (content / "nav.xhtml").write_text(
        "<html xmlns='http://www.w3.org/1999/xhtml' xmlns:epub='http://www.idpf.org/2007/ops'>"
        "<head><title>目录</title></head><body><nav epub:type='toc'><ol>"
        + "".join(nav_links) + "</ol></nav></body></html>", encoding="utf-8",
    )
    manifest_items.extend([
        "<item id='style' href='book.css' media-type='text/css'/>",
        "<item id='nav' href='nav.xhtml' media-type='application/xhtml+xml' properties='nav'/>",
    ])
    (content / "content.opf").write_text(
        "<?xml version='1.0' encoding='utf-8'?>"
        "<package xmlns='http://www.idpf.org/2007/opf' version='3.0' unique-identifier='book-id'>"
        "<metadata xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        f"<dc:identifier id='book-id'>urn:aat:{html.escape(session.manifest.project_id)}</dc:identifier>"
        f"<dc:title>{html.escape(session.manifest.title)}</dc:title><dc:language>{session.manifest.language or 'und'}</dc:language>"
        "</metadata><manifest>" + "".join(manifest_items) + "</manifest><spine>"
        + "".join(spine_items) + "</spine></package>", encoding="utf-8",
    )


def _opf_location(root: Path) -> Path:
    container = ElementTree.parse(root / "META-INF" / "container.xml").getroot()
    for element in container.iter():
        if element.tag.rsplit("}", 1)[-1] == "rootfile":
            return root / element.attrib["full-path"]
    raise ValueError("EPUB 缺少 OPF rootfile")


def _namespace(element) -> str:
    return element.tag.split("}", 1)[0].lstrip("{") if "}" in element.tag else ""


def _q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}" if namespace else name


def _normalized_nodes(soup: BeautifulSoup):
    nodes = [
        node for node in soup.find_all(string=True)
        if node.parent and node.parent.name not in {"style", "script", "title", "svg", "rt", "rp"}
    ]
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
    return nodes, "".join(characters), mapping


def _match_projection(value: str) -> tuple[str, list[int]]:
    """Build punctuation-insensitive matching text and retain raw offsets."""
    characters: list[str] = []
    raw_offsets: list[int] = []
    for raw_offset, character in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if unicodedata.category(normalized).startswith(("P", "Z", "C")):
                continue
            characters.append(normalized)
            raw_offsets.append(raw_offset)
    return "".join(characters), raw_offsets


def _fuzzy_projection_match(
    document: str, target: str, cursor: int,
) -> tuple[int, int] | None:
    """Find a small, local textual variation without breaking source order.

    Media Overlay export must tolerate an edited spelling or an omitted short
    word, but a global fuzzy search can attach a repeated sentence to the wrong
    paragraph.  Candidate starts therefore come from matching blocks in a
    bounded forward window and must pass a deliberately high similarity bar.
    """
    if len(target) < 8 or cursor >= len(document):
        return None
    window_end = min(len(document), cursor + max(900, len(target) * 4))
    window = document[cursor:window_end]
    matcher = SequenceMatcher(None, target, window, autojunk=False)
    blocks = sorted(matcher.get_matching_blocks(), key=lambda item: item.size, reverse=True)
    minimum_block = max(3, min(12, len(target) // 5))
    starts = {cursor + block.b - block.a for block in blocks if block.size >= minimum_block}
    starts = {max(cursor, value) for value in starts if value < window_end}
    if not starts:
        return None

    variance = max(2, min(24, round(len(target) * 0.12)))
    candidate_lengths = {
        max(1, len(target) - variance), len(target), len(target) + variance,
    }
    best: tuple[float, int, int] | None = None
    for start in starts:
        for length in candidate_lengths:
            end = min(window_end, start + length)
            if end <= start:
                continue
            score = SequenceMatcher(
                None, target, document[start:end], autojunk=False,
            ).ratio()
            if best is None or score > best[0]:
                best = score, start, end
    threshold = 0.90 if len(target) < 20 else 0.88
    if best is None or best[0] < threshold:
        return None
    return best[1], best[2]


def _original_segment_targets(
    session: ProjectSession, chapter_id: int,
) -> dict[int, str]:
    """Recover immutable source text ranges for edited segment rows."""
    fragments = session.repository.source_fragments(chapter_id)
    targets: dict[int, str] = {}
    for segment in session.repository.segments(chapter_id):
        if segment.source_start_char is None or segment.source_end_char is None:
            continue
        parts: list[str] = []
        for fragment in fragments:
            overlap_start = max(segment.source_start_char, fragment.source_start_char)
            overlap_end = min(segment.source_end_char, fragment.source_end_char)
            if overlap_end <= overlap_start:
                continue
            local_start = overlap_start - fragment.source_start_char
            local_end = overlap_end - fragment.source_start_char
            parts.append(fragment.text[local_start:local_end])
        target = "".join(parts)
        if target:
            targets[segment.position] = target
    return targets


def _editable_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
    return " ".join(value.split())


def _locate_source_target(
    document: str,
    target: str,
    cursor: int,
) -> tuple[int, int] | None:
    target = " ".join(target.split())
    start = document.find(target, cursor) if target else -1
    if start >= 0:
        return start, start + len(target)
    projected_document, projected_offsets = _match_projection(document)
    projected_target, _unused = _match_projection(target)
    projected_cursor = bisect_left(projected_offsets, cursor)
    projected_start = (
        projected_document.find(projected_target, projected_cursor)
        if projected_target else -1
    )
    if projected_start < 0:
        fuzzy = _fuzzy_projection_match(
            projected_document, projected_target, projected_cursor,
        ) if projected_target else None
        if fuzzy is None:
            return None
        projected_start, projected_end = fuzzy
    else:
        projected_end = projected_start + len(projected_target)
    if projected_start >= len(projected_offsets) or projected_end <= projected_start:
        return None
    return projected_offsets[projected_start], projected_offsets[projected_end - 1] + 1


def _apply_text_edits_to_xhtml(
    source_html: str,
    segments: list[TextSegment],
    original_targets: dict[int, str],
    applied_positions: set[int],
) -> tuple[str, set[int], set[int]]:
    """Apply only edits confined to one ordinary text node.

    Cross-node replacements could destroy emphasis, drop caps, ruby or links;
    those edits deliberately remain in the project but not in the EPUB copy.
    """
    soup = BeautifulSoup(source_html, "html.parser")
    nodes, document, mapping = _normalized_nodes(soup)
    cursor = 0
    edits: dict[int, list[tuple[int, int, str, int]]] = {}
    modified: set[int] = set()
    unsafe: set[int] = set()
    for segment in segments:
        if segment.position in applied_positions:
            continue
        original = original_targets.get(segment.position, "")
        if not original or _editable_text(original) == _editable_text(segment.text):
            continue
        modified.add(segment.position)
        located = _locate_source_target(document, original, cursor)
        if located is None:
            continue
        start, end = located
        cursor = end
        per_node: dict[int, list[int]] = {}
        for item in mapping[start:end]:
            if item is not None:
                per_node.setdefault(item[0], []).append(item[1])
        if len(per_node) != 1 or not segment.text:
            unsafe.add(segment.position)
            continue
        node_index, offsets = next(iter(per_node.items()))
        local_start, local_end = min(offsets), max(offsets) + 1
        node_value = str(nodes[node_index])
        covered = set(offsets)
        if any(
            offset not in covered and not node_value[offset].isspace()
            for offset in range(local_start, local_end)
        ):
            unsafe.add(segment.position)
            continue
        node = nodes[node_index]
        if node.parent and node.parent.name in {"ruby", "rt", "rp", "a"}:
            unsafe.add(segment.position)
            continue
        edits.setdefault(node_index, []).append(
            (local_start, local_end, segment.text, segment.position)
        )
    for node_index, node_edits in edits.items():
        value = str(nodes[node_index])
        for start, end, replacement, position in sorted(node_edits, reverse=True):
            value = value[:start] + replacement + value[end:]
            applied_positions.add(position)
        nodes[node_index].replace_with(NavigableString(value))
    return str(soup), modified, unsafe


def _audio_chunks(session: ProjectSession, chapter_id: int, start_ms: int, end_ms: int):
    cursor = 0
    result = []
    for link in session.repository.chapter_links(chapter_id):
        duration = max(0, link.source_end_ms - link.source_start_ms)
        local_end = cursor + duration
        overlap_start, overlap_end = max(start_ms, cursor), min(end_ms, local_end)
        if overlap_end > overlap_start:
            result.append((
                link.audio_id,
                link.source_start_ms + overlap_start - cursor,
                link.source_start_ms + overlap_end - cursor,
            ))
        cursor = local_end
    return result


def _export_segment_ranges(
    segments: list[TextSegment], options: EpubMediaOverlayOptions,
) -> dict[int, tuple[int, int]]:
    """Return export-only cue ranges, optionally retaining short inter-cue pauses.

    Only directly adjacent, timed text segments participate.  This deliberately
    avoids stretching a cue across unmatched source text, long pauses, or an
    existing overlap.  Project timing is never changed.
    """
    ranges = {
        segment.position: (int(segment.start_ms), int(segment.end_ms))
        for segment in segments
        if segment.end_ms > segment.start_ms
    }
    if not options.extend_segment_ends:
        return ranges

    maximum = max(0, int(options.max_end_extension_ms))
    if maximum <= 0:
        return ranges
    for current, following in zip(segments, segments[1:]):
        if current.end_ms <= current.start_ms or following.end_ms <= following.start_ms:
            continue
        gap = int(following.start_ms) - int(current.end_ms)
        if 0 < gap <= maximum:
            ranges[current.position] = (int(current.start_ms), int(following.start_ms))
    return ranges


def _annotate_for_overlay(
    source_html: str, session: ProjectSession, chapter: Chapter, *,
    missing_is_error: bool = True,
    matched_positions: set[int] | None = None,
    segment_ranges: dict[int, tuple[int, int]] | None = None,
    target_texts: dict[int, str] | None = None,
) -> tuple[str, list[_OverlayUnit], list[str]]:
    soup = BeautifulSoup(source_html, "html.parser")
    for tag in soup.find_all(["script", "noscript"]):
        tag.decompose()
    nodes, document, mapping = _normalized_nodes(soup)
    projected_document, projected_offsets = _match_projection(document)
    metadata_title = " ".join(soup.title.get_text().split()) if soup.title else ""
    cursor = 0
    intervals: dict[int, list[tuple[int, int, str]]] = {}
    units: list[_OverlayUnit] = []
    errors: list[str] = []
    unit_index = 0
    for segment in session.repository.segments(chapter.id or 0):
        if matched_positions is not None and segment.position in matched_positions:
            continue
        if segment.end_ms <= segment.start_ms:
            continue
        export_start_ms, export_end_ms = (
            segment_ranges.get(segment.position, (segment.start_ms, segment.end_ms))
            if segment_ranges is not None
            else (segment.start_ms, segment.end_ms)
        )
        target = " ".join(
            (target_texts.get(segment.position, segment.text) if target_texts else segment.text).split()
        )
        start = document.find(target, cursor) if target else -1
        if start < 0 and target:
            projected_target, _unused = _match_projection(target)
            projected_cursor = bisect_left(projected_offsets, cursor)
            projected_start = (
                projected_document.find(projected_target, projected_cursor)
                if projected_target else -1
            )
            if projected_start >= 0:
                start = projected_offsets[projected_start]
                projected_end = projected_start + len(projected_target) - 1
                end = projected_offsets[projected_end] + 1
            else:
                fuzzy = _fuzzy_projection_match(
                    projected_document, projected_target, projected_cursor,
                ) if projected_target else None
                if fuzzy:
                    projected_start, projected_end = fuzzy
                    start = projected_offsets[projected_start]
                    end = projected_offsets[projected_end - 1] + 1
                else:
                    end = -1
        else:
            end = start + len(target) if start >= 0 else -1
        if start < 0:
            # Old imports could expose XHTML head/title as a text cue.  It has
            # no visible body node and therefore cannot own a Media Overlay.
            if target and target == metadata_title:
                continue
            if missing_is_error:
                errors.append(f"{chapter.title} · 句段 {segment.position + 1} 无法映射回原 XHTML")
            continue
        cursor = end
        chunks = _audio_chunks(session, chapter.id or 0, export_start_ms, export_end_ms)
        covered = sum(chunk_end - chunk_start for _asset, chunk_start, chunk_end in chunks)
        if not chunks or covered < export_end_ms - export_start_ms - 2:
            errors.append(f"{chapter.title} · 句段 {segment.position + 1} 超出已配对媒体范围")
            continue
        chunk_total = sum(chunk_end - chunk_start for _asset, chunk_start, chunk_end in chunks)
        global_cursor = start
        for chunk_position, (audio_id, clip_start, clip_end) in enumerate(chunks):
            if chunk_position + 1 == len(chunks):
                global_end = end
            else:
                ratio = (clip_end - clip_start) / max(1, chunk_total)
                global_end = min(end, global_cursor + max(1, round((end - start) * ratio)))
            per_node: dict[int, list[int]] = {}
            for item in mapping[global_cursor:global_end]:
                if item is not None:
                    per_node.setdefault(item[0], []).append(item[1])
            parts = [(node_index, min(offsets), max(offsets) + 1) for node_index, offsets in per_node.items()]
            part_characters = sum(part_end - part_start for _node, part_start, part_end in parts)
            clip_cursor = clip_start
            for part_position, (node_index, part_start, part_end) in enumerate(parts):
                unit_index += 1
                text_id = f"aat-{chapter.id}-{segment.position}-{unit_index}"
                if part_position + 1 == len(parts):
                    part_clip_end = clip_end
                else:
                    part_clip_end = clip_cursor + round(
                        (clip_end - clip_start) * (part_end - part_start) / max(1, part_characters)
                    )
                intervals.setdefault(node_index, []).append((part_start, part_end, text_id))
                units.append(_OverlayUnit(text_id, audio_id, clip_cursor, max(clip_cursor + 1, part_clip_end)))
                clip_cursor = part_clip_end
            global_cursor = global_end
        if matched_positions is not None:
            matched_positions.add(segment.position)

    for node_index in sorted(intervals, reverse=True):
        node = nodes[node_index]
        value = str(node)
        replacements = []
        offset = 0
        for start, end, text_id in sorted(intervals[node_index], key=lambda item: item[0]):
            if start > offset:
                replacements.append(NavigableString(value[offset:start]))
            span = soup.new_tag("span", id=text_id)
            span.string = value[start:end]
            replacements.append(span)
            offset = end
        if offset < len(value):
            replacements.append(NavigableString(value[offset:]))
        node.replace_with(*replacements)
    return str(soup), units, errors


def _transcode(
    source: Path, destination: Path, codec: str, bitrate: int,
    progress: Callable[[float], None] | None = None,
) -> None:
    import av  # type: ignore

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    input_container = av.open(str(source))
    output_container = av.open(str(temporary), mode="w", format="ipod" if codec == "aac" else "mp3")
    input_stream = input_container.streams.audio[0]
    rate = int(input_stream.codec_context.sample_rate or 48_000)
    channels = int(input_stream.codec_context.channels or 2)
    output_stream = output_container.add_stream(codec if codec == "aac" else "libmp3lame", rate=rate)
    output_stream.bit_rate = bitrate
    layout = "mono" if channels == 1 else "stereo"
    sample_format = "fltp" if codec == "aac" else "s16p"
    resampler = av.audio.resampler.AudioResampler(format=sample_format, layout=layout, rate=rate)
    duration_seconds = max(
        0.001,
        float(input_stream.duration * input_stream.time_base)
        if input_stream.duration is not None and input_stream.time_base is not None
        else float(input_container.duration or 0) / 1_000_000,
    )
    decoded_samples = 0
    last_report = 0.0
    try:
        for frame in input_container.decode(input_stream):
            decoded_samples += int(frame.samples or 0)
            converted = resampler.resample(frame)
            for output_frame in converted if isinstance(converted, list) else [converted]:
                if output_frame is None:
                    continue
                output_frame.pts = None
                for packet in output_stream.encode(output_frame):
                    output_container.mux(packet)
            now = time.monotonic()
            if progress and now - last_report >= 0.25:
                progress(min(0.99, decoded_samples / max(1, rate) / duration_seconds))
                last_report = now
        for packet in output_stream.encode(None):
            output_container.mux(packet)
    finally:
        output_container.close()
        input_container.close()
    os.replace(temporary, destination)
    if progress:
        progress(1.0)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _conversion_cache_path(
    source: Path, cache_root: Path, policy: AudioConversionPolicy,
    bitrate: int, extension: str,
) -> Path:
    signature = json.dumps(
        {
            "version": 1,
            "source": fingerprint_file(source),
            "policy": policy.value,
            "bitrate": bitrate,
            "extension": extension,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return cache_root / f"{hashlib.sha256(signature).hexdigest()}{extension}"


def _prepare_audio(
    source: Path, destination_root: Path, audio_id: int,
    options: EpubMediaOverlayOptions, cache_root: Path,
    progress: Callable[[float], None] | None = None,
) -> tuple[Path, str]:
    probe = probe_audio(source)
    codec = probe.codec.casefold()
    destination_root.mkdir(parents=True, exist_ok=True)
    if options.audio_policy == AudioConversionPolicy.AUTO_COMPATIBLE and codec in {"mp3", "mp3float"}:
        destination = destination_root / f"audio-{audio_id}.mp3"
        _link_or_copy(source, destination)
        if progress:
            progress(1.0)
        return destination, "audio/mpeg"
    if (
        options.audio_policy == AudioConversionPolicy.AUTO_COMPATIBLE
        and codec == "aac" and source.suffix.lower() == ".m4a"
    ):
        destination = destination_root / f"audio-{audio_id}.m4a"
        _link_or_copy(source, destination)
        if progress:
            progress(1.0)
        return destination, "audio/mp4"
    if options.audio_policy == AudioConversionPolicy.AUTO_COMPATIBLE and codec == "aac":
        destination = destination_root / f"audio-{audio_id}.m4a"
        cache = _conversion_cache_path(
            source, cache_root, options.audio_policy, 0, destination.suffix,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.is_file() or cache.stat().st_size == 0:
            create_m4a_proxy(source, cache)
        _link_or_copy(cache, destination)
        if progress:
            progress(1.0)
        return destination, "audio/mp4"
    target_codec = "libmp3lame" if options.audio_policy == AudioConversionPolicy.FORCE_MP3 else "aac"
    extension, media_type = (".mp3", "audio/mpeg") if target_codec == "libmp3lame" else (".m4a", "audio/mp4")
    destination = destination_root / f"audio-{audio_id}{extension}"
    bitrate = options.mono_bitrate if probe.channels == 1 else options.stereo_bitrate
    cache = _conversion_cache_path(
        source, cache_root, options.audio_policy, bitrate, extension,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file() or cache.stat().st_size == 0:
        _transcode(
            source, cache, "mp3" if target_codec == "libmp3lame" else "aac", bitrate,
            progress,
        )
    elif progress:
        progress(1.0)
    _link_or_copy(cache, destination)
    return destination, media_type


_ZIP_STORED_SUFFIXES = {
    ".aac", ".avif", ".flac", ".gif", ".jpeg", ".jpg", ".m4a", ".mp3",
    ".mp4", ".ogg", ".opus", ".png", ".webm", ".webp", ".woff", ".woff2",
}


def _pack_epub(root: Path, destination: Path, progress: Callable[[float], None] | None = None) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    files = [item for item in root.rglob("*") if item.is_file()]
    total_size = max(1, sum(item.stat().st_size for item in files))
    completed = 0
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        mimetype = root / "mimetype"
        if not mimetype.exists():
            mimetype.write_text("application/epub+zip", encoding="ascii")
        archive.write(mimetype, "mimetype", compress_type=zipfile.ZIP_STORED)
        completed += mimetype.stat().st_size
        for item in files:
            if item == mimetype:
                continue
            relative = item.relative_to(root).as_posix()
            if item.suffix.casefold() in _ZIP_STORED_SUFFIXES:
                info = zipfile.ZipInfo.from_file(item, relative)
                info.compress_type = zipfile.ZIP_STORED
                with item.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    while block := source.read(4 * 1024 * 1024):
                        target.write(block)
                        completed += len(block)
                        if progress:
                            progress(min(1.0, completed / total_size))
            else:
                archive.write(item, relative, compress_type=zipfile.ZIP_DEFLATED)
                completed += item.stat().st_size
                if progress:
                    progress(min(1.0, completed / total_size))
    os.replace(temporary, destination)


def export_epub_media_overlay(
    session: ProjectSession,
    output: str | Path,
    options: EpubMediaOverlayOptions | None = None,
    progress: Progress | None = None,
) -> Path:
    options = options or EpubMediaOverlayOptions()
    destination = Path(output)
    if destination.suffix.lower() != ".epub":
        destination = destination.with_suffix(".epub")
    progress = progress or (lambda _fraction, _message: None)
    work_root = session.root / "cache" / "epub-export-work"
    work_root.mkdir(parents=True, exist_ok=True)
    conversion_cache = session.root / "cache" / "epub-media"
    with tempfile.TemporaryDirectory(prefix="aat-epub-", dir=work_root) as temporary:
        root = Path(temporary) / "book"
        root.mkdir()
        source = _source_epub(session)
        if source:
            _copy_epub_tree(source, root)
        else:
            _write_template_epub(session, root)
        opf_path = _opf_location(root)
        opf_tree = ElementTree.parse(opf_path)
        package = opf_tree.getroot()
        namespace = _namespace(package)
        manifest = package.find(_q(namespace, "manifest"))
        spine = package.find(_q(namespace, "spine"))
        if manifest is None or spine is None:
            raise ValueError("EPUB OPF 缺少 manifest 或 spine")
        items = {item.attrib.get("id", ""): item for item in manifest.findall(_q(namespace, "item"))}
        spine_items = [items.get(itemref.attrib.get("idref", "")) for itemref in spine.findall(_q(namespace, "itemref"))]
        xhtml_items = [item for item in spine_items if item is not None and item.attrib.get("media-type") == "application/xhtml+xml"]
        chapters = session.repository.chapters()
        chapter_items: list[list] = []
        unused = list(xhtml_items)
        for chapter in chapters:
            exact_parts = []
            for document, entry_path, _selector in session.repository.chapter_source_parts(chapter.id or 0):
                if document.kind.value != "epub" or not entry_path:
                    continue
                normalized_entry = Path(entry_path).as_posix()
                candidate = next((
                    item for item in unused
                    if (opf_path.parent.relative_to(root) / unquote(item.attrib["href"])).as_posix()
                    == normalized_entry
                ), None)
                if candidate is not None:
                    exact_parts.append(candidate)
                    unused.remove(candidate)
            if exact_parts:
                chapter_items.append(exact_parts)
                continue
            selected = None
            source_info = session.repository.chapter_source_document(chapter.id or 0)
            if source_info and source_info[0].kind.value == "epub":
                entry_path = Path(source_info[1]).as_posix()
                for candidate in unused:
                    archive_path = (
                        opf_path.parent.relative_to(root) / unquote(candidate.attrib["href"])
                    ).as_posix()
                    if archive_path == entry_path:
                        selected = candidate
                        break
            if selected is None and chapter.source_html:
                target = normalize_for_match(html_to_text(chapter.source_html))[:1200]
                scored = []
                for candidate in unused:
                    candidate_path = opf_path.parent / unquote(candidate.attrib["href"])
                    candidate_text = normalize_for_match(
                        html_to_text(candidate_path.read_text("utf-8", errors="replace"))
                    )[:1200]
                    scored.append((SequenceMatcher(None, target, candidate_text).ratio(), candidate))
                if scored:
                    score, candidate = max(scored, key=lambda item: item[0])
                    if score >= 0.45:
                        selected = candidate
            if selected is None and unused:
                selected = unused[0]
            if selected is None:
                raise ValueError(f"找不到章节“{chapter.title}”对应的原 EPUB 页面")
            unused.remove(selected)
            chapter_items.append([selected])

        overlay_root = opf_path.parent / "audioalign"
        media_root = overlay_root / "media"
        asset_outputs: dict[int, tuple[Path, str]] = {}
        used_audio_ids = sorted({link.audio_id for chapter in chapters for link in session.repository.chapter_links(chapter.id or 0)})
        media_jobs = []
        for audio_id in used_audio_ids:
            asset = session.repository.audio(audio_id)
            source_audio = session.resolve_audio(asset) if asset else None
            if not asset or not source_audio:
                raise ValueError(f"找不到媒体资源 #{audio_id}")
            media_jobs.append((audio_id, asset, source_audio, max(1, asset.duration_ms)))
        media_durations = {audio_id: duration for audio_id, _asset, _path, duration in media_jobs}
        total_media_duration = max(1, sum(media_durations.values()))
        worker_count = min(len(media_jobs), 4, max(1, (os.cpu_count() or 2) // 2))
        media_progress = {audio_id: 0.0 for audio_id in used_audio_ids}
        progress_lock = Lock()
        media_started = time.monotonic()

        def report_media(audio_id: int, value: float, name: str) -> None:
            with progress_lock:
                media_progress[audio_id] = max(media_progress[audio_id], min(1.0, value))
                completed = sum(
                    media_durations[item_id] * fraction
                    for item_id, fraction in media_progress.items()
                )
                elapsed = max(0.001, time.monotonic() - media_started)
                speed = completed / 1000 / elapsed
            progress(
                0.05 + 0.4 * completed / total_media_duration,
                f"准备 EPUB 音频（{worker_count} 路并行 · {speed:.1f}×）：{name}",
            )

        def prepare(job):
            audio_id, _asset, source_audio, _duration = job
            return audio_id, _prepare_audio(
                source_audio, media_root, audio_id, options, conversion_cache,
                lambda value: report_media(audio_id, value, source_audio.name),
            )

        if not media_jobs:
            prepared = []
        elif worker_count == 1:
            prepared = [prepare(job) for job in media_jobs]
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="epub-audio") as pool:
                futures = [pool.submit(prepare, job) for job in media_jobs]
                prepared = [future.result() for future in as_completed(futures)]
        asset_outputs.update(prepared)
        for audio_id in used_audio_ids:
            output_path, media_type = asset_outputs[audio_id]
            ElementTree.SubElement(manifest, _q(namespace, "item"), {
                "id": f"aat-audio-{audio_id}",
                "href": output_path.relative_to(opf_path.parent).as_posix(),
                "media-type": media_type,
            })

        all_errors: list[str] = []
        for index, chapter in enumerate(chapters):
            items_for_chapter = chapter_items[index]
            matched_positions: set[int] = set()
            chapter_segments = session.repository.segments(chapter.id or 0)
            segment_ranges = _export_segment_ranges(chapter_segments, options)
            original_targets = _original_segment_targets(session, chapter.id or 0)
            modified_positions = {
                segment.position for segment in chapter_segments
                if original_targets.get(segment.position)
                and _editable_text(original_targets[segment.position])
                != _editable_text(segment.text)
            }
            applied_edit_positions: set[int] = set()
            page_titles: set[str] = set()
            for part_index, item in enumerate(items_for_chapter):
                xhtml_path = opf_path.parent / unquote(item.attrib["href"])
                source_html = xhtml_path.read_text("utf-8", errors="replace")
                source_soup = BeautifulSoup(source_html, "html.parser")
                if source_soup.title:
                    page_titles.add(" ".join(source_soup.title.get_text().split()))
                if options.text_policy == EpubTextPolicy.APPLY_EDITS:
                    source_html, _modified_here, _unsafe_here = _apply_text_edits_to_xhtml(
                        source_html, chapter_segments, original_targets,
                        applied_edit_positions,
                    )
                target_texts = {
                    segment.position: (
                        segment.text
                        if segment.position in applied_edit_positions
                        else original_targets.get(segment.position, segment.text)
                    )
                    for segment in chapter_segments
                }
                rendered, units, errors = _annotate_for_overlay(
                    source_html, session, chapter,
                    missing_is_error=len(items_for_chapter) == 1,
                    matched_positions=matched_positions,
                    segment_ranges=segment_ranges,
                    target_texts=target_texts,
                )
                all_errors.extend(errors)
                if not units:
                    continue
                xhtml_path.write_text(rendered, encoding="utf-8")
                smil_id = f"aat-smil-{index + 1}-{part_index + 1}"
                smil_path = overlay_root / f"chapter-{index + 1:03d}-{part_index + 1:02d}.smil"
                smil_path.parent.mkdir(parents=True, exist_ok=True)
                sequence = []
                for unit_index, unit in enumerate(units):
                    audio_path, _media_type = asset_outputs[unit.audio_id]
                    text_src = os.path.relpath(xhtml_path, smil_path.parent).replace("\\", "/") + f"#{unit.text_id}"
                    audio_src = os.path.relpath(audio_path, smil_path.parent).replace("\\", "/")
                    sequence.append(
                        f"<par id='par-{unit_index + 1}'><text src='{html.escape(text_src)}'/>"
                        f"<audio src='{html.escape(audio_src)}' clipBegin='{format_time_ms(unit.clip_start_ms)}' "
                        f"clipEnd='{format_time_ms(unit.clip_end_ms)}'/></par>"
                    )
                smil_path.write_text(
                    "<?xml version='1.0' encoding='utf-8'?>"
                    "<smil xmlns='http://www.w3.org/ns/SMIL' version='3.0'><body><seq>"
                    + "".join(sequence) + "</seq></body></smil>", encoding="utf-8",
                )
                ElementTree.SubElement(manifest, _q(namespace, "item"), {
                    "id": smil_id,
                    "href": smil_path.relative_to(opf_path.parent).as_posix(),
                    "media-type": "application/smil+xml",
                })
                item.set("media-overlay", smil_id)
            if len(items_for_chapter) > 1:
                for segment in chapter_segments:
                    target = " ".join(
                        original_targets.get(segment.position, segment.text).split()
                    )
                    if (
                        segment.end_ms > segment.start_ms
                        and segment.position not in matched_positions
                        and target not in page_titles
                    ):
                        all_errors.append(
                            f"{chapter.title} · 句段 {segment.position + 1} 无法映射回任何原 XHTML 页面"
                        )
            if options.text_policy == EpubTextPolicy.APPLY_EDITS:
                for position in sorted(modified_positions - applied_edit_positions):
                    all_errors.append(
                        f"{chapter.title} · 句段 {position + 1} 的文字修改跨越样式、链接、ruby，"
                        "或无法可靠定位；导出副本已保持原文"
                    )
            progress(0.5 + 0.4 * (index + 1) / max(1, len(chapters)), f"生成 Media Overlay：{chapter.title}")
        options.warnings[:] = all_errors
        if all_errors:
            progress(
                0.92,
                f"已跳过 {len(all_errors)} 个无法映射或超出媒体范围的句段；其余内容继续导出",
            )
        ElementTree.register_namespace("", namespace)
        opf_tree.write(opf_path, encoding="utf-8", xml_declaration=True)
        progress(0.95, "打包 EPUB 3 Media Overlays")
        _pack_epub(
            root, destination,
            lambda value: progress(0.95 + 0.049 * value, "打包 EPUB 3 Media Overlays"),
        )
    progress(1.0, "EPUB 3 Media Overlays 导出完成")
    return destination

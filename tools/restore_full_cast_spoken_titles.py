from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from bs4 import BeautifulSoup


TEMP_POSITION = 1_000_000


def _flat(value: str) -> str:
    return " ".join((value or "").split())


def _key(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _visible_key(source_html: str) -> str:
    soup = BeautifulSoup(source_html or "", "html.parser")
    if soup.head:
        soup.head.decompose()
    return _key(soup.get_text(" ", strip=True))


def _display_label(value: str) -> str:
    cleaned = value.replace("\xad", " ")
    match = re.search(r"CHAPTER\s+[A-Z0-9-]+", cleaned, flags=re.IGNORECASE)
    return match.group(0).upper() if match else _flat(cleaned).strip("�◆◇—–- ")


def _combine_html_documents(left: str, right: str) -> str:
    left_soup = BeautifulSoup(left or "<html><body></body></html>", "html.parser")
    right_soup = BeautifulSoup(right or "<html><body></body></html>", "html.parser")
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


def _backup(connection: sqlite3.Connection, destination: Path) -> None:
    target = sqlite3.connect(destination)
    try:
        connection.backup(target)
    finally:
        target.close()


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _spine_pages(epub: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(epub) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(
            node.attrib["full-path"]
            for node in container.iter()
            if node.tag.rsplit("}", 1)[-1] == "rootfile"
        )
        opf = ElementTree.fromstring(archive.read(rootfile))
        base = Path(rootfile).parent
        manifest: dict[str, str] = {}
        spine: list[str] = []
        for node in opf.iter():
            name = node.tag.rsplit("}", 1)[-1]
            if name == "item" and node.attrib.get("id") and node.attrib.get("href"):
                manifest[node.attrib["id"]] = node.attrib["href"]
            elif name == "itemref" and node.attrib.get("idref"):
                spine.append(node.attrib["idref"])
        pages = []
        for item_id in spine:
            href = manifest.get(item_id)
            if not href:
                continue
            entry = (base / href).as_posix()
            try:
                source_html = archive.read(entry).decode("utf-8", errors="replace")
            except KeyError:
                continue
            pages.append((entry, _visible_key(source_html)))
        return pages


def _match_page(source_html: str, pages: list[tuple[str, str]]) -> str:
    target = _visible_key(source_html)
    exact = [entry for entry, value in pages if target and value == target]
    if len(exact) == 1:
        return exact[0]
    contained = [
        (min(len(target), len(value)), entry)
        for entry, value in pages
        if target and value and (target in value or value in target)
    ]
    return max(contained, default=(0, ""))[1]


def _ensure_source_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_documents (
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, original_path TEXT NOT NULL,
            stored_path TEXT NOT NULL, fingerprint TEXT NOT NULL DEFAULT '',
            entry_path TEXT NOT NULL DEFAULT '', resource_root TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS chapter_source_documents (
            chapter_id INTEGER PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
            document_id INTEGER NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
            entry_path TEXT NOT NULL DEFAULT '', selector TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS chapter_source_parts (
            chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            document_id INTEGER NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
            entry_path TEXT NOT NULL DEFAULT '', selector TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(chapter_id, position)
        );
        CREATE TABLE IF NOT EXISTS source_fragment_locators (
            fragment_id INTEGER PRIMARY KEY REFERENCES source_fragments(id) ON DELETE CASCADE,
            dom_locator TEXT NOT NULL DEFAULT ''
        );
        """
    )


def _source_document(connection: sqlite3.Connection, project: Path) -> tuple[int, list[tuple[str, str]]]:
    epubs = sorted((project / "source").glob("*.epub"))
    if not epubs:
        return 0, []
    epub = epubs[0]
    stored_path = epub.relative_to(project).as_posix()
    row = connection.execute(
        "SELECT id FROM source_documents WHERE stored_path=?", (stored_path,)
    ).fetchone()
    if row:
        document_id = int(row[0])
    else:
        digest = hashlib.sha256(epub.read_bytes()).hexdigest()
        cursor = connection.execute(
            """INSERT INTO source_documents
            (kind,original_path,stored_path,fingerprint,entry_path,resource_root)
            VALUES ('epub',?,?,?,?,?)""",
            (str(epub), stored_path, digest, epub.name, "source"),
        )
        document_id = int(cursor.lastrowid)
    return document_id, _spine_pages(epub)


def _copy_anchor(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    old_segment_id: int,
    new_chapter_id: int,
    new_segment_id: int,
    offset: int,
) -> None:
    if not _table_exists(source, "text_audio_anchors"):
        return
    for anchor in source.execute(
        "SELECT * FROM text_audio_anchors WHERE segment_id=? ORDER BY id", (old_segment_id,)
    ):
        destination.execute(
            """INSERT INTO text_audio_anchors
            (chapter_id,segment_id,source_start_char,source_end_char,start_ms,end_ms,confidence,method)
            VALUES (?,?,?,?,?,?,?,?)""",
            (
                new_chapter_id,
                new_segment_id,
                anchor["source_start_char"] + offset,
                anchor["source_end_char"] + offset,
                anchor["start_ms"],
                anchor["end_ms"],
                anchor["confidence"],
                anchor["method"],
            ),
        )


def restore(project: Path, *, apply: bool) -> dict[str, object]:
    database = project / "project.sqlite3"
    candidates = sorted((project / "backups").glob("project-before-title-cleanup-*.sqlite3"))
    if not database.is_file() or not candidates:
        raise FileNotFoundError(f"缺少项目数据库或修复前备份：{project}")
    source_path = candidates[-1]
    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    chapters = source.execute("SELECT * FROM chapters ORDER BY position,id").fetchall()
    invisible_ids: set[int] = set()
    pairs: list[tuple[sqlite3.Row, sqlite3.Row]] = []
    timed_heading_ids: set[int] = set()
    for index, chapter in enumerate(chapters):
        soup = BeautifulSoup(chapter["source_html"] or "", "html.parser")
        metadata_title = _flat(soup.title.get_text()) if soup.title else ""
        chapter_key = _key(chapter["title"])
        segments = source.execute(
            "SELECT * FROM text_segments WHERE chapter_id=? ORDER BY position,id",
            (chapter["id"],),
        ).fetchall()
        for segment in segments:
            if (
                metadata_title
                and _flat(segment["text"]) == metadata_title
                and segment["source_fragment_id"] is None
                and segment["source_start_char"] is None
                and segment["source_end_char"] is None
            ):
                invisible_ids.add(int(segment["id"]))
            if (
                segment["position"] <= 1
                and segment["end_ms"] > segment["start_ms"]
                and chapter_key
                and _key(segment["text"]) == chapter_key
            ):
                timed_heading_ids.add(int(segment["id"]))
        media_count = source.execute(
            "SELECT count(*) FROM chapter_audio_links WHERE chapter_id=?", (chapter["id"],)
        ).fetchone()[0]
        title_page = (
            chapter_key.startswith("chapter")
            and len(chapter_key) <= 36
            and len(segments) <= 5
            and not any(segment["end_ms"] > segment["start_ms"] for segment in segments)
            and int(media_count) == 0
            and index + 1 < len(chapters)
        )
        if title_page:
            pairs.append((chapter, chapters[index + 1]))

    safety_backup = ""
    restored_headings = 0
    inserted_title_segments = 0
    mapped_source_pairs = 0
    pair_offsets: dict[int, int] = {}
    if apply:
        destination = sqlite3.connect(database)
        destination.row_factory = sqlite3.Row
        destination.execute("PRAGMA foreign_keys=ON")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safety_path = project / "backups" / f"project-before-spoken-title-merge-{stamp}.sqlite3"
        _backup(destination, safety_path)
        safety_backup = str(safety_path)
        try:
            _ensure_source_tables(destination)
            destination.execute("BEGIN IMMEDIATE")
            document_id, pages = _source_document(destination, project)
            for title_chapter, body_chapter in pairs:
                target_id = int(body_chapter["id"])
                target = destination.execute(
                    "SELECT * FROM chapters WHERE id=?", (target_id,)
                ).fetchone()
                if target is None:
                    raise RuntimeError(f"找不到待合并正文：chapter_id={target_id}")
                if destination.execute(
                    "SELECT 1 FROM chapters WHERE id=?", (title_chapter["id"],)
                ).fetchone():
                    raise RuntimeError(f"标题页仍存在，拒绝重复合并：chapter_id={title_chapter['id']}")

                old_fragments = source.execute(
                    "SELECT * FROM source_fragments WHERE chapter_id=? ORDER BY position,id",
                    (title_chapter["id"],),
                ).fetchall()
                visible_segments = [
                    row
                    for row in source.execute(
                        "SELECT * FROM text_segments WHERE chapter_id=? ORDER BY position,id",
                        (title_chapter["id"],),
                    )
                    if int(row["id"]) not in invisible_ids
                ]
                title_length = max(
                    [int(row["source_end_char"]) for row in old_fragments]
                    + [
                        int(row["source_end_char"])
                        for row in visible_segments
                        if row["source_end_char"] is not None
                    ]
                    + [len("\n\n".join(row["text"] for row in visible_segments))]
                )
                offset = title_length + 2
                pair_offsets[target_id] = offset
                first_current = destination.execute(
                    "SELECT text FROM text_segments WHERE chapter_id=? ORDER BY position,id LIMIT 1",
                    (target_id,),
                ).fetchone()
                already_merged = bool(
                    visible_segments
                    and first_current
                    and _key(first_current[0]) == _key(visible_segments[0]["text"])
                )
                if already_merged:
                    continue
                destination.execute(
                    """UPDATE source_fragments SET position=position+?,
                    source_start_char=source_start_char+?,source_end_char=source_end_char+?
                    WHERE chapter_id=?""",
                    (TEMP_POSITION, offset, offset, target_id),
                )
                destination.execute(
                    "UPDATE source_fragments SET position=position-?+? WHERE chapter_id=?",
                    (TEMP_POSITION, len(old_fragments), target_id),
                )
                destination.execute(
                    """UPDATE text_segments SET position=position+?,
                    source_start_char=CASE WHEN source_start_char IS NULL THEN NULL ELSE source_start_char+? END,
                    source_end_char=CASE WHEN source_end_char IS NULL THEN NULL ELSE source_end_char+? END
                    WHERE chapter_id=?""",
                    (TEMP_POSITION, offset, offset, target_id),
                )
                destination.execute(
                    "UPDATE text_segments SET position=position-?+? WHERE chapter_id=?",
                    (TEMP_POSITION, len(visible_segments), target_id),
                )
                destination.execute(
                    """UPDATE text_audio_anchors SET source_start_char=source_start_char+?,
                    source_end_char=source_end_char+? WHERE chapter_id=?""",
                    (offset, offset, target_id),
                )

                fragment_ids: dict[int, int] = {}
                for position, fragment in enumerate(old_fragments):
                    cursor = destination.execute(
                        """INSERT INTO source_fragments
                        (chapter_id,position,kind,text,source_start_char,source_end_char)
                        VALUES (?,?,?,?,?,?)""",
                        (
                            target_id,
                            position,
                            fragment["kind"],
                            fragment["text"],
                            fragment["source_start_char"],
                            fragment["source_end_char"],
                        ),
                    )
                    fragment_ids[int(fragment["id"])] = int(cursor.lastrowid)
                    if _table_exists(source, "source_fragment_locators"):
                        locator = source.execute(
                            "SELECT dom_locator FROM source_fragment_locators WHERE fragment_id=?",
                            (fragment["id"],),
                        ).fetchone()
                        if locator:
                            destination.execute(
                                "INSERT OR REPLACE INTO source_fragment_locators(fragment_id,dom_locator) VALUES (?,?)",
                                (cursor.lastrowid, locator[0]),
                            )
                for position, segment in enumerate(visible_segments):
                    cursor = destination.execute(
                        """INSERT INTO text_segments
                        (chapter_id,position,text,start_ms,end_ms,confidence,status,locked,origin,
                         source_fragment_id,source_start_char,source_end_char)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            target_id,
                            position,
                            segment["text"],
                            segment["start_ms"],
                            segment["end_ms"],
                            segment["confidence"],
                            segment["status"],
                            segment["locked"],
                            segment["origin"],
                            fragment_ids.get(segment["source_fragment_id"]),
                            segment["source_start_char"],
                            segment["source_end_char"],
                        ),
                    )
                    _copy_anchor(
                        source, destination, int(segment["id"]), target_id,
                        int(cursor.lastrowid), 0,
                    )
                    inserted_title_segments += 1

                label = _display_label(title_chapter["title"])
                combined_title = f"{label} — {target['title']}" if label else target["title"]
                destination.execute(
                    "UPDATE chapters SET title=?,source_html=? WHERE id=?",
                    (
                        combined_title,
                        _combine_html_documents(title_chapter["source_html"], target["source_html"]),
                        target_id,
                    ),
                )
                destination.execute("DELETE FROM alignment_runs WHERE chapter_id=?", (target_id,))

                if document_id and pages:
                    title_entry = _match_page(title_chapter["source_html"], pages)
                    body_entry = _match_page(body_chapter["source_html"], pages)
                    entries = [entry for entry in (title_entry, body_entry) if entry]
                    if len(entries) == 2 and entries[0] != entries[1]:
                        destination.execute(
                            """INSERT INTO chapter_source_documents
                            (chapter_id,document_id,entry_path,selector) VALUES (?,?,?,'')
                            ON CONFLICT(chapter_id) DO UPDATE SET document_id=excluded.document_id,
                            entry_path=excluded.entry_path,selector=''""",
                            (target_id, document_id, entries[0]),
                        )
                        destination.execute(
                            "DELETE FROM chapter_source_parts WHERE chapter_id=?", (target_id,)
                        )
                        destination.executemany(
                            """INSERT INTO chapter_source_parts
                            (chapter_id,position,document_id,entry_path,selector)
                            VALUES (?,?,?,?, '')""",
                            [
                                (target_id, position, document_id, entry)
                                for position, entry in enumerate(entries)
                            ],
                        )
                        mapped_source_pairs += 1
            for heading_id in sorted(timed_heading_ids):
                heading = source.execute(
                    "SELECT * FROM text_segments WHERE id=?", (heading_id,)
                ).fetchone()
                current = destination.execute(
                    "SELECT 1 FROM text_segments WHERE id=?", (heading_id,)
                ).fetchone()
                if not heading or not current:
                    continue
                destination.execute(
                    """UPDATE text_segments SET start_ms=?,end_ms=?,confidence=?,status=?,locked=?
                    WHERE id=?""",
                    (
                        heading["start_ms"], heading["end_ms"], heading["confidence"],
                        heading["status"], heading["locked"], heading_id,
                    ),
                )
                destination.execute(
                    "DELETE FROM text_audio_anchors WHERE segment_id=?", (heading_id,)
                )
                _copy_anchor(
                    source, destination, heading_id, int(heading["chapter_id"]), heading_id,
                    pair_offsets.get(int(heading["chapter_id"]), 0),
                )
                restored_headings += 1
            destination.commit()
            integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_keys = len(destination.execute("PRAGMA foreign_key_check").fetchall())
            if integrity != "ok" or foreign_keys:
                raise RuntimeError(
                    f"合并后数据库校验失败：integrity={integrity}, foreign_keys={foreign_keys}"
                )
        except Exception:
            destination.rollback()
            raise
        finally:
            destination.close()
    source.close()
    return {
        "project": project.name,
        "merged_spoken_title_pages": len(pairs),
        "inserted_spoken_title_segments": inserted_title_segments,
        "restored_timed_visible_titles": restored_headings,
        "excluded_invisible_head_titles": len(invisible_ids),
        "mapped_two_source_pages": mapped_source_pairs,
        "source_backup": str(source_path),
        "safety_backup": safety_backup,
        "applied": apply,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge spoken CHAPTER pages without deleting their text or timings."
    )
    parser.add_argument("projects", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    for project in arguments.projects:
        print(json.dumps(restore(project.resolve(), apply=arguments.apply), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

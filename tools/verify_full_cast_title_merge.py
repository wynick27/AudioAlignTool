from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def key(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def flat(value: str) -> str:
    return " ".join((value or "").split())


def verify(project: Path) -> dict[str, object]:
    backup = sorted((project / "backups").glob("project-before-title-cleanup-*.sqlite3"))[-1]
    old = sqlite3.connect(f"file:{backup.as_posix()}?mode=ro", uri=True)
    old.row_factory = sqlite3.Row
    current = sqlite3.connect(f"file:{(project / 'project.sqlite3').as_posix()}?mode=ro", uri=True)
    current.row_factory = sqlite3.Row
    chapters = old.execute("SELECT * FROM chapters ORDER BY position,id").fetchall()
    pairs = []
    timed = []
    for index, chapter in enumerate(chapters):
        segments = old.execute(
            "SELECT * FROM text_segments WHERE chapter_id=? ORDER BY position,id",
            (chapter["id"],),
        ).fetchall()
        media = old.execute(
            "SELECT count(*) FROM chapter_audio_links WHERE chapter_id=?", (chapter["id"],)
        ).fetchone()[0]
        chapter_key = key(chapter["title"])
        if (
            chapter_key.startswith("chapter") and len(chapter_key) <= 36
            and len(segments) <= 5
            and not any(row["end_ms"] > row["start_ms"] for row in segments)
            and not media and index + 1 < len(chapters)
        ):
            visible = [row for row in segments if row["source_fragment_id"] is not None]
            pairs.append((chapter, chapters[index + 1], visible))
        soup = BeautifulSoup(chapter["source_html"] or "", "html.parser")
        metadata = flat(soup.title.get_text()) if soup.title else ""
        for row in segments:
            invisible = (
                metadata and flat(row["text"]) == metadata
                and row["source_fragment_id"] is None
                and row["source_start_char"] is None and row["source_end_char"] is None
            )
            if invisible:
                continue
            if (
                row["position"] <= 1 and row["end_ms"] > row["start_ms"]
                and chapter_key and key(row["text"]) == chapter_key
            ):
                timed.append((chapter, row))

    missing_titles = []
    wrong_first = []
    wrong_parts = []
    for title, body, visible in pairs:
        target = current.execute("SELECT * FROM chapters WHERE id=?", (body["id"],)).fetchone()
        first = current.execute(
            "SELECT text FROM text_segments WHERE chapter_id=? ORDER BY position,id LIMIT 1",
            (body["id"],),
        ).fetchone()
        if not target:
            missing_titles.append({"title": title["title"], "target": body["title"]})
            continue
        expected = visible[0]["text"] if visible else title["title"]
        if not first or key(first[0]) != key(expected):
            wrong_first.append({"chapter": target["title"], "expected": expected, "actual": first[0] if first else ""})
        parts = current.execute(
            "SELECT entry_path FROM chapter_source_parts WHERE chapter_id=? ORDER BY position",
            (body["id"],),
        ).fetchall()
        if len(parts) != 2:
            wrong_parts.append({"chapter": target["title"], "parts": [row[0] for row in parts]})

    missing_timed = []
    changed_timed = []
    remapped_timed = []
    for chapter, expected in timed:
        actual = current.execute("SELECT * FROM text_segments WHERE id=?", (expected["id"],)).fetchone()
        if not actual:
            replacement = current.execute(
                """SELECT * FROM text_segments WHERE chapter_id=? AND text=?
                AND end_ms>start_ms ORDER BY locked DESC,
                CASE status WHEN 'manual' THEN 0 ELSE 1 END,position,id LIMIT 1""",
                (chapter["id"], expected["text"]),
            ).fetchone()
            if replacement:
                remapped_timed.append({
                    "old_id": expected["id"], "current_id": replacement["id"],
                    "chapter": chapter["title"], "text": expected["text"],
                    "current": [replacement["start_ms"], replacement["end_ms"], replacement["status"]],
                })
                continue
            candidates = current.execute(
                "SELECT id,position,text,start_ms,end_ms,status FROM text_segments "
                "WHERE chapter_id=? ORDER BY position,id LIMIT 6",
                (chapter["id"],),
            ).fetchall()
            missing_timed.append({
                "id": expected["id"], "chapter_id": chapter["id"],
                "chapter": chapter["title"], "text": expected["text"],
                "current_start": [dict(row) for row in candidates],
            })
        elif (actual["start_ms"], actual["end_ms"], actual["status"]) != (
            expected["start_ms"], expected["end_ms"], expected["status"]
        ):
            changed_timed.append({
                "id": expected["id"], "chapter": chapter["title"], "text": expected["text"],
                "expected": [expected["start_ms"], expected["end_ms"], expected["status"]],
                "actual": [actual["start_ms"], actual["end_ms"], actual["status"]],
            })
    result = {
        "project": project.name,
        "expected_merged_pages": len(pairs),
        "missing_merged_targets": missing_titles,
        "wrong_first_spoken_title": wrong_first,
        "wrong_two_page_mapping": wrong_parts,
        "expected_timed_visible_titles": len(timed),
        "missing_timed_visible_titles": missing_timed,
        "changed_timed_visible_titles": changed_timed,
        "preserved_remapped_timed_titles": remapped_timed,
        "integrity": current.execute("PRAGMA integrity_check").fetchone()[0],
        "foreign_key_errors": len(current.execute("PRAGMA foreign_key_check").fetchall()),
    }
    old.close()
    current.close()
    return result


for raw in sys.argv[1:]:
    print(json.dumps(verify(Path(raw).resolve()), ensure_ascii=False))

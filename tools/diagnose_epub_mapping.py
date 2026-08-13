from __future__ import annotations

import argparse
import json
import sqlite3
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup


def visible_text(source: str) -> str:
    soup = BeautifulSoup(source, "html.parser")
    if soup.head:
        soup.head.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--chapter", type=int, default=1, help="one-based logical chapter")
    parser.add_argument("--segments", default="1,100", help="one-based segment numbers")
    arguments = parser.parse_args()
    project = arguments.project.resolve()
    connection = sqlite3.connect(project / "project.sqlite3")
    connection.row_factory = sqlite3.Row
    chapters = connection.execute("SELECT * FROM chapters ORDER BY position,id").fetchall()
    chapter = chapters[arguments.chapter - 1]
    segments = connection.execute(
        "SELECT * FROM text_segments WHERE chapter_id=? ORDER BY position,id", (chapter["id"],)
    ).fetchall()
    parts = connection.execute(
        "SELECT entry_path FROM chapter_source_parts WHERE chapter_id=? ORDER BY position",
        (chapter["id"],),
    ).fetchall()
    epub = next((project / "source").glob("*.epub"))
    requested = [int(value) - 1 for value in arguments.segments.split(",")]
    result: dict[str, object] = {
        "chapter": dict(chapter),
        "parts": [row["entry_path"] for row in parts],
        "segments": [],
    }
    pages = []
    with zipfile.ZipFile(epub) as archive:
        for part in parts:
            source = archive.read(part["entry_path"]).decode("utf-8", errors="replace")
            pages.append((part["entry_path"], visible_text(source)))
    for position in requested:
        if not 0 <= position < len(segments):
            continue
        segment = segments[position]
        target = " ".join(segment["text"].split())
        result["segments"].append({
            "number": position + 1,
            "row": dict(segment),
            "normalized_target": target,
            "page_hits": [
                {
                    "entry": entry,
                    "offset": document.find(target),
                    "page_head": document[:180],
                }
                for entry, document in pages
            ],
        })
    try:
        from audioalign.core.epub_media_overlay import _annotate_for_overlay
        from audioalign.core.storage import ProjectSession

        session = ProjectSession.open(project)
        try:
            model = next(item for item in session.repository.chapters() if item.id == chapter["id"])
            matched: set[int] = set()
            errors: list[str] = []
            with zipfile.ZipFile(epub) as archive:
                for part in parts:
                    source = archive.read(part["entry_path"]).decode("utf-8", errors="replace")
                    _rendered, _units, page_errors = _annotate_for_overlay(
                        source, session, model, missing_is_error=False,
                        matched_positions=matched,
                    )
                    errors.extend(page_errors)
            result["actual_mapping"] = {
                "matched_requested": [position + 1 for position in requested if position in matched],
                "unmatched_requested": [position + 1 for position in requested if position not in matched],
                "errors": errors,
            }
        finally:
            session.close()
    except Exception as exc:
        result["actual_mapping_error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

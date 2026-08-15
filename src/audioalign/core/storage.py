from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Sequence

from .models import (
    ASRToken,
    AudioAsset,
    AudioChapterMarker,
    BoundaryCandidate,
    Chapter,
    ChapterAudioLink,
    ProjectManifest,
    RecognitionChunk,
    RecognitionRun,
    SegmentOrigin,
    SegmentStatus,
    SourceDocument,
    SourceDocumentKind,
    SourceFragment,
    TextAudioAnchor,
    TextSegment,
)
from .text import display_chapter_title, source_fragments
from .paths import ApplicationPaths, sanitize_project_name


SUPPORTED_SCHEMA_VERSION = 2

SCHEMA_V2 = """
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    position INTEGER NOT NULL,
    source_html TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS source_fragments (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'paragraph',
    text TEXT NOT NULL,
    source_start_char INTEGER NOT NULL,
    source_end_char INTEGER NOT NULL,
    UNIQUE(chapter_id, position)
);
CREATE TABLE IF NOT EXISTS text_segments (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_ms INTEGER NOT NULL DEFAULT 0,
    end_ms INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'unmatched',
    locked INTEGER NOT NULL DEFAULT 0,
    origin TEXT NOT NULL DEFAULT 'source',
    source_fragment_id INTEGER REFERENCES source_fragments(id),
    source_start_char INTEGER,
    source_end_char INTEGER,
    UNIQUE(chapter_id, position)
);
CREATE TABLE IF NOT EXISTS audio_assets (
    id INTEGER PRIMARY KEY,
    absolute_path TEXT NOT NULL,
    relative_path TEXT,
    fingerprint TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER NOT NULL DEFAULT 0,
    sample_rate INTEGER NOT NULL DEFAULT 0,
    channels INTEGER NOT NULL DEFAULT 0,
    format TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS source_documents (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    original_path TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '',
    entry_path TEXT NOT NULL DEFAULT '',
    resource_root TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chapter_source_documents (
    chapter_id INTEGER PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    entry_path TEXT NOT NULL DEFAULT '',
    selector TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chapter_source_parts (
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    document_id INTEGER NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    entry_path TEXT NOT NULL DEFAULT '',
    selector TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(chapter_id, position)
);
CREATE TABLE IF NOT EXISTS source_fragment_locators (
    fragment_id INTEGER PRIMARY KEY REFERENCES source_fragments(id) ON DELETE CASCADE,
    dom_locator TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS audio_chapters (
    id INTEGER PRIMARY KEY,
    audio_id INTEGER NOT NULL REFERENCES audio_assets(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    UNIQUE(audio_id, position)
);
CREATE TABLE IF NOT EXISTS chapter_audio_links (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    audio_id INTEGER NOT NULL REFERENCES audio_assets(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    source_start_ms INTEGER NOT NULL DEFAULT 0,
    source_end_ms INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    UNIQUE(chapter_id, position)
);
CREATE TABLE IF NOT EXISTS chapter_media_state (
    chapter_id INTEGER PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
    mapping_signature TEXT NOT NULL DEFAULT '',
    needs_review INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS text_audio_anchors (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    segment_id INTEGER REFERENCES text_segments(id) ON DELETE CASCADE,
    source_start_char INTEGER NOT NULL,
    source_end_char INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    method TEXT NOT NULL DEFAULT 'asr'
);
CREATE TABLE IF NOT EXISTS asr_tokens (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    probability REAL NOT NULL DEFAULT 0,
    UNIQUE(chapter_id, position)
);
CREATE TABLE IF NOT EXISTS recognition_runs (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    cache_key TEXT NOT NULL UNIQUE,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'auto',
    audio_signature TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    actual_device TEXT NOT NULL DEFAULT '',
    compute_type TEXT NOT NULL DEFAULT '',
    device_name TEXT NOT NULL DEFAULT '',
    fallback_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recognition_chunks (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES recognition_runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    source_start_ms INTEGER NOT NULL,
    source_end_ms INTEGER NOT NULL,
    core_start_ms INTEGER NOT NULL,
    core_end_ms INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    transcript TEXT NOT NULL DEFAULT '',
    elapsed_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    UNIQUE(run_id, position)
);
CREATE TABLE IF NOT EXISTS recognition_tokens (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES recognition_runs(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES recognition_chunks(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    probability REAL NOT NULL DEFAULT 0,
    UNIQUE(chunk_id, position)
);
CREATE TABLE IF NOT EXISTS alignment_runs (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    recognition_run_id INTEGER NOT NULL REFERENCES recognition_runs(id) ON DELETE CASCADE,
    text_hash TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    silence_signature TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(chapter_id, recognition_run_id, text_hash, algorithm_version, silence_signature)
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS silence_candidates (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    time_ms INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    start_ms INTEGER,
    end_ms INTEGER,
    signature TEXT NOT NULL,
    UNIQUE(chapter_id, position)
);
CREATE INDEX IF NOT EXISTS idx_segments_chapter ON text_segments(chapter_id, position);
CREATE INDEX IF NOT EXISTS idx_source_fragments_chapter ON source_fragments(chapter_id, position);
CREATE INDEX IF NOT EXISTS idx_tokens_chapter ON asr_tokens(chapter_id, position);
CREATE INDEX IF NOT EXISTS idx_recognition_runs_chapter ON recognition_runs(chapter_id, backend, model);
CREATE INDEX IF NOT EXISTS idx_recognition_chunks_run ON recognition_chunks(run_id, position);
CREATE INDEX IF NOT EXISTS idx_recognition_tokens_run ON recognition_tokens(run_id, chunk_id, position);
CREATE INDEX IF NOT EXISTS idx_links_chapter ON chapter_audio_links(chapter_id, position);
CREATE INDEX IF NOT EXISTS idx_anchors_chapter ON text_audio_anchors(chapter_id, source_start_char);
CREATE INDEX IF NOT EXISTS idx_silence_chapter ON silence_candidates(chapter_id, signature, position);
PRAGMA user_version=2;
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_recognition_chunk(
    database: str | Path,
    chunk: RecognitionChunk,
    tokens: Sequence[ASRToken],
) -> None:
    """Commit one complete ASR chunk from a background worker connection."""
    connection = sqlite3.connect(Path(database), timeout=30)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN")
        connection.execute(
            """INSERT INTO recognition_chunks
            (run_id,position,source_start_ms,source_end_ms,core_start_ms,core_end_ms,status,transcript,elapsed_ms,error)
            VALUES (?,?,?,?,?,?,'complete',?,?,?)
            ON CONFLICT(run_id,position) DO UPDATE SET
            source_start_ms=excluded.source_start_ms,source_end_ms=excluded.source_end_ms,
            core_start_ms=excluded.core_start_ms,core_end_ms=excluded.core_end_ms,
            status='complete',transcript=excluded.transcript,elapsed_ms=excluded.elapsed_ms,error=''""",
            (chunk.run_id, chunk.position, chunk.source_start_ms, chunk.source_end_ms,
             chunk.core_start_ms, chunk.core_end_ms, chunk.transcript, chunk.elapsed_ms, ""),
        )
        chunk_id = int(connection.execute(
            "SELECT id FROM recognition_chunks WHERE run_id=? AND position=?",
            (chunk.run_id, chunk.position),
        ).fetchone()[0])
        connection.execute("DELETE FROM recognition_tokens WHERE chunk_id=?", (chunk_id,))
        connection.executemany(
            """INSERT INTO recognition_tokens
            (run_id,chunk_id,position,text,start_ms,end_ms,probability) VALUES (?,?,?,?,?,?,?)""",
            [(chunk.run_id, chunk_id, index, token.text, token.start_ms, token.end_ms, token.probability)
             for index, token in enumerate(tokens)],
        )
        connection.execute(
            "UPDATE recognition_runs SET status='running',updated_at=? WHERE id=?",
            (utc_now(), chunk.run_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


class UnsupportedProjectError(RuntimeError):
    """Raised when a normal application open sees anything except schema v2."""


def migrate_manifest(data: dict, project_name: str | None = None) -> dict:
    version = int(data.get("schema_version", 0))
    if version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedProjectError(f"不支持的项目格式（schema {version}）")
    payload = dict(data)
    if project_name:
        payload["project_id"] = sanitize_project_name(project_name)
    return payload


def fingerprint_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    size = source.stat().st_size
    digest.update(str(size).encode("ascii"))
    with source.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def full_fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProjectExistsError(FileExistsError):
    pass


class ProjectRepository:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.connection = sqlite3.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables:
            self.connection.executescript(SCHEMA_V2)
            self.connection.commit()
            return
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version != SUPPORTED_SCHEMA_VERSION:
            raise UnsupportedProjectError(f"项目数据库不是 schema v{SUPPORTED_SCHEMA_VERSION}")
        self.connection.executescript(SCHEMA_V2)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(text_segments)")}
        extensions = {
            "origin": "TEXT NOT NULL DEFAULT 'source'",
            "source_fragment_id": "INTEGER REFERENCES source_fragments(id)",
            "source_start_char": "INTEGER",
            "source_end_char": "INTEGER",
        }
        for name, declaration in extensions.items():
            if name not in columns:
                self.connection.execute(f"ALTER TABLE text_segments ADD COLUMN {name} {declaration}")
        audio_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(audio_assets)")}
        if "position" not in audio_columns:
            self.connection.execute("ALTER TABLE audio_assets ADD COLUMN position INTEGER NOT NULL DEFAULT 0")
            self.connection.execute("UPDATE audio_assets SET position=id - 1")
        # This index must be created after the idempotent column extension.
        # Otherwise executescript fails before ALTER TABLE can run for a schema
        # v2 project created before media ordering was introduced.
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_audio_position ON audio_assets(position, id)"
        )
        self.connection.commit()
        self._initialize_missing_source_fragments()

    def _initialize_missing_source_fragments(self) -> None:
        """Populate immutable blocks once for current projects without a format prompt."""
        chapters = self.connection.execute(
            """SELECT c.id,c.source_html FROM chapters c
            WHERE NOT EXISTS (SELECT 1 FROM source_fragments f WHERE f.chapter_id=c.id)"""
        ).fetchall()
        for chapter in chapters:
            segment_rows = self.connection.execute(
                "SELECT id,text FROM text_segments WHERE chapter_id=? ORDER BY position,id",
                (chapter["id"],),
            ).fetchall()
            fallback = "\n\n".join(row["text"] for row in segment_rows)
            fragments = source_fragments(chapter["source_html"], fallback)
            if fragments:
                self.replace_source_fragments(
                    chapter["id"],
                    [
                        SourceFragment(
                            None, chapter["id"], item.position, item.kind, item.text,
                            item.source_start_char, item.source_end_char,
                        )
                        for item in fragments
                    ],
                    assign_existing=True,
                )

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def counts(self) -> dict[str, int]:
        return {
            table: int(self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in ("chapters", "text_segments", "audio_assets", "asr_tokens")
        }

    def add_chapter(self, chapter: Chapter) -> int:
        cursor = self.connection.execute(
            "INSERT INTO chapters(title, position, source_html) VALUES (?, ?, ?)",
            (chapter.title, chapter.position, chapter.source_html),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def replace_source_fragments(
        self,
        chapter_id: int,
        fragments: Sequence[SourceFragment],
        *,
        assign_existing: bool = False,
    ) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM source_fragments WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                """INSERT INTO source_fragments
                (chapter_id,position,kind,text,source_start_char,source_end_char)
                VALUES (?,?,?,?,?,?)""",
                [
                    (chapter_id, item.position, item.kind, item.text,
                     item.source_start_char, item.source_end_char)
                    for item in fragments
                ],
            )
            if assign_existing:
                stored = connection.execute(
                    "SELECT * FROM source_fragments WHERE chapter_id=? ORDER BY position",
                    (chapter_id,),
                ).fetchall()
                cursor = 0
                rows = connection.execute(
                    "SELECT id,text FROM text_segments WHERE chapter_id=? ORDER BY position,id",
                    (chapter_id,),
                ).fetchall()
                for row in rows:
                    text = row["text"]
                    owner = next(
                        (item for item in stored
                         if item["source_start_char"] <= cursor
                         and cursor + len(text) <= item["source_end_char"]),
                        None,
                    )
                    if owner:
                        connection.execute(
                            """UPDATE text_segments SET origin='source',source_fragment_id=?,
                            source_start_char=?,source_end_char=? WHERE id=?""",
                            (owner["id"], cursor, cursor + len(text), row["id"]),
                        )
                    cursor += len(text)

    def source_fragments(self, chapter_id: int) -> list[SourceFragment]:
        return [SourceFragment(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM source_fragments WHERE chapter_id=? ORDER BY position,id",
            (chapter_id,),
        )]

    def add_source_document(self, document: SourceDocument) -> int:
        cursor = self.connection.execute(
            """INSERT INTO source_documents
            (kind,original_path,stored_path,fingerprint,entry_path,resource_root)
            VALUES (?,?,?,?,?,?)""",
            (document.kind.value, document.original_path, document.stored_path,
             document.fingerprint, document.entry_path, document.resource_root),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def source_document(self, document_id: int) -> SourceDocument | None:
        row = self.connection.execute(
            "SELECT * FROM source_documents WHERE id=?", (document_id,)
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["kind"] = SourceDocumentKind(data["kind"])
        return SourceDocument(**data)

    def set_chapter_source_document(
        self, chapter_id: int, document_id: int, *, entry_path: str = "", selector: str = "",
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO chapter_source_documents(chapter_id,document_id,entry_path,selector)
                VALUES (?,?,?,?) ON CONFLICT(chapter_id) DO UPDATE SET
                document_id=excluded.document_id,entry_path=excluded.entry_path,selector=excluded.selector""",
                (chapter_id, document_id, entry_path, selector),
            )
            connection.execute(
                """INSERT INTO chapter_source_parts(chapter_id,position,document_id,entry_path,selector)
                VALUES (?,?,?,?,?) ON CONFLICT(chapter_id,position) DO UPDATE SET
                document_id=excluded.document_id,entry_path=excluded.entry_path,selector=excluded.selector""",
                (chapter_id, 0, document_id, entry_path, selector),
            )

    def set_chapter_source_parts(
        self, chapter_id: int, document_id: int, parts: Sequence[tuple[str, str]],
    ) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM chapter_source_parts WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                """INSERT INTO chapter_source_parts
                (chapter_id,position,document_id,entry_path,selector) VALUES (?,?,?,?,?)""",
                [
                    (chapter_id, position, document_id, entry_path, selector)
                    for position, (entry_path, selector) in enumerate(parts)
                ],
            )

    def chapter_source_parts(
        self, chapter_id: int,
    ) -> list[tuple[SourceDocument, str, str]]:
        rows = self.connection.execute(
            """SELECT d.*,p.entry_path AS part_entry_path,p.selector AS part_selector
            FROM chapter_source_parts p JOIN source_documents d ON d.id=p.document_id
            WHERE p.chapter_id=? ORDER BY p.position""",
            (chapter_id,),
        )
        return [
            (
                SourceDocument(**{key: row[key] for key in (
                    "id", "kind", "original_path", "stored_path", "fingerprint",
                    "entry_path", "resource_root",
                )} | {"kind": SourceDocumentKind(row["kind"])}),
                row["part_entry_path"], row["part_selector"],
            )
            for row in rows
        ]

    def chapter_source_document(self, chapter_id: int) -> tuple[SourceDocument, str, str] | None:
        row = self.connection.execute(
            """SELECT d.*,c.entry_path AS chapter_entry_path,c.selector
            FROM chapter_source_documents c JOIN source_documents d ON d.id=c.document_id
            WHERE c.chapter_id=?""", (chapter_id,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        entry_path = data.pop("chapter_entry_path")
        selector = data.pop("selector")
        data["kind"] = SourceDocumentKind(data["kind"])
        return SourceDocument(**data), entry_path, selector

    def set_fragment_locator(self, fragment_id: int, locator: str) -> None:
        self.connection.execute(
            """INSERT INTO source_fragment_locators(fragment_id,dom_locator) VALUES (?,?)
            ON CONFLICT(fragment_id) DO UPDATE SET dom_locator=excluded.dom_locator""",
            (fragment_id, locator),
        )
        self.connection.commit()

    def fragment_locators(self, chapter_id: int) -> dict[int, str]:
        return {
            int(row["fragment_id"]): row["dom_locator"]
            for row in self.connection.execute(
                """SELECT l.fragment_id,l.dom_locator FROM source_fragment_locators l
                JOIN source_fragments f ON f.id=l.fragment_id WHERE f.chapter_id=?""",
                (chapter_id,),
            )
        }

    def chapters(self) -> list[Chapter]:
        chapters = [Chapter(**dict(row)) for row in self.connection.execute("SELECT * FROM chapters ORDER BY position,id")]
        for chapter in chapters:
            chapter.title = display_chapter_title(chapter.title, chapter.source_html)
        return chapters

    def replace_segments(self, chapter_id: int, segments: Sequence[TextSegment]) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM text_segments WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                """INSERT INTO text_segments
                (chapter_id,position,text,start_ms,end_ms,confidence,status,locked,origin,
                 source_fragment_id,source_start_char,source_end_char)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (chapter_id, s.position, s.text, s.start_ms, s.end_ms, s.confidence,
                     s.status.value, int(s.locked), s.origin.value, s.source_fragment_id,
                     s.source_start_char, s.source_end_char)
                    for s in segments
                ],
            )

    def replace_chapter_edit_state(
        self,
        chapter_id: int,
        segments: Sequence[TextSegment],
        anchors: Sequence[TextAudioAnchor],
    ) -> None:
        """Atomically restore one chapter without borrowing identity from another.

        Segment row ids are database-local and can change after a split/merge.
        Anchors are therefore rebound through the old-id/position map first and
        source character ranges second.
        """
        if any(segment.chapter_id != chapter_id for segment in segments):
            raise ValueError("Chapter edit state contains segments from another chapter")
        if any(anchor.chapter_id != chapter_id for anchor in anchors):
            raise ValueError("Chapter edit state contains anchors from another chapter")
        old_positions = {
            segment.id: position for position, segment in enumerate(segments)
            if segment.id is not None
        }
        with self.transaction() as connection:
            current_ids = [
                int(row["id"]) for row in connection.execute(
                    "SELECT id FROM text_segments WHERE chapter_id=? ORDER BY position,id",
                    (chapter_id,),
                )
            ]
            target_ids = [segment.id for segment in segments]
            if target_ids and all(value is not None for value in target_ids) and target_ids == current_ids:
                # The edit changed text/timing only.  Keep stable row ids instead
                # of deleting and reinserting an entire audiobook chapter.
                connection.executemany(
                    """UPDATE text_segments SET position=?,text=?,start_ms=?,end_ms=?,
                    confidence=?,status=?,locked=?,origin=?,source_fragment_id=?,
                    source_start_char=?,source_end_char=? WHERE id=? AND chapter_id=?""",
                    [
                        (
                            position, segment.text, segment.start_ms, segment.end_ms,
                            segment.confidence, segment.status.value, int(segment.locked),
                            segment.origin.value, segment.source_fragment_id,
                            segment.source_start_char, segment.source_end_char,
                            segment.id, chapter_id,
                        )
                        for position, segment in enumerate(segments)
                    ],
                )
                connection.execute(
                    "DELETE FROM text_audio_anchors WHERE chapter_id=?", (chapter_id,),
                )
                connection.executemany(
                    """INSERT INTO text_audio_anchors
                    (chapter_id,segment_id,source_start_char,source_end_char,start_ms,end_ms,confidence,method)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    [
                        (
                            chapter_id, anchor.segment_id, anchor.source_start_char,
                            anchor.source_end_char, anchor.start_ms, anchor.end_ms,
                            anchor.confidence, anchor.method,
                        )
                        for anchor in anchors
                    ],
                )
                return
            connection.execute("DELETE FROM text_audio_anchors WHERE chapter_id=?", (chapter_id,))
            connection.execute("DELETE FROM text_segments WHERE chapter_id=?", (chapter_id,))
            inserted: list[tuple[int, TextSegment]] = []
            for position, segment in enumerate(segments):
                cursor = connection.execute(
                    """INSERT INTO text_segments
                    (chapter_id,position,text,start_ms,end_ms,confidence,status,locked,origin,
                     source_fragment_id,source_start_char,source_end_char)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (chapter_id, position, segment.text, segment.start_ms, segment.end_ms,
                     segment.confidence, segment.status.value, int(segment.locked),
                     segment.origin.value, segment.source_fragment_id,
                     segment.source_start_char, segment.source_end_char),
                )
                inserted.append((int(cursor.lastrowid), segment))
            anchor_rows = []
            for anchor in anchors:
                target_id = None
                old_position = old_positions.get(anchor.segment_id)
                if old_position is not None and old_position < len(inserted):
                    target_id = inserted[old_position][0]
                if target_id is None:
                    target_id = next((
                        row_id for row_id, segment in inserted
                        if segment.source_start_char is not None
                        and segment.source_end_char is not None
                        and segment.source_start_char <= anchor.source_start_char
                        and anchor.source_end_char <= segment.source_end_char
                    ), None)
                anchor_rows.append((
                    chapter_id, target_id, anchor.source_start_char, anchor.source_end_char,
                    anchor.start_ms, anchor.end_ms, anchor.confidence, anchor.method,
                ))
            connection.executemany(
                """INSERT INTO text_audio_anchors
                (chapter_id,segment_id,source_start_char,source_end_char,start_ms,end_ms,confidence,method)
                VALUES (?,?,?,?,?,?,?,?)""",
                anchor_rows,
            )

    def segments(self, chapter_id: int) -> list[TextSegment]:
        rows = self.connection.execute("SELECT * FROM text_segments WHERE chapter_id=? ORDER BY position,id", (chapter_id,))
        return [
            TextSegment(
                id=row["id"], chapter_id=row["chapter_id"], position=row["position"], text=row["text"],
                start_ms=row["start_ms"], end_ms=row["end_ms"], confidence=row["confidence"],
                status=SegmentStatus(row["status"]), locked=bool(row["locked"]),
                origin=SegmentOrigin(row["origin"]), source_fragment_id=row["source_fragment_id"],
                source_start_char=row["source_start_char"], source_end_char=row["source_end_char"],
            ) for row in rows
        ]

    def update_segment(self, segment: TextSegment) -> None:
        segment.validate()
        self.connection.execute(
            """UPDATE text_segments SET text=?,start_ms=?,end_ms=?,confidence=?,status=?,locked=?,
            origin=?,source_fragment_id=?,source_start_char=?,source_end_char=? WHERE id=?""",
            (segment.text, segment.start_ms, segment.end_ms, segment.confidence, segment.status.value,
             int(segment.locked), segment.origin.value, segment.source_fragment_id,
             segment.source_start_char, segment.source_end_char, segment.id),
        )
        self.connection.commit()

    def update_segments(self, segments: Sequence[TextSegment]) -> None:
        """Persist one interactive multi-cue edit as a single transaction."""
        with self.transaction() as connection:
            connection.executemany(
                """UPDATE text_segments SET text=?,start_ms=?,end_ms=?,confidence=?,status=?,locked=?,
                origin=?,source_fragment_id=?,source_start_char=?,source_end_char=? WHERE id=?""",
                [
                    (segment.text, segment.start_ms, segment.end_ms, segment.confidence,
                     segment.status.value, int(segment.locked), segment.origin.value,
                     segment.source_fragment_id, segment.source_start_char,
                     segment.source_end_char, segment.id)
                    for segment in segments if segment.id is not None
                ],
            )

    def replace_one_segment(
        self,
        chapter_id: int,
        original_id: int,
        replacements: Sequence[TextSegment],
    ) -> None:
        """Replace one cue locally while preserving unrelated rows and anchors."""
        values = list(replacements)
        if not values or any(item.chapter_id != chapter_id for item in values):
            raise ValueError("Invalid local segment replacement")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT position FROM text_segments WHERE id=? AND chapter_id=?",
                (original_id, chapter_id),
            ).fetchone()
            if row is None:
                raise ValueError("The segment no longer exists")
            position = int(row["position"])
            delta = len(values) - 1
            if delta:
                connection.execute(
                    "UPDATE text_segments SET position=position+? WHERE chapter_id=? AND position>?",
                    (delta, chapter_id, position),
                )

            first = values[0]
            connection.execute(
                """UPDATE text_segments SET position=?,text=?,start_ms=?,end_ms=?,confidence=?,
                status=?,locked=?,origin=?,source_fragment_id=?,source_start_char=?,source_end_char=?
                WHERE id=? AND chapter_id=?""",
                (
                    position, first.text, first.start_ms, first.end_ms, first.confidence,
                    first.status.value, int(first.locked), first.origin.value,
                    first.source_fragment_id, first.source_start_char, first.source_end_char,
                    original_id, chapter_id,
                ),
            )
            inserted_ids = [original_id]
            for offset, segment in enumerate(values[1:], 1):
                cursor = connection.execute(
                    """INSERT INTO text_segments
                    (chapter_id,position,text,start_ms,end_ms,confidence,status,locked,origin,
                     source_fragment_id,source_start_char,source_end_char)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        chapter_id, position + offset, segment.text, segment.start_ms,
                        segment.end_ms, segment.confidence, segment.status.value,
                        int(segment.locked), segment.origin.value, segment.source_fragment_id,
                        segment.source_start_char, segment.source_end_char,
                    ),
                )
                inserted_ids.append(int(cursor.lastrowid))

            if len(inserted_ids) > 1:
                anchors = connection.execute(
                    "SELECT id,source_start_char,source_end_char,start_ms,end_ms "
                    "FROM text_audio_anchors WHERE chapter_id=? AND segment_id=?",
                    (chapter_id, original_id),
                ).fetchall()
                updates = []
                for anchor in anchors:
                    character_midpoint = (
                        int(anchor["source_start_char"]) + int(anchor["source_end_char"])
                    ) / 2
                    time_midpoint = (int(anchor["start_ms"]) + int(anchor["end_ms"])) / 2
                    target = 0
                    for index, segment in enumerate(values):
                        if (
                            segment.source_start_char is not None
                            and segment.source_end_char is not None
                            and segment.source_start_char <= character_midpoint <= segment.source_end_char
                        ):
                            target = index
                            break
                        if segment.start_ms <= time_midpoint <= segment.end_ms:
                            target = index
                    updates.append((inserted_ids[target], int(anchor["id"])))
                connection.executemany(
                    "UPDATE text_audio_anchors SET segment_id=? WHERE id=?", updates,
                )

    def merge_adjacent_segments(
        self, chapter_id: int, left_id: int, right_id: int, merged: TextSegment,
    ) -> None:
        """Merge adjacent rows without rewriting the chapter-sized anchor table."""
        if merged.chapter_id != chapter_id:
            raise ValueError("Invalid merged segment chapter")
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id,position FROM text_segments WHERE id IN (?,?) AND chapter_id=? ORDER BY position",
                (left_id, right_id, chapter_id),
            ).fetchall()
            if len(rows) != 2 or int(rows[1]["position"]) != int(rows[0]["position"]) + 1:
                raise ValueError("Segments are no longer adjacent")
            left_id, right_id = int(rows[0]["id"]), int(rows[1]["id"])
            position = int(rows[0]["position"])
            connection.execute(
                """UPDATE text_segments SET position=?,text=?,start_ms=?,end_ms=?,confidence=?,
                status=?,locked=?,origin=?,source_fragment_id=?,source_start_char=?,source_end_char=?
                WHERE id=? AND chapter_id=?""",
                (
                    position, merged.text, merged.start_ms, merged.end_ms, merged.confidence,
                    merged.status.value, int(merged.locked), merged.origin.value,
                    merged.source_fragment_id, merged.source_start_char, merged.source_end_char,
                    left_id, chapter_id,
                ),
            )
            connection.execute(
                "UPDATE text_audio_anchors SET segment_id=? WHERE chapter_id=? AND segment_id=?",
                (left_id, chapter_id, right_id),
            )
            connection.execute(
                "DELETE FROM text_segments WHERE id=? AND chapter_id=?", (right_id, chapter_id),
            )
            connection.execute(
                "UPDATE text_segments SET position=position-1 WHERE chapter_id=? AND position>?",
                (chapter_id, position + 1),
            )

    def mark_segments_unmatched(self, segment_ids: Sequence[int]) -> None:
        """Remove audio timing while preserving the source text and row identity."""
        ids = [int(value) for value in segment_ids if value is not None]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self.transaction() as connection:
            connection.execute(
                f"DELETE FROM text_audio_anchors WHERE segment_id IN ({placeholders})",
                ids,
            )
            connection.execute(
                f"""UPDATE text_segments
                SET start_ms=0,end_ms=0,confidence=0,status=?,locked=0
                WHERE id IN ({placeholders})""",
                [SegmentStatus.UNMATCHED.value, *ids],
            )

    def delete_segments(self, chapter_id: int, segment_ids: Sequence[int]) -> None:
        ids = [int(value) for value in segment_ids if value is not None]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self.transaction() as connection:
            connection.execute(f"DELETE FROM text_segments WHERE id IN ({placeholders})", ids)
            rows = connection.execute(
                "SELECT id FROM text_segments WHERE chapter_id=? ORDER BY position,id",
                (chapter_id,),
            ).fetchall()
            connection.executemany(
                "UPDATE text_segments SET position=? WHERE id=?",
                [(position, row["id"]) for position, row in enumerate(rows)],
            )

    def add_audio(self, asset: AudioAsset) -> int:
        existing = self.audio_by_path(asset.absolute_path)
        if existing is not None and existing.id is not None:
            return existing.id
        position = asset.position
        if position <= 0:
            position = int(self.connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM audio_assets"
            ).fetchone()[0])
        cursor = self.connection.execute(
            """INSERT INTO audio_assets
            (absolute_path,relative_path,fingerprint,duration_ms,sample_rate,channels,format,title,position)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (asset.absolute_path, asset.relative_path, asset.fingerprint, asset.duration_ms,
             asset.sample_rate, asset.channels, asset.format, asset.title, position),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def audio(self, audio_id: int) -> AudioAsset | None:
        row = self.connection.execute("SELECT * FROM audio_assets WHERE id=?", (audio_id,)).fetchone()
        return AudioAsset(**dict(row)) if row else None

    def all_audio(self) -> list[AudioAsset]:
        return [AudioAsset(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM audio_assets ORDER BY position,id"
        )]

    def audio_by_path(self, path: str | Path) -> AudioAsset | None:
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        for asset in self.all_audio():
            if os.path.normcase(os.path.abspath(asset.absolute_path)) == key:
                return asset
        return None

    def audio_by_fingerprint(self, fingerprint: str) -> list[AudioAsset]:
        if not fingerprint:
            return []
        return [AudioAsset(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM audio_assets WHERE fingerprint=? ORDER BY position,id", (fingerprint,)
        )]

    def update_audio(self, asset: AudioAsset) -> None:
        if asset.id is None:
            raise ValueError("Cannot update an unstored media asset")
        self.connection.execute(
            """UPDATE audio_assets SET absolute_path=?,relative_path=?,fingerprint=?,duration_ms=?,
            sample_rate=?,channels=?,format=?,title=?,position=? WHERE id=?""",
            (asset.absolute_path, asset.relative_path, asset.fingerprint, asset.duration_ms,
             asset.sample_rate, asset.channels, asset.format, asset.title, asset.position, asset.id),
        )
        self.connection.commit()

    def reorder_audio(self, ordered_ids: Sequence[int]) -> None:
        with self.transaction() as connection:
            connection.executemany(
                "UPDATE audio_assets SET position=? WHERE id=?",
                [(position, int(audio_id)) for position, audio_id in enumerate(ordered_ids)],
            )

    def audio_usage(self, audio_id: int) -> list[ChapterAudioLink]:
        return [ChapterAudioLink(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM chapter_audio_links WHERE audio_id=? ORDER BY chapter_id,position", (audio_id,)
        )]

    def remove_audio(self, audio_id: int, *, unlink: bool = False) -> None:
        with self.transaction() as connection:
            usage = int(connection.execute(
                "SELECT count(*) FROM chapter_audio_links WHERE audio_id=?", (audio_id,)
            ).fetchone()[0])
            if usage and not unlink:
                raise ValueError("Media asset is still linked to one or more chapters")
            if unlink:
                connection.execute("DELETE FROM chapter_audio_links WHERE audio_id=?", (audio_id,))
            connection.execute("DELETE FROM audio_assets WHERE id=?", (audio_id,))
            rows = connection.execute("SELECT id FROM audio_assets ORDER BY position,id").fetchall()
            connection.executemany(
                "UPDATE audio_assets SET position=? WHERE id=?",
                [(position, row["id"]) for position, row in enumerate(rows)],
            )

    def replace_media_library(
        self,
        assets: Sequence[AudioAsset],
        markers: dict[int, Sequence[AudioChapterMarker]],
        links: Sequence[ChapterAudioLink],
        new_chapters: Sequence[Chapter] = (),
    ) -> dict[int, int]:
        """Apply a staged media library and all mappings in one transaction.

        Negative asset ids identify rows that only exist in the dialog draft.
        The returned map resolves those temporary ids to database ids.
        """
        normalized_paths: dict[str, int | None] = {}
        for asset in assets:
            path_key = os.path.normcase(os.path.abspath(asset.absolute_path))
            if path_key in normalized_paths:
                raise ValueError("The staged media library contains the same path more than once")
            normalized_paths[path_key] = asset.id
        existing_ids = {
            int(row[0]) for row in self.connection.execute("SELECT id FROM audio_assets")
        }
        staged_existing = {int(asset.id) for asset in assets if asset.id is not None and asset.id > 0}
        unknown = staged_existing - existing_ids
        if unknown:
            raise ValueError(f"Unknown media ids in staged library: {sorted(unknown)}")
        id_map: dict[int, int] = {}
        chapter_map: dict[int, int] = {}
        with self.transaction() as connection:
            old_mapping: dict[int, tuple[tuple[str, int, int], ...]] = {}
            for row in connection.execute(
                """SELECT links.chapter_id,assets.fingerprint,
                links.source_start_ms,links.source_end_ms
                FROM chapter_audio_links AS links
                JOIN audio_assets AS assets ON assets.id=links.audio_id
                ORDER BY links.chapter_id,links.position"""
            ):
                old_mapping.setdefault(int(row["chapter_id"]), tuple())
                old_mapping[int(row["chapter_id"])] += ((
                    str(row["fingerprint"]), int(row["source_start_ms"]), int(row["source_end_ms"]),
                ),)
            next_position = int(connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM chapters"
            ).fetchone()[0])
            for chapter in new_chapters:
                if chapter.id is None or chapter.id >= 0:
                    raise ValueError("New staged chapters must have temporary negative ids")
                cursor = connection.execute(
                    "INSERT INTO chapters(title,position,source_html) VALUES (?,?,?)",
                    (chapter.title, next_position, chapter.source_html),
                )
                chapter_map[int(chapter.id)] = int(cursor.lastrowid)
                next_position += 1
            for position, asset in enumerate(assets):
                if asset.id is not None and asset.id > 0:
                    connection.execute(
                        """UPDATE audio_assets SET absolute_path=?,relative_path=?,fingerprint=?,
                        duration_ms=?,sample_rate=?,channels=?,format=?,title=?,position=? WHERE id=?""",
                        (asset.absolute_path, asset.relative_path, asset.fingerprint,
                         asset.duration_ms, asset.sample_rate, asset.channels, asset.format,
                         asset.title, position, asset.id),
                    )
                    id_map[int(asset.id)] = int(asset.id)
                else:
                    cursor = connection.execute(
                        """INSERT INTO audio_assets
                        (absolute_path,relative_path,fingerprint,duration_ms,sample_rate,channels,format,title,position)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                        (asset.absolute_path, asset.relative_path, asset.fingerprint,
                         asset.duration_ms, asset.sample_rate, asset.channels, asset.format,
                         asset.title, position),
                    )
                    if asset.id is None:
                        raise ValueError("New staged media must have a temporary negative id")
                    id_map[int(asset.id)] = int(cursor.lastrowid)

            removed = existing_ids - staged_existing
            connection.execute("DELETE FROM chapter_audio_links")
            for audio_id in removed:
                connection.execute("DELETE FROM audio_assets WHERE id=?", (audio_id,))

            for source_id, items in markers.items():
                target_id = id_map.get(int(source_id))
                if target_id is None:
                    continue
                connection.execute("DELETE FROM audio_chapters WHERE audio_id=?", (target_id,))
                connection.executemany(
                    """INSERT INTO audio_chapters(audio_id,position,title,start_ms,end_ms)
                    VALUES (?,?,?,?,?)""",
                    [(target_id, item.position, item.title, item.start_ms, item.end_ms) for item in items],
                )

            positions: dict[int, int] = {}
            for link in links:
                target_audio_id = id_map.get(int(link.audio_id))
                if target_audio_id is None:
                    raise ValueError("A chapter mapping references removed media")
                target_chapter_id = chapter_map.get(link.chapter_id, link.chapter_id)
                position = positions.get(target_chapter_id, 0)
                positions[target_chapter_id] = position + 1
                connection.execute(
                    """INSERT INTO chapter_audio_links
                    (chapter_id,audio_id,position,source_start_ms,source_end_ms,confidence)
                    VALUES (?,?,?,?,?,?)""",
                    (target_chapter_id, target_audio_id, position,
                     max(0, link.source_start_ms), max(link.source_start_ms, link.source_end_ms),
                     link.confidence),
                )
            new_mapping: dict[int, tuple[tuple[str, int, int], ...]] = {}
            for row in connection.execute(
                """SELECT links.chapter_id,assets.fingerprint,
                links.source_start_ms,links.source_end_ms
                FROM chapter_audio_links AS links
                JOIN audio_assets AS assets ON assets.id=links.audio_id
                ORDER BY links.chapter_id,links.position"""
            ):
                new_mapping.setdefault(int(row["chapter_id"]), tuple())
                new_mapping[int(row["chapter_id"])] += ((
                    str(row["fingerprint"]), int(row["source_start_ms"]), int(row["source_end_ms"]),
                ),)
            for chapter_id in set(old_mapping) | set(new_mapping):
                signature = hashlib.sha256(repr(new_mapping.get(chapter_id, ())).encode("utf-8")).hexdigest()
                changed = bool(old_mapping.get(chapter_id)) and old_mapping.get(chapter_id) != new_mapping.get(chapter_id)
                connection.execute(
                    """INSERT INTO chapter_media_state(chapter_id,mapping_signature,needs_review)
                    VALUES (?,?,?) ON CONFLICT(chapter_id) DO UPDATE SET
                    mapping_signature=excluded.mapping_signature,
                    needs_review=MAX(chapter_media_state.needs_review,excluded.needs_review)""",
                    (chapter_id, signature, int(changed)),
                )
        return id_map

    def chapter_media_needs_review(self, chapter_id: int) -> bool:
        row = self.connection.execute(
            "SELECT needs_review FROM chapter_media_state WHERE chapter_id=?", (chapter_id,)
        ).fetchone()
        return bool(row and row[0])

    def set_chapter_media_review(self, chapter_id: int, needs_review: bool) -> None:
        self.connection.execute(
            """INSERT INTO chapter_media_state(chapter_id,needs_review) VALUES (?,?)
            ON CONFLICT(chapter_id) DO UPDATE SET needs_review=excluded.needs_review""",
            (chapter_id, int(needs_review)),
        )
        self.connection.commit()

    def add_audio_chapters(self, markers: Sequence[AudioChapterMarker]) -> None:
        if not markers:
            return
        audio_id = markers[0].audio_id
        with self.transaction() as connection:
            connection.execute("DELETE FROM audio_chapters WHERE audio_id=?", (audio_id,))
            connection.executemany(
                "INSERT INTO audio_chapters(audio_id,position,title,start_ms,end_ms) VALUES (?,?,?,?,?)",
                [(m.audio_id, m.position, m.title, m.start_ms, m.end_ms) for m in markers],
            )

    def audio_chapters(self, audio_id: int) -> list[AudioChapterMarker]:
        return [AudioChapterMarker(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM audio_chapters WHERE audio_id=? ORDER BY position", (audio_id,)
        )]

    def set_chapter_links(self, chapter_id: int, links: Sequence[ChapterAudioLink]) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM chapter_audio_links WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                """INSERT INTO chapter_audio_links
                (chapter_id,audio_id,position,source_start_ms,source_end_ms,confidence)
                VALUES (?,?,?,?,?,?)""",
                [(chapter_id, l.audio_id, i, l.source_start_ms, l.source_end_ms, l.confidence) for i, l in enumerate(links)],
            )

    def chapter_links(self, chapter_id: int) -> list[ChapterAudioLink]:
        return [ChapterAudioLink(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM chapter_audio_links WHERE chapter_id=? ORDER BY position", (chapter_id,)
        )]

    def all_links(self) -> list[ChapterAudioLink]:
        return [ChapterAudioLink(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM chapter_audio_links ORDER BY chapter_id,position"
        )]

    def replace_all_chapter_links(self, links: Sequence[ChapterAudioLink]) -> None:
        """Replace every chapter/audio mapping in one transaction."""
        with self.transaction() as connection:
            connection.execute("DELETE FROM chapter_audio_links")
            connection.executemany(
                """INSERT INTO chapter_audio_links
                (chapter_id,audio_id,position,source_start_ms,source_end_ms,confidence)
                VALUES (?,?,?,?,?,?)""",
                [
                    (
                        link.chapter_id,
                        link.audio_id,
                        link.position,
                        max(0, link.source_start_ms),
                        max(link.source_start_ms, link.source_end_ms),
                        link.confidence,
                    )
                    for link in links
                ],
            )

    def audio_for_chapter(self, chapter_id: int) -> AudioAsset | None:
        row = self.connection.execute(
            """SELECT a.* FROM audio_assets a JOIN chapter_audio_links l ON l.audio_id=a.id
            WHERE l.chapter_id=? ORDER BY l.position LIMIT 1""", (chapter_id,),
        ).fetchone()
        return AudioAsset(**dict(row)) if row else None

    def add_audio_and_link(self, chapter_id: int, asset: AudioAsset) -> int:
        audio_id = self.add_audio(asset)
        self.set_chapter_links(chapter_id, [ChapterAudioLink(None, chapter_id, audio_id, 0, 0, asset.duration_ms, 1.0)])
        return audio_id

    def replace_asr_tokens(self, chapter_id: int, tokens: Sequence[ASRToken]) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM asr_tokens WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                "INSERT INTO asr_tokens(chapter_id,position,text,start_ms,end_ms,probability) VALUES (?,?,?,?,?,?)",
                [(chapter_id, t.position, t.text, t.start_ms, t.end_ms, t.probability) for t in tokens],
            )

    def asr_tokens(self, chapter_id: int) -> list[ASRToken]:
        return [ASRToken(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM asr_tokens WHERE chapter_id=? ORDER BY position", (chapter_id,)
        )]

    def recognition_run(self, cache_key: str) -> RecognitionRun | None:
        row = self.connection.execute(
            "SELECT * FROM recognition_runs WHERE cache_key=?", (cache_key,)
        ).fetchone()
        return RecognitionRun(**dict(row)) if row else None

    def ensure_recognition_run(
        self,
        *,
        chapter_id: int,
        cache_key: str,
        backend: str,
        model: str,
        language: str,
        audio_signature: str,
        parameters_json: str,
    ) -> RecognitionRun:
        existing = self.recognition_run(cache_key)
        if existing:
            return existing
        now = utc_now()
        cursor = self.connection.execute(
            """INSERT INTO recognition_runs
            (chapter_id,cache_key,backend,model,language,audio_signature,parameters_json,status,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,'pending',?,?)""",
            (chapter_id, cache_key, backend, model, language, audio_signature, parameters_json, now, now),
        )
        self.connection.commit()
        return RecognitionRun(
            int(cursor.lastrowid), chapter_id, cache_key, backend, model, language,
            audio_signature, parameters_json, "pending", created_at=now, updated_at=now,
        )

    def reset_recognition_run(self, cache_key: str) -> None:
        self.connection.execute("DELETE FROM recognition_runs WHERE cache_key=?", (cache_key,))
        self.connection.commit()

    def recognition_chunks(self, run_id: int) -> list[RecognitionChunk]:
        return [RecognitionChunk(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM recognition_chunks WHERE run_id=? ORDER BY position", (run_id,)
        )]

    def recognition_tokens(self, run_id: int) -> list[ASRToken]:
        rows = self.connection.execute(
            """SELECT t.text,t.start_ms,t.end_ms,t.probability
            FROM recognition_tokens t JOIN recognition_chunks c ON c.id=t.chunk_id
            WHERE t.run_id=? AND c.status='complete'
            ORDER BY c.position,t.position""",
            (run_id,),
        )
        return [
            ASRToken(None, 0, position, row["text"], row["start_ms"], row["end_ms"], row["probability"])
            for position, row in enumerate(rows)
        ]

    def complete_recognition_run(self, run_id: int, device) -> None:
        self.connection.execute(
            """UPDATE recognition_runs SET status='complete',actual_device=?,compute_type=?,device_name=?,
            fallback_reason=?,updated_at=? WHERE id=?""",
            (device.actual_device, device.compute_type, device.device_name, device.fallback_reason, utc_now(), run_id),
        )
        self.connection.commit()

    def alignment_is_current(
        self,
        chapter_id: int,
        recognition_run_id: int,
        text_hash: str,
        algorithm_version: str,
        silence_signature: str,
    ) -> bool:
        row = self.connection.execute(
            """SELECT 1 FROM alignment_runs
            WHERE chapter_id=? AND recognition_run_id=? AND text_hash=?
              AND algorithm_version=? AND silence_signature=?""",
            (chapter_id, recognition_run_id, text_hash, algorithm_version, silence_signature),
        ).fetchone()
        return row is not None

    def record_alignment_run(
        self,
        chapter_id: int,
        recognition_run_id: int,
        text_hash: str,
        algorithm_version: str,
        silence_signature: str,
    ) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO alignment_runs
            (chapter_id,recognition_run_id,text_hash,algorithm_version,silence_signature,created_at)
            VALUES (?,?,?,?,?,?)""",
            (chapter_id, recognition_run_id, text_hash, algorithm_version, silence_signature, utc_now()),
        )
        self.connection.commit()

    def delete_recognition_cache(self, chapter_id: int, backend: str | None = None, model: str | None = None) -> int:
        clauses = ["chapter_id=?"]
        values: list[object] = [chapter_id]
        if backend:
            clauses.append("backend=?")
            values.append(backend)
        if model:
            clauses.append("model=?")
            values.append(model)
        cursor = self.connection.execute(
            f"DELETE FROM recognition_runs WHERE {' AND '.join(clauses)}", values
        )
        self.connection.commit()
        return cursor.rowcount

    def replace_anchors(self, chapter_id: int, anchors: Sequence[TextAudioAnchor]) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM text_audio_anchors WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                """INSERT INTO text_audio_anchors
                (chapter_id,segment_id,source_start_char,source_end_char,start_ms,end_ms,confidence,method)
                VALUES (?,?,?,?,?,?,?,?)""",
                [(chapter_id, a.segment_id, a.source_start_char, a.source_end_char, a.start_ms, a.end_ms, a.confidence, a.method) for a in anchors],
            )

    def anchors(self, chapter_id: int) -> list[TextAudioAnchor]:
        return [TextAudioAnchor(**dict(row)) for row in self.connection.execute(
            "SELECT * FROM text_audio_anchors WHERE chapter_id=? ORDER BY source_start_char", (chapter_id,)
        )]

    def replace_silence_candidates(
        self, chapter_id: int, candidates: Sequence[BoundaryCandidate], signature: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM silence_candidates WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                """INSERT INTO silence_candidates
                (chapter_id,position,time_ms,score,start_ms,end_ms,signature)
                VALUES (?,?,?,?,?,?,?)""",
                [(chapter_id, index, c.time_ms, c.score, c.start_ms, c.end_ms, signature)
                 for index, c in enumerate(candidates)],
            )

    def silence_candidates(self, chapter_id: int, signature: str) -> list[BoundaryCandidate]:
        rows = self.connection.execute(
            """SELECT time_ms,score,start_ms,end_ms FROM silence_candidates
            WHERE chapter_id=? AND signature=? ORDER BY position""",
            (chapter_id, signature),
        )
        return [BoundaryCandidate(**dict(row)) for row in rows]

    def invalidate_silence_candidates(self, chapter_id: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM silence_candidates WHERE chapter_id=?", (chapter_id,))

    def replace_segments_and_anchors(
        self,
        chapter_id: int,
        segments: Sequence[TextSegment],
        anchors: Sequence[TextAudioAnchor],
    ) -> None:
        """Atomically replace timings and their character/audio anchors."""
        with self.transaction() as connection:
            connection.execute("DELETE FROM text_audio_anchors WHERE chapter_id=?", (chapter_id,))
            connection.execute("DELETE FROM text_segments WHERE chapter_id=?", (chapter_id,))
            inserted: list[tuple[int, int, int]] = []
            source_offset = 0
            for segment in segments:
                cursor = connection.execute(
                    """INSERT INTO text_segments
                    (chapter_id,position,text,start_ms,end_ms,confidence,status,locked,origin,
                     source_fragment_id,source_start_char,source_end_char)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (chapter_id, segment.position, segment.text, segment.start_ms, segment.end_ms,
                     segment.confidence, segment.status.value, int(segment.locked), segment.origin.value,
                     segment.source_fragment_id, segment.source_start_char, segment.source_end_char),
                )
                end_offset = source_offset + len(segment.text)
                inserted.append((source_offset, end_offset, int(cursor.lastrowid)))
                source_offset = end_offset
            anchor_rows = []
            for anchor in anchors:
                segment_id = next(
                    (item_id for start, end, item_id in inserted
                     if start <= anchor.source_start_char < max(start + 1, end)),
                    None,
                )
                anchor_rows.append((
                    chapter_id, segment_id, anchor.source_start_char, anchor.source_end_char,
                    anchor.start_ms, anchor.end_ms, anchor.confidence, anchor.method,
                ))
            connection.executemany(
                """INSERT INTO text_audio_anchors
                (chapter_id,segment_id,source_start_char,source_end_char,start_ms,end_ms,confidence,method)
                VALUES (?,?,?,?,?,?,?,?)""",
                anchor_rows,
            )
    def replace_recognition_alignment(
        self,
        chapter_id: int,
        tokens: Sequence[ASRToken],
        segments: Sequence[TextSegment],
        anchors: Sequence[TextAudioAnchor],
    ) -> None:
        """Atomically publish one chapter's complete ASR and alignment result."""
        with self.transaction() as connection:
            connection.execute("DELETE FROM text_audio_anchors WHERE chapter_id=?", (chapter_id,))
            connection.execute("DELETE FROM asr_tokens WHERE chapter_id=?", (chapter_id,))
            connection.execute("DELETE FROM text_segments WHERE chapter_id=?", (chapter_id,))
            connection.executemany(
                "INSERT INTO asr_tokens(chapter_id,position,text,start_ms,end_ms,probability) VALUES (?,?,?,?,?,?)",
                [
                    (chapter_id, token.position, token.text, token.start_ms, token.end_ms, token.probability)
                    for token in tokens
                ],
            )
            inserted: list[tuple[int, int, int]] = []
            source_offset = 0
            for segment in segments:
                cursor = connection.execute(
                    """INSERT INTO text_segments
                    (chapter_id,position,text,start_ms,end_ms,confidence,status,locked,origin,
                     source_fragment_id,source_start_char,source_end_char)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (chapter_id, segment.position, segment.text, segment.start_ms, segment.end_ms,
                     segment.confidence, segment.status.value, int(segment.locked), segment.origin.value,
                     segment.source_fragment_id, segment.source_start_char, segment.source_end_char),
                )
                end_offset = source_offset + len(segment.text)
                inserted.append((source_offset, end_offset, int(cursor.lastrowid)))
                source_offset = end_offset
            anchor_rows = []
            for anchor in anchors:
                segment_id = next(
                    (item_id for start, end, item_id in inserted
                     if start <= anchor.source_start_char < max(start + 1, end)),
                    None,
                )
                anchor_rows.append((
                    chapter_id, segment_id, anchor.source_start_char, anchor.source_end_char,
                    anchor.start_ms, anchor.end_ms, anchor.confidence, anchor.method,
                ))
            connection.executemany(
                """INSERT INTO text_audio_anchors
                (chapter_id,segment_id,source_start_char,source_end_char,start_ms,end_ms,confidence,method)
                VALUES (?,?,?,?,?,?,?,?)""",
                anchor_rows,
            )
            connection.execute(
                "UPDATE chapter_media_state SET needs_review=0 WHERE chapter_id=?",
                (chapter_id,),
            )


class ProjectSession:
    def __init__(self, root: Path, manifest: ProjectManifest, *, archive_path: Path | None = None) -> None:
        self.root = root
        self.manifest = manifest
        self.archive_path = archive_path
        self.dirty = False
        self.repository = ProjectRepository(root / "project.sqlite3")

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @classmethod
    def create(cls, title: str, root: str | Path | None = None, *, internal: bool = True) -> "ProjectSession":
        name = sanitize_project_name(title)
        paths = ApplicationPaths.current()
        paths.ensure()
        root_path = Path(root) if root is not None else paths.project_dir(name)
        if root_path.exists():
            raise ProjectExistsError(f"项目已存在：{root_path}")
        root_path.mkdir(parents=True)
        for directory in ("source", "media", "cache"):
            (root_path / directory).mkdir()
        now = utc_now()
        manifest = ProjectManifest(project_id=root_path.name, title=title.strip(), created_at=now, updated_at=now)
        session = cls(root_path, manifest)
        session.save()
        return session

    @classmethod
    def open(cls, path: str | Path) -> "ProjectSession":
        source = Path(path)
        paths = ApplicationPaths.current()
        paths.ensure()
        if source.is_dir():
            root, archive_path = source, None
        elif source.suffix.lower() == ".aatproj":
            root = paths.work / sanitize_project_name(source.stem)
            marker = root / ".archive-source"
            expected = str(source.resolve())
            needs_extract = not (root / "manifest.json").exists() or not marker.exists() or marker.read_text("utf-8") != expected
            if needs_extract:
                if root.exists():
                    shutil.rmtree(root)
                root.mkdir(parents=True)
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(root)
                marker.write_text(expected, encoding="utf-8")
            archive_path = source
        else:
            raise ValueError("项目必须是文件夹或 .aatproj 文件")
        for directory in ("source", "media", "cache"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        data = migrate_manifest(json.loads((root / "manifest.json").read_text("utf-8")), root.name)
        manifest = ProjectManifest.from_dict(data)
        session = cls(root, manifest, archive_path=archive_path)
        session.write_manifest()
        return session

    def write_manifest(self) -> None:
        self.manifest.schema_version = SUPPORTED_SCHEMA_VERSION
        self.manifest.project_id = self.root.name
        self.manifest.updated_at = utc_now()
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def mark_dirty(self) -> None:
        self.dirty = True

    def autosave(self) -> None:
        self.repository.connection.commit()
        self.write_manifest()
        if not self.archive_path:
            self.dirty = False

    def save(self) -> None:
        self.repository.connection.commit()
        self.repository.connection.execute("PRAGMA wal_checkpoint(FULL)")
        self.write_manifest()
        if self.archive_path:
            self._pack_archive(self.archive_path)
        self.dirty = False

    def _pack_archive(self, destination: Path, *, include_cache: bool = True) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=destination.stem + "-", suffix=".tmp", dir=destination.parent)
        os.close(handle)
        temporary = Path(name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for item in self.root.rglob("*"):
                    if not item.is_file() or item.name in {".archive-source", "project.sqlite3-wal", "project.sqlite3-shm"}:
                        continue
                    relative = item.relative_to(self.root)
                    if not include_cache and relative.parts[0] == "cache":
                        continue
                    compression = zipfile.ZIP_STORED if relative.parts[0] == "media" else zipfile.ZIP_DEFLATED
                    archive.write(item, relative.as_posix(), compress_type=compression)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def save_as_archive(self, destination: str | Path, *, include_audio: bool = False, include_cache: bool = True) -> Path:
        target = Path(destination)
        if target.suffix.lower() != ".aatproj":
            target = target.with_suffix(".aatproj")
        if include_audio:
            self._copy_referenced_audio()
        self.write_manifest()
        self._pack_archive(target, include_cache=include_cache)
        return target

    def save_as_folder(self, destination: str | Path, *, include_audio: bool = False, include_cache: bool = True) -> Path:
        target = Path(destination)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError("目标项目文件夹必须为空")
        target.mkdir(parents=True, exist_ok=True)
        self.repository.connection.commit()
        self.write_manifest()
        for directory in ("source", "media", "cache"):
            (target / directory).mkdir(exist_ok=True)
        for item in self.root.rglob("*"):
            if not item.is_file() or item.name in {".archive-source", "project.sqlite3-wal", "project.sqlite3-shm"}:
                continue
            relative = item.relative_to(self.root)
            if not include_cache and relative.parts[0] == "cache":
                continue
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, output)
        clone = ProjectSession.open(target)
        try:
            if include_audio:
                clone._copy_referenced_audio()
            clone.save()
        finally:
            clone.close()
        return target

    def _copy_referenced_audio(self) -> None:
        media = self.root / "media"
        for asset in self.repository.all_audio():
            source = self.resolve_audio(asset)
            if not source:
                continue
            destination = media / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            self.repository.connection.execute(
                "UPDATE audio_assets SET absolute_path=?,relative_path=? WHERE id=?",
                (str(destination.resolve()), str(destination.relative_to(self.root)), asset.id),
            )
        self.repository.connection.commit()

    def resolve_audio(self, asset: AudioAsset | None) -> Path | None:
        if asset is None:
            return None
        candidates = [Path(asset.absolute_path)]
        if asset.relative_path:
            candidates.insert(0, self.root / asset.relative_path)
        candidates.extend([self.root / Path(asset.absolute_path).name, self.root.parent / Path(asset.absolute_path).name, self.root / "media" / Path(asset.absolute_path).name])
        return next((path for path in candidates if path.exists()), None)

    def close(self) -> None:
        self.repository.close()

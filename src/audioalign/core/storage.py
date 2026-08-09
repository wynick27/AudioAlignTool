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
    SegmentStatus,
    TextAudioAnchor,
    TextSegment,
)
from .text import display_chapter_title
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
    title TEXT NOT NULL DEFAULT ''
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
        self.connection.commit()

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
                (chapter_id,position,text,start_ms,end_ms,confidence,status,locked)
                VALUES (?,?,?,?,?,?,?,?)""",
                [
                    (chapter_id, s.position, s.text, s.start_ms, s.end_ms, s.confidence, s.status.value, int(s.locked))
                    for s in segments
                ],
            )

    def segments(self, chapter_id: int) -> list[TextSegment]:
        rows = self.connection.execute("SELECT * FROM text_segments WHERE chapter_id=? ORDER BY position,id", (chapter_id,))
        return [
            TextSegment(
                id=row["id"], chapter_id=row["chapter_id"], position=row["position"], text=row["text"],
                start_ms=row["start_ms"], end_ms=row["end_ms"], confidence=row["confidence"],
                status=SegmentStatus(row["status"]), locked=bool(row["locked"]),
            ) for row in rows
        ]

    def update_segment(self, segment: TextSegment) -> None:
        segment.validate()
        self.connection.execute(
            "UPDATE text_segments SET text=?,start_ms=?,end_ms=?,confidence=?,status=?,locked=? WHERE id=?",
            (segment.text, segment.start_ms, segment.end_ms, segment.confidence, segment.status.value, int(segment.locked), segment.id),
        )
        self.connection.commit()

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

    def add_audio(self, asset: AudioAsset) -> int:
        cursor = self.connection.execute(
            """INSERT INTO audio_assets
            (absolute_path,relative_path,fingerprint,duration_ms,sample_rate,channels,format,title)
            VALUES (?,?,?,?,?,?,?,?)""",
            (asset.absolute_path, asset.relative_path, asset.fingerprint, asset.duration_ms, asset.sample_rate, asset.channels, asset.format, asset.title),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def audio(self, audio_id: int) -> AudioAsset | None:
        row = self.connection.execute("SELECT * FROM audio_assets WHERE id=?", (audio_id,)).fetchone()
        return AudioAsset(**dict(row)) if row else None

    def all_audio(self) -> list[AudioAsset]:
        return [AudioAsset(**dict(row)) for row in self.connection.execute("SELECT * FROM audio_assets ORDER BY id")]

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
                    (chapter_id,position,text,start_ms,end_ms,confidence,status,locked)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (chapter_id, segment.position, segment.text, segment.start_ms, segment.end_ms,
                     segment.confidence, segment.status.value, int(segment.locked)),
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

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audioalign.core.exporters import export_html, export_json, export_subtitles
from audioalign.core.models import (
    ASRToken, BoundaryCandidate, Chapter, InferenceDeviceInfo, RecognitionChunk,
    SegmentOrigin, SegmentStatus, SourceFragment, TextSegment,
)
from audioalign.core.storage import (
    ProjectSession, UnsupportedProjectError, migrate_manifest, write_recognition_chunk,
)


class StorageExportTests(unittest.TestCase):
    def test_source_fragment_and_segment_provenance_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("source-test", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0, "<p>原始段落。</p>"))
            session.repository.replace_source_fragments(
                chapter_id,
                [SourceFragment(None, chapter_id, 0, "p", "原始段落。", 0, 5)],
            )
            fragment = session.repository.source_fragments(chapter_id)[0]
            session.repository.replace_segments(chapter_id, [TextSegment(
                None, chapter_id, 0, "原始段落。", origin=SegmentOrigin.SOURCE,
                source_fragment_id=fragment.id, source_start_char=0, source_end_char=5,
            )])
            loaded = session.repository.segments(chapter_id)[0]
            self.assertEqual(SegmentOrigin.SOURCE, loaded.origin)
            self.assertEqual(fragment.id, loaded.source_fragment_id)
            session.close()

    def test_chunked_recognition_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("cache-test", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            run = session.repository.ensure_recognition_run(
                chapter_id=chapter_id, cache_key="key", backend="faster-whisper", model="small",
                language="es", audio_signature="audio", parameters_json="{}",
            )
            chunk = RecognitionChunk(None, run.id or 0, 0, 0, 10_000, 0, 9_000, "complete", "hola", 25, "")
            write_recognition_chunk(
                session.repository.database, chunk,
                [ASRToken(None, chapter_id, 0, "hola", 100, 500, 0.9)],
            )
            session.repository.complete_recognition_run(
                run.id or 0,
                InferenceDeviceInfo("faster-whisper", "small", actual_device="cuda", compute_type="float16"),
            )
            restored = session.repository.recognition_run("key")
            self.assertEqual("complete", restored.status)
            self.assertEqual("cuda", restored.actual_device)
            self.assertEqual("hola", session.repository.recognition_tokens(run.id or 0)[0].text)
            session.close()

    def test_alignment_cache_requires_matching_text_and_silence_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("alignment-cache", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
            run = session.repository.ensure_recognition_run(
                chapter_id=chapter_id, cache_key="alignment-key", backend="faster-whisper",
                model="small", language="es", audio_signature="audio", parameters_json="{}",
            )
            session.repository.record_alignment_run(
                chapter_id, run.id or 0, "text-a", "algorithm-v2", "silence-a",
            )
            self.assertTrue(session.repository.alignment_is_current(
                chapter_id, run.id or 0, "text-a", "algorithm-v2", "silence-a",
            ))
            self.assertFalse(session.repository.alignment_is_current(
                chapter_id, run.id or 0, "text-b", "algorithm-v2", "silence-a",
            ))
            self.assertFalse(session.repository.alignment_is_current(
                chapter_id, run.id or 0, "text-a", "algorithm-v2", "silence-b",
            ))
            session.close()

    def test_silence_candidates_round_trip_by_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("静音缓存", Path(temporary) / "project")
            chapter_id = session.repository.add_chapter(Chapter(None, "章节", 0))
            expected = [BoundaryCandidate(800, 0.9, start_ms=650, end_ms=950)]
            session.repository.replace_silence_candidates(chapter_id, expected, "signature-a")
            loaded = session.repository.silence_candidates(chapter_id, "signature-a")
            self.assertEqual((800, 650, 950), (loaded[0].time_ms, loaded[0].start_ms, loaded[0].end_ms))
            self.assertEqual([], session.repository.silence_candidates(chapter_id, "signature-b"))
            session.close()

    def test_normal_open_accepts_only_schema_v2(self) -> None:
        current = migrate_manifest({"schema_version": 2, "title": "当前项目", "project_id": "x"})
        self.assertEqual((2, "当前项目"), (current["schema_version"], current["title"]))
        with self.assertRaises(UnsupportedProjectError):
            migrate_manifest({"schema_version": 1, "title": "旧项目"})
        with self.assertRaises(UnsupportedProjectError):
            migrate_manifest({"schema_version": 999})

    def test_folder_archive_round_trip_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"LOCALAPPDATA": temporary}):
            base = Path(temporary)
            session = ProjectSession.create("测试项目", base / "source-project", internal=False)
            chapter_id = session.repository.add_chapter(Chapter(None, "第一章", 0))
            session.repository.replace_segments(
                chapter_id,
                [
                    TextSegment(None, chapter_id, 0, "第一句", 0, 1000, 0.9, SegmentStatus.MANUAL),
                    TextSegment(None, chapter_id, 1, "第二句", 900, 2000, 0.9, SegmentStatus.MANUAL),
                ],
            )
            session.save()
            archive = session.save_as_archive(base / "portable.aatproj", include_cache=False)
            folder = session.save_as_folder(base / "folder-copy", include_cache=False)
            session.close()

            self.assertTrue(archive.exists())
            self.assertTrue((folder / "manifest.json").exists())
            reopened = ProjectSession.open(archive)
            try:
                self.assertEqual("测试项目", reopened.manifest.title)
                self.assertEqual(2, len(reopened.repository.segments(chapter_id)))
                output = base / "exports"
                srt = export_subtitles(reopened, output, "srt")[0]
                self.assertIn("00:00:01,000 --> 00:00:02,000", srt.read_text("utf-8-sig"))
                vtt = export_subtitles(reopened, output, "vtt")[0]
                self.assertTrue(vtt.read_text("utf-8").startswith("WEBVTT"))
                html = export_html(reopened, output / "html")
                self.assertIn("第一句", html.read_text("utf-8"))
                json_path = export_json(reopened, output / "alignment.json")
                payload = json.loads(json_path.read_text("utf-8"))
                self.assertEqual(2, payload["schema_version"])
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()

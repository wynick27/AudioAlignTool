from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audioalign.core.mapping import AudioSlice, automatic_links, normalize_title
from audioalign.core.models import AudioAsset, Chapter, ChapterAudioLink
from audioalign.core.paths import ApplicationPaths, sanitize_project_name
from audioalign.core.storage import ProjectSession


class MappingPathTests(unittest.TestCase):
    def test_project_name_is_the_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("哈利波特", root / "哈利波特")
            try:
                self.assertEqual("哈利波特", session.root.name)
                self.assertEqual("哈利波特", session.manifest.project_id)
            finally:
                session.close()

    def test_windows_name_sanitizing(self) -> None:
        self.assertEqual("书_第一部", sanitize_project_name("书:第一部. "))
        self.assertNotEqual("CON", sanitize_project_name("CON"))

    def test_monotonic_title_matching_and_non_body_skip(self) -> None:
        chapters = [
            Chapter(1, "Copyright", 0),
            Chapter(2, "CAPÍTULO 1", 1),
            Chapter(3, "CAPÍTULO 2", 2),
        ]
        choices = [
            AudioSlice(1, "01 El niño que sobrevivió", "01.opus", 0, 10_000, 0),
            AudioSlice(2, "02 El vidrio", "02.opus", 0, 12_000, 1),
        ]
        links = automatic_links(chapters, choices)
        self.assertEqual([2, 3], [link.chapter_id for link in links])
        self.assertEqual([1, 2], [link.audio_id for link in links])
        self.assertEqual("capitulo1", normalize_title("CAPÍTULO 1"))

    def test_short_front_matter_does_not_consume_first_audiobook_chapter(self) -> None:
        chapters = [
            Chapter(1, "章节 1", 0),
            Chapter(2, "Title page", 1),
            Chapter(3, "CAPÍTULO 1 El niño", 2),
            Chapter(4, "CAPÍTULO 2 El vidrio", 3),
        ]
        choices = [
            AudioSlice(10, "01 El niño", "01.opus", 0, 20_000, 0),
            AudioSlice(11, "02 El vidrio", "02.opus", 0, 20_000, 1),
        ]
        links = automatic_links(chapters, choices, {1: 5, 2: 100, 3: 30_000, 4: 25_000})
        self.assertEqual([(3, 10), (4, 11)], [(link.chapter_id, link.audio_id) for link in links])

    def test_one_chapter_can_have_multiple_ordered_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("多切片", Path(temporary) / "project")
            try:
                chapter_id = session.repository.add_chapter(Chapter(None, "第一章", 0))
                first = session.repository.add_audio(AudioAsset(None, "a.m4b", duration_ms=30_000))
                second = session.repository.add_audio(AudioAsset(None, "b.opus", duration_ms=20_000))
                session.repository.replace_all_chapter_links(
                    [
                        ChapterAudioLink(None, chapter_id, first, 0, 5_000, 15_000, 1.0),
                        ChapterAudioLink(None, chapter_id, second, 1, 0, 20_000, 1.0),
                    ]
                )
                links = session.repository.chapter_links(chapter_id)
                self.assertEqual([0, 1], [link.position for link in links])
                self.assertEqual([(5_000, 15_000), (0, 20_000)], [(link.source_start_ms, link.source_end_ms) for link in links])
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()

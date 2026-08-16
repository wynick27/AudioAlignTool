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

    def test_soft_hyphen_chapter_word_matches_numeric_audio(self) -> None:
        chapter = Chapter(1, "�� CHAP\u00adTER THIRTEEN �� — Nicolas Flamel", 0)
        choice = AudioSlice(9, "13 Nicolas Flamel", "13.mp3", 0, 10_000, 0)
        links = automatic_links([chapter], [choice], {1: 5000})
        self.assertEqual([(1, 9)], [(link.chapter_id, link.audio_id) for link in links])
        self.assertGreater(links[0].confidence, 0.7)

    def test_japanese_numbered_chapters_do_not_skip_later_audio(self) -> None:
        chapters = [Chapter(index, f"第{index}章　題名{index}", index - 1) for index in range(1, 18)]
        choices = [
            AudioSlice(index, f"{index:02d}. 題名{index}", f"{index:02d}.opus", 0, 60_000, index - 1)
            for index in range(1, 18)
        ]
        links = automatic_links(chapters, choices, {index: 10_000 for index in range(1, 18)})
        self.assertEqual(list(range(1, 18)), [link.chapter_id for link in links])
        self.assertEqual(list(range(1, 18)), [link.audio_id for link in links])

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

    def test_generic_m4b_markers_use_global_duration_shape_not_marker_number(self) -> None:
        names = (
            "One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve "
            "Thirteen Fourteen Fifteen Sixteen"
        ).split()
        lengths = [
            17914, 22763, 18871, 11865, 14653, 15043, 12473, 16040,
            27757, 12381, 15172, 12898, 14157, 17332, 15443, 7408,
        ]
        body_durations = [
            1199191, 1599674, 1435625, 924186, 1058678, 1078192,
            962260, 1182903, 2129176, 963984, 1097796, 1065104,
            1053010, 1225914, 1104770, 636853,
        ]
        chapters = [Chapter(1, "And Then There Were None", 0)]
        chapters.extend(Chapter(index + 2, name, index + 1) for index, name in enumerate(names))
        chapters.extend([
            Chapter(18, "Epilogue", 17),
            Chapter(19, "A Manuscript Document Sent To Scotland Yard", 18),
            Chapter(20, "And Then There Were None", 19),
        ])
        durations = [42_214, 77_676, *body_durations, 1_013_539, 1_864_451]
        cursor = 0
        choices = []
        for index, duration in enumerate(durations):
            choices.append(AudioSlice(1, f"Chapter {index + 1:03d}", "book.m4b", cursor, cursor + duration, index))
            cursor += duration
        text_lengths = {
            1: 937, **{index + 2: length for index, length in enumerate(lengths)},
            18: 15370, 19: 20292, 20: 881,
        }

        links = automatic_links(chapters, choices, text_lengths)

        by_chapter = {link.chapter_id: link for link in links}
        self.assertEqual(42_214, by_chapter[1].source_start_ms)
        for offset, chapter_id in enumerate(range(2, 18), start=2):
            self.assertEqual(choices[offset].start_ms, by_chapter[chapter_id].source_start_ms)
        self.assertEqual(choices[18].start_ms, by_chapter[18].source_start_ms)
        self.assertEqual(choices[19].start_ms, by_chapter[19].source_start_ms)
        self.assertNotIn(20, by_chapter)

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

    def test_staged_media_library_commits_assets_and_links_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("media-transaction", Path(temporary) / "project")
            try:
                chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
                staged = AudioAsset(-1, str(Path(temporary) / "video.mkv"), fingerprint="fingerprint", duration_ms=5000)
                id_map = session.repository.replace_media_library(
                    [staged], {-1: []},
                    [ChapterAudioLink(None, chapter_id, -1, 0, 1000, 4000, 1.0)],
                )
                self.assertGreater(id_map[-1], 0)
                self.assertEqual(id_map[-1], session.repository.chapter_links(chapter_id)[0].audio_id)
                duplicate_id = session.repository.add_audio(
                    AudioAsset(None, str(Path(temporary) / "video.mkv"), fingerprint="another")
                )
                self.assertEqual(id_map[-1], duplicate_id)
                self.assertEqual(1, len(session.repository.all_audio()))
                with self.assertRaises(ValueError):
                    session.repository.replace_media_library(
                        [], {}, [ChapterAudioLink(None, chapter_id, id_map[-1], 0, 0, 1000, 1.0)],
                    )
                self.assertEqual(1, len(session.repository.all_audio()))
            finally:
                session.close()

    def test_changed_media_content_marks_existing_timeline_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("media-review", Path(temporary) / "project")
            try:
                chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
                staged = AudioAsset(-1, str(Path(temporary) / "first.mp3"), fingerprint="first", duration_ms=5000)
                media_id = session.repository.replace_media_library(
                    [staged], {-1: []},
                    [ChapterAudioLink(None, chapter_id, -1, 0, 0, 5000, 1.0)],
                )[-1]
                self.assertFalse(session.repository.chapter_media_needs_review(chapter_id))

                replacement = AudioAsset(
                    media_id, str(Path(temporary) / "replacement.mp3"),
                    fingerprint="different-content", duration_ms=5000,
                )
                session.repository.replace_media_library(
                    [replacement], {media_id: []},
                    [ChapterAudioLink(None, chapter_id, media_id, 0, 0, 5000, 1.0)],
                )
                self.assertTrue(session.repository.chapter_media_needs_review(chapter_id))

                session.repository.replace_recognition_alignment(chapter_id, [], [], [])
                self.assertFalse(session.repository.chapter_media_needs_review(chapter_id))
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()

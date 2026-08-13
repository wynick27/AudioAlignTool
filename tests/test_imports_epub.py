from __future__ import annotations

import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from bs4 import BeautifulSoup

from audioalign.core.epub_media_overlay import (
    EpubMediaOverlayOptions,
    _annotate_for_overlay,
    _export_segment_ranges,
    export_epub_media_overlay,
)
from audioalign.core.models import (
    AudioAsset, Chapter, ChapterAudioLink, SourceDocument, SourceDocumentKind,
    TextAudioAnchor, TextSegment,
)
from audioalign.core.storage import ProjectSession
from audioalign.core.subtitles import parse_srt_text
from audioalign.core.text import collect_local_html_resources, import_book
from audioalign.core.timecode import format_time_ms, parse_srt_time
from audioalign.gui.original_book_view import annotate_html


class ImportAndEpubTests(unittest.TestCase):
    def test_epub_end_extension_only_fills_safe_adjacent_short_gaps(self) -> None:
        segments = [
            TextSegment(None, 1, 0, "one", 100, 300),
            TextSegment(None, 1, 1, "two", 500, 700),
            TextSegment(None, 1, 2, "unmatched", 0, 0),
            TextSegment(None, 1, 3, "three", 900, 1_100),
            TextSegment(None, 1, 4, "overlap", 1_050, 1_200),
            TextSegment(None, 1, 5, "long gap", 3_000, 3_200),
        ]
        options = EpubMediaOverlayOptions(
            extend_segment_ends=True,
            max_end_extension_ms=500,
        )
        ranges = _export_segment_ranges(segments, options)
        self.assertEqual((100, 500), ranges[0])
        self.assertEqual((500, 700), ranges[1])
        self.assertNotIn(2, ranges)
        self.assertEqual((900, 1_100), ranges[3])
        self.assertEqual((1_050, 1_200), ranges[4])

    def test_epub_mapping_tolerates_soft_hyphen_and_closing_quote_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("projection", Path(temporary) / "project")
            try:
                chapter_id = session.repository.add_chapter(Chapter(None, "Chapter One", 0))
                session.repository.replace_segments(chapter_id, [
                    TextSegment(None, chapter_id, 0, "— CHAPTER ONE —", 100, 300),
                    TextSegment(
                        None, chapter_id, 1,
                        "She learnt a new word (‘Shan’t!).", 300, 900,
                    ),
                ])
                audio_id = session.repository.add_audio(AudioAsset(
                    None, str(Path(temporary) / "audio.opus"), duration_ms=1_000,
                    sample_rate=48_000, channels=1, format="opus",
                ))
                session.repository.set_chapter_links(chapter_id, [
                    ChapterAudioLink(None, chapter_id, audio_id, 0, 0, 1_000, 1.0),
                ])
                chapter = next(item for item in session.repository.chapters() if item.id == chapter_id)
                matched: set[int] = set()
                rendered, units, errors = _annotate_for_overlay(
                    "<html><body><h1>— CHAP\u00adTER ONE —</h1>"
                    "<p>She learnt a new word (‘Shan’t!’).</p></body></html>",
                    session, chapter, matched_positions=matched,
                )
                self.assertEqual([], errors)
                self.assertEqual({0, 1}, matched)
                self.assertEqual(2, len(units))
                self.assertIn("CHAP\u00adTER ONE", rendered)
                self.assertIn("Shan’t!’", BeautifulSoup(rendered, "html.parser").get_text())
            finally:
                session.close()

    def test_timecode_and_srt_over_one_hour(self) -> None:
        self.assertEqual("01:02:03.456", format_time_ms(3_723_456))
        self.assertEqual(3_723_456, parse_srt_time("01:02:03,456"))
        cues = parse_srt_text("1\n01:02:03,456 --> 01:02:05,000\nHello\nworld\n")
        self.assertEqual((3_723_456, "Hello\nworld"), (cues[0].start_ms, cues[0].text))

    def test_markdown_and_html_split_at_level_one_headings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            markdown = root / "book.md"
            markdown.write_text("# One\nFirst *chapter*.\n# Two\nSecond chapter.", encoding="utf-8")
            chapters = import_book(markdown)
            self.assertEqual(["One", "Two"], [chapter.title for chapter in chapters])
            self.assertIn("<em>chapter</em>", chapters[0].source_html)

            (root / "style.css").write_text("body{color:red}", encoding="utf-8")
            html = root / "book.html"
            html.write_text(
                "<html><head><link rel='stylesheet' href='style.css'></head><body>"
                "<h1>One</h1><p>First.</p><h1>Two</h1><p>Second.</p></body></html>",
                encoding="utf-8",
            )
            chapters = import_book(html)
            self.assertEqual(["One", "Two"], [chapter.title for chapter in chapters])
            resources, warnings = collect_local_html_resources(html)
            self.assertEqual([root / "style.css"], resources)
            self.assertEqual([], warnings)

    def test_multiple_document_mode_keeps_each_markdown_or_html_as_one_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            markdown = root / "part-one.md"
            markdown.write_text("# Internal one\nFirst.\n# Internal two\nSecond.", encoding="utf-8")
            markdown_chapters = import_book(markdown, one_chapter=True)
            self.assertEqual(["part-one"], [chapter.title for chapter in markdown_chapters])
            self.assertIn("First", markdown_chapters[0].text)
            self.assertIn("Second", markdown_chapters[0].text)

            html = root / "part-two.html"
            html.write_text(
                "<html><head><title>Part two</title></head><body>"
                "<h1>Internal one</h1><p>First.</p><h1>Internal two</h1><p>Second.</p>"
                "</body></html>", encoding="utf-8",
            )
            html_chapters = import_book(html, one_chapter=True)
            self.assertEqual(["Part two"], [chapter.title for chapter in html_chapters])
            self.assertNotIn("Part two", html_chapters[0].text)
            self.assertIn("Internal two", html_chapters[0].text)

    def test_chapter_source_parts_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = ProjectSession.create("parts", root / "project")
            try:
                document_id = session.repository.add_source_document(SourceDocument(
                    None, SourceDocumentKind.EPUB, "book.epub", "source/book.epub",
                ))
                chapter_id = session.repository.add_chapter(Chapter(None, "Chapter", 0))
                session.repository.set_chapter_source_document(
                    chapter_id, document_id, entry_path="text/label.xhtml",
                )
                session.repository.set_chapter_source_parts(
                    chapter_id, document_id,
                    [("text/label.xhtml", "spine:0"), ("text/body.xhtml", "spine:1")],
                )
                parts = session.repository.chapter_source_parts(chapter_id)
                self.assertEqual(
                    ["text/label.xhtml", "text/body.xhtml"],
                    [entry_path for _document, entry_path, _selector in parts],
                )
            finally:
                session.close()

    def test_chapter_state_restore_rebinds_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = ProjectSession.create("undo", Path(temporary) / "project")
            try:
                chapter_id = session.repository.add_chapter(Chapter(None, "chapter", 0))
                session.repository.replace_segments(
                    chapter_id, [TextSegment(None, chapter_id, 0, "hello", 100, 900)],
                )
                original = session.repository.segments(chapter_id)
                session.repository.replace_anchors(chapter_id, [TextAudioAnchor(
                    None, chapter_id, original[0].id, 0, 5, 100, 900, 1.0, "test",
                )])
                anchors = session.repository.anchors(chapter_id)
                changed = [TextSegment(None, chapter_id, 0, "changed", 0, 0)]
                session.repository.replace_chapter_edit_state(chapter_id, changed, [])
                session.repository.replace_chapter_edit_state(chapter_id, original, anchors)
                restored = session.repository.segments(chapter_id)
                restored_anchor = session.repository.anchors(chapter_id)[0]
                self.assertEqual("hello", restored[0].text)
                self.assertEqual(restored[0].id, restored_anchor.segment_id)
            finally:
                session.close()

    def test_original_book_annotation_preserves_markup_and_styles(self) -> None:
        source = (
            "<html><head><style>.lead{color:red}</style></head><body>"
            "<p class='lead'>Hello <em>world</em>.</p></body></html>"
        )
        rendered = annotate_html(
            source, [TextSegment(42, 1, 0, "Hello world.", 0, 1000)], "aatbook://book/source/",
        )
        self.assertIn(".lead{color:red}", rendered)
        self.assertIn('class="lead"', rendered)
        self.assertIn("<em>", rendered)
        self.assertGreaterEqual(rendered.count('data-aat-segment="42"'), 2)

    def test_generated_epub_contains_media_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            rate = 16_000
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(np.zeros(rate, dtype=np.int16).tobytes())
            session = ProjectSession.create("audio-book", root / "project")
            try:
                chapter_id = session.repository.add_chapter(Chapter(None, "Chapter", 0))
                session.repository.replace_segments(
                    chapter_id, [
                        TextSegment(None, chapter_id, 0, "Hello world.", 0, 300),
                        TextSegment(None, chapter_id, 1, "Second sentence.", 500, 900),
                    ],
                )
                audio_id = session.repository.add_audio(AudioAsset(
                    None, str(audio), duration_ms=1000, sample_rate=rate, channels=1, format="wav",
                ))
                session.repository.set_chapter_links(
                    chapter_id, [ChapterAudioLink(None, chapter_id, audio_id, 0, 0, 1000, 1.0)],
                )
                options = EpubMediaOverlayOptions(
                    extend_segment_ends=True,
                    max_end_extension_ms=300,
                )
                output = export_epub_media_overlay(session, root / "book.epub", options)
                with zipfile.ZipFile(output) as archive:
                    names = archive.namelist()
                    self.assertEqual("mimetype", names[0])
                    self.assertTrue(any(name.endswith(".smil") for name in names))
                    self.assertTrue(any(name.endswith(".m4a") for name in names))
                    audio_name = next(name for name in names if name.endswith(".m4a"))
                    self.assertEqual(zipfile.ZIP_STORED, archive.getinfo(audio_name).compress_type)
                    opf = archive.read("OEBPS/content.opf").decode("utf-8")
                    self.assertIn("media-overlay", opf)
                    smil_name = next(name for name in names if name.endswith(".smil"))
                    smil = archive.read(smil_name).decode("utf-8")
                    self.assertIn("clipEnd='00:00:00.500'", smil)
                with patch(
                    "audioalign.core.epub_media_overlay._transcode",
                    side_effect=AssertionError("cached EPUB audio must not be transcoded again"),
                ):
                    second = export_epub_media_overlay(session, root / "book-again.epub")
                self.assertTrue(second.is_file())
                self.assertTrue(any((session.root / "cache" / "epub-media").glob("*.m4a")))
            finally:
                session.close()

    def test_epub_export_skips_one_unmapped_segment_instead_of_failing_book(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_epub = root / "source.epub"
            with zipfile.ZipFile(source_epub, "w") as archive:
                archive.writestr("mimetype", "application/epub+zip")
                archive.writestr(
                    "META-INF/container.xml",
                    '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
                )
                archive.writestr(
                    "OEBPS/content.opf",
                    '<package><manifest><item id="c1" href="c1.xhtml" '
                    'media-type="application/xhtml+xml"/></manifest><spine>'
                    '<itemref idref="c1"/></spine></package>',
                )
                archive.writestr("OEBPS/c1.xhtml", "<html><body><p>Visible body.</p></body></html>")
            audio = root / "sample.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000)
                handle.writeframes(np.zeros(16_000, dtype=np.int16).tobytes())
            session = ProjectSession.create("partial", root / "project")
            try:
                stored = session.root / "source" / source_epub.name
                stored.parent.mkdir(exist_ok=True)
                stored.write_bytes(source_epub.read_bytes())
                session.manifest.source_name = str(stored.relative_to(session.root))
                chapter_id = session.repository.add_chapter(Chapter(
                    None, "Chapter", 0, "<html><body><p>Visible body.</p></body></html>",
                ))
                session.repository.replace_segments(chapter_id, [
                    TextSegment(None, chapter_id, 0, "This sentence is absent.", 0, 900),
                ])
                audio_id = session.repository.add_audio(AudioAsset(
                    None, str(audio), duration_ms=1000, sample_rate=16_000, channels=1, format="wav",
                ))
                session.repository.set_chapter_links(
                    chapter_id, [ChapterAudioLink(None, chapter_id, audio_id, 0, 0, 1000, 1.0)],
                )
                options = EpubMediaOverlayOptions()
                output = export_epub_media_overlay(session, root / "partial.epub", options)
                self.assertTrue(output.is_file())
                self.assertEqual(1, len(options.warnings))
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()

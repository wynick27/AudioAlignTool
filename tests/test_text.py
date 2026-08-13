from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from audioalign.core.text import (
    cursor_split_offset,
    import_epub,
    normalize_for_match,
    preferred_split_offset,
    split_sentences,
)


class TextTests(unittest.TestCase):
    def test_english_ellipsis_is_a_sentence_boundary(self) -> None:
        text = (
            "nor that he would spend the next few weeks being prodded and pinched "
            "by his cousin Dudley … He could not know that at this very moment,，另外"
        )
        result = split_sentences(text)
        self.assertEqual(2, len(result))
        self.assertTrue(result[0].endswith("Dudley …"))
        self.assertTrue(result[1].startswith("He could not know"))

    def test_new_direct_speech_after_narration_starts_a_new_segment(self) -> None:
        text = (
            "Quir\u00adrell laughed and it wasn’t his usual quiv\u00ader\u00ading treble, "
            "either, but cold and sharp. ‘Yes, Severus does seem the type, doesn’t he?"
        )
        result = split_sentences(text)
        self.assertEqual(2, len(result))
        self.assertTrue(result[0].endswith("cold and sharp."))
        self.assertTrue(result[1].startswith("‘Yes, Severus"))
        self.assertEqual(
            ["Narration ends.", "‘New speaker?’"],
            split_sentences("Narration ends.‘New speaker?’"),
        )

    def test_cursor_split_keeps_following_punctuation_on_the_left(self) -> None:
        text = "First clause, second clause"
        self.assertEqual(len("First clause,"), cursor_split_offset(text, len("First clause")))

    def test_multilingual_sentence_split(self) -> None:
        text = "第一句。第二句！\n\nこれは三文目です。\n\nHola mundo. Esta es una prueba."
        result = split_sentences(text)
        self.assertEqual(5, len(result))
        self.assertEqual("第一句。", result[0])
        self.assertEqual("Esta es una prueba.", result[-1])

    def test_normalization_does_not_keep_punctuation(self) -> None:
        self.assertEqual("ａｂｃ".encode("utf-8") is not None, True)
        self.assertEqual("abc你好", normalize_for_match("ＡＢＣ， 你好！"))

    def test_long_sentence_uses_multiple_comma_boundaries_without_punctuation_rows(self) -> None:
        clause = (
            "这是第一部分，包含一些说明，这是第二部分，也有更多说明，"
            "这是第三部分，仍然继续补充，这是第四部分，还要继续补充，"
            "这是第五部分，仍有一些内容，这是第六部分，最后再作说明，"
        )
        text = clause * 2 + "这是最后一部分，最终结束。"
        result = split_sentences(text)
        self.assertGreaterEqual(len(result), 3)
        self.assertTrue(all(any(character.isalnum() for character in item) for item in result))
        self.assertEqual("，", text[preferred_split_offset(text, 42) - 1])

    def test_epub_standard_library_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            epub = Path(temporary) / "book.epub"
            with zipfile.ZipFile(epub, "w") as archive:
                archive.writestr(
                    "META-INF/container.xml",
                    '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
                )
                archive.writestr(
                    "OEBPS/content.opf",
                    '<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="c1" href="c1.xhtml"/></manifest><spine><itemref idref="c1"/></spine></package>',
                )
                archive.writestr("OEBPS/c1.xhtml", "<h1>第一章</h1><p>正文内容。</p>")
            chapters = import_epub(epub)
            self.assertEqual("第一章", chapters[0].title)
            self.assertIn("正文内容", chapters[0].text)

    def test_epub_merges_duplicate_title_page_without_losing_spoken_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            epub = Path(temporary) / "japanese.epub"
            with zipfile.ZipFile(epub, "w") as archive:
                archive.writestr(
                    "META-INF/container.xml",
                    '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
                )
                archive.writestr(
                    "OEBPS/content.opf",
                    '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
                    '<item id="cover" href="cover.xhtml"/><item id="title" href="title.xhtml"/>'
                    '<item id="body" href="body.xhtml"/></manifest><spine>'
                    '<itemref idref="cover"/><itemref idref="title"/><itemref idref="body"/>'
                    '</spine></package>',
                )
                archive.writestr("OEBPS/cover.xhtml", "<p>Cover</p>")
                archive.writestr("OEBPS/title.xhtml", '<p class="chapter-title">第１章　生き残った男の子</p>')
                archive.writestr(
                    "OEBPS/body.xhtml",
                    "<p>第１章　生き残った男の子</p><p>これは十分に長い本文です。" * 30 + "</p>",
                )
            chapters = import_epub(epub)
            self.assertEqual(1, len(chapters))
            self.assertEqual("第１章　生き残った男の子", chapters[0].title)
            self.assertIn("第１章　生き残った男の子", chapters[0].text)
            self.assertIn("これは十分に長い本文", chapters[0].text)

    def test_epub_merges_spoken_chapter_label_and_excludes_head_title(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            epub = Path(temporary) / "english.epub"
            with zipfile.ZipFile(epub, "w") as archive:
                archive.writestr(
                    "META-INF/container.xml",
                    '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
                )
                archive.writestr(
                    "OEBPS/content.opf",
                    '<package><manifest><item id="label" href="label.xhtml"/>'
                    '<item id="body" href="body.xhtml"/></manifest><spine>'
                    '<itemref idref="label"/><itemref idref="body"/></spine></package>',
                )
                archive.writestr(
                    "OEBPS/label.xhtml",
                    "<html><head><title>Book - Chapter 13</title></head>"
                    "<body><h1>CHAPTER THIRTEEN</h1></body></html>",
                )
                archive.writestr(
                    "OEBPS/body.xhtml",
                    "<html><head><title>Book - Chapter 13</title></head><body>"
                    "<h1>Nicolas Flamel</h1><p>" + "Long visible body text. " * 30
                    + "</p></body></html>",
                )
            chapters = import_epub(epub)
            self.assertEqual(1, len(chapters))
            self.assertEqual(2, len(chapters[0].source_parts))
            self.assertIn("CHAPTER THIRTEEN", chapters[0].text)
            self.assertIn("Nicolas Flamel", chapters[0].text)
            self.assertNotIn("Book - Chapter 13", chapters[0].text)


if __name__ == "__main__":
    unittest.main()

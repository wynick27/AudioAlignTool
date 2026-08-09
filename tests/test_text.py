from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from audioalign.core.text import import_epub, normalize_for_match, split_sentences


class TextTests(unittest.TestCase):
    def test_multilingual_sentence_split(self) -> None:
        text = "第一句。第二句！\n\nこれは三文目です。\n\nHola mundo. Esta es una prueba."
        result = split_sentences(text)
        self.assertEqual(5, len(result))
        self.assertEqual("第一句。", result[0])
        self.assertEqual("Esta es una prueba.", result[-1])

    def test_normalization_does_not_keep_punctuation(self) -> None:
        self.assertEqual("ａｂｃ".encode("utf-8") is not None, True)
        self.assertEqual("abc你好", normalize_for_match("ＡＢＣ， 你好！"))

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

    def test_epub_merges_styled_title_page_with_following_body(self) -> None:
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
            self.assertIn("これは十分に長い本文", chapters[0].text)


if __name__ == "__main__":
    unittest.main()

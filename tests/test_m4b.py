from __future__ import annotations

import unittest
from fractions import Fraction
from unittest.mock import patch

from audioalign.core.audio import probe_audio


class _Codec:
    sample_rate = 44100
    channels = 2


class _Stream:
    codec_context = _Codec()
    duration = 120_000
    time_base = Fraction(1, 1000)


class _Chapter:
    time_base = Fraction(1, 1000)

    def __init__(self, title: str, start: int, end: int) -> None:
        self.metadata = {"title": title}
        self.start = start
        self.end = end


class _Container:
    streams = type("Streams", (), {"audio": [_Stream()]})()
    duration = 120_000_000
    metadata = {"title": "Audiobook"}
    format = type("Format", (), {"name": "mov,mp4,m4a"})()

    def chapters(self):
        return [_Chapter("Part 1", 0, 60_000), _Chapter("Part 2", 60_000, 120_000)]

    def close(self):
        pass


class M4BTests(unittest.TestCase):
    def test_probe_keeps_embedded_chapters_as_slices(self) -> None:
        with patch("av.open", return_value=_Container()):
            result = probe_audio("book.m4b")
        self.assertEqual("m4b", result.format)
        self.assertEqual(120_000, result.duration_ms)
        self.assertEqual(
            [("Part 1", 0, 60_000), ("Part 2", 60_000, 120_000)],
            result.chapters,
        )

    def test_probe_supports_pyav_18_mapping_chapters(self) -> None:
        container = _Container()
        container.chapters = lambda: [
            {
                "id": 0,
                "start": 0,
                "end": 42_214,
                "time_base": Fraction(1, 1000),
                "metadata": {"title": "Chapter 001"},
            },
            {
                "id": 1,
                "start": 42_214,
                "end": 120_000,
                "time_base": Fraction(1, 1000),
                "metadata": {},
            },
        ]
        with patch("av.open", return_value=container):
            result = probe_audio("book.m4b")
        self.assertEqual(
            [("Chapter 001", 0, 42_214), ("章节 2", 42_214, 120_000)],
            result.chapters,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
import tempfile
import wave
from pathlib import Path

import numpy as np

from audioalign.core.audio import audio_metadata, detect_silence_candidates
from audioalign.core.models import SilenceSettings
from audioalign.core.spectrogram import AudioVisualizationCache, SpectrogramCache, build_spectrogram_cache


class AudioTests(unittest.TestCase):
    def test_energy_silence_detection(self) -> None:
        rate = 16000
        voice = np.sin(np.linspace(0, 100, rate)).astype(np.float32) * 0.2
        silence = np.zeros(rate, dtype=np.float32)
        samples = np.concatenate([voice, silence, voice])
        candidates = detect_silence_candidates(
            samples, rate, SilenceSettings(min_silence_ms=350, energy_percentile=20)
        )
        self.assertTrue(any(900 <= item.time_ms <= 2100 for item in candidates))

    def test_spectrogram_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            rate = 16000
            values = (np.sin(np.linspace(0, 200, rate)) * 20000).astype(np.int16)
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(values.tobytes())
            duration, sample_rate, channels = audio_metadata(audio)
            self.assertGreaterEqual(duration, 990)
            self.assertEqual((rate, 1), (sample_rate, channels))
            cache = SpectrogramCache(build_spectrogram_cache(audio, root / "spectrum"))
            visible, start, end = cache.slice(100, 800, 320)
            self.assertEqual(192, visible.shape[0])
            self.assertGreater(visible.shape[1], 1)
            self.assertEqual((100, 800), (start, end))
            self.assertGreaterEqual(cache.metadata.duration_ms, 990)
            minimum, maximum, wave_start, wave_end = cache.waveform_slice(100, 800, 320)
            self.assertEqual((100, 800), (wave_start, wave_end))
            self.assertEqual(minimum.shape, maximum.shape)
            empty_spectrum, empty_start, empty_end = cache.spectrogram_slice(5_000, 6_000, 100)
            empty_minimum, empty_maximum, _, _ = cache.waveform_slice(5_000, 6_000, 100)
            self.assertEqual(0, empty_spectrum.shape[1])
            self.assertEqual(empty_start, empty_end)
            self.assertEqual(0, empty_minimum.size)
            self.assertEqual(0, empty_maximum.size)
            self.assertGreater(minimum.size, 1)
            self.assertTrue(np.all(minimum <= maximum))
            cache.close()



if __name__ == "__main__":
    unittest.main()

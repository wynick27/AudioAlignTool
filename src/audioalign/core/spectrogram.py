from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import decode_audio_mono


VISUALIZATION_CACHE_VERSION = 2


@dataclass(slots=True)
class AudioVisualizationMetadata:
    visualization_version: int
    sample_rate: int
    duration_ms: int
    window_size: int
    hop_size: int
    frequency_bins: int
    frame_count: int
    waveform_bucket_samples: int
    waveform_bucket_count: int
    min_frequency: int = 50
    max_frequency: int = 8000
    spectrogram_levels: int = 1
    waveform_levels: int = 1


# Kept as an import alias for callers that only inspect spectrogram metadata.
SpectrogramMetadata = AudioVisualizationMetadata


def _spectrum_path(root: Path, level: int) -> Path:
    return root / f"spectrogram-level-{level}.npy"


def _waveform_path(root: Path, level: int) -> Path:
    return root / f"waveform-level-{level}.npy"


def is_visualization_cache(root: str | Path) -> bool:
    metadata_path = Path(root) / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        data = json.loads(metadata_path.read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return (
        data.get("visualization_version") == VISUALIZATION_CACHE_VERSION
        and _spectrum_path(Path(root), 0).exists()
        and _waveform_path(Path(root), 0).exists()
    )


def build_audio_visualization_cache(
    audio_path: str | Path,
    cache_dir: str | Path,
    *,
    start_ms: int = 0,
    end_ms: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> Path:
    samples, rate = decode_audio_mono(audio_path, 16000, start_ms=start_ms, end_ms=end_ms)
    return _build_visualization_samples(samples, rate, Path(cache_dir), progress)


def build_audio_visualization_cache_from_slices(
    slices: list[tuple[str | Path, int, int]],
    cache_dir: str | Path,
    *,
    progress: Callable[[float], None] | None = None,
) -> Path:
    chunks: list[np.ndarray] = []
    total = max(1, len(slices))
    for index, (path, start_ms, end_ms) in enumerate(slices):
        chunks.append(decode_audio_mono(path, 16000, start_ms=start_ms, end_ms=end_ms)[0])
        if progress:
            progress(0.10 * (index + 1) / total)
    samples = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
    return _build_visualization_samples(samples, 16000, Path(cache_dir), progress, progress_base=0.10)


def _build_visualization_samples(
    samples: np.ndarray,
    rate: int,
    root: Path,
    progress: Callable[[float], None] | None,
    *,
    progress_base: float = 0.0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype=np.float32)
    if progress:
        progress(max(progress_base, 0.11))

    # A one millisecond base bucket keeps boundary editing precise while being
    # much smaller than retaining the decoded PCM for long audiobook chapters.
    bucket_samples = max(1, rate // 1000)
    bucket_count = max(1, math.ceil(samples.size / bucket_samples))
    padded = np.pad(samples, (0, bucket_count * bucket_samples - samples.size))
    buckets = padded.reshape(bucket_count, bucket_samples)
    wave0 = np.lib.format.open_memmap(
        _waveform_path(root, 0), mode="w+", dtype=np.int16, shape=(2, bucket_count)
    )
    scale = 32767.0
    wave0[0] = np.clip(buckets.min(axis=1) * scale, -32767, 32767).astype(np.int16)
    wave0[1] = np.clip(buckets.max(axis=1) * scale, -32767, 32767).astype(np.int16)
    wave0.flush()
    if progress:
        progress(max(progress_base, 0.20))

    waveform_levels = 1
    previous_wave = wave0
    while previous_wave.shape[1] > 2048:
        columns = math.ceil(previous_wave.shape[1] / 2)
        target = np.lib.format.open_memmap(
            _waveform_path(root, waveform_levels), mode="w+", dtype=np.int16, shape=(2, columns)
        )
        usable = previous_wave[:, : columns * 2]
        if usable.shape[1] % 2:
            usable = np.pad(usable, ((0, 0), (0, 1)), mode="edge")
        paired = usable.reshape(2, -1, 2)
        target[0] = paired[0].min(axis=1)
        target[1] = paired[1].max(axis=1)
        target.flush()
        previous_wave = target
        waveform_levels += 1

    window_size, hop_size, fft_size = 400, 160, 512
    frame_count = max(1, 1 + max(0, samples.size - window_size) // hop_size)
    output_bins = 192
    frequencies = np.fft.rfftfreq(fft_size, 1 / rate)
    target_freq = np.geomspace(50, min(8000, rate // 2), output_bins)
    indices = np.clip(np.searchsorted(frequencies, target_freq), 0, frequencies.size - 1)
    spectrum0 = np.lib.format.open_memmap(
        _spectrum_path(root, 0), mode="w+", dtype=np.uint8, shape=(output_bins, frame_count)
    )
    window = np.hanning(window_size).astype(np.float32)
    batch = 2048
    for begin in range(0, frame_count, batch):
        count = min(batch, frame_count - begin)
        offsets = (begin + np.arange(count))[:, None] * hop_size + np.arange(window_size)[None, :]
        frames = (
            samples[np.minimum(offsets, max(0, samples.size - 1))]
            if samples.size
            else np.zeros((count, window_size), np.float32)
        )
        magnitude = np.abs(np.fft.rfft(frames * window, n=fft_size, axis=1)) / (fft_size / 2)
        db = 20 * np.log10(np.maximum(magnitude[:, indices], 1e-5))
        spectrum0[:, begin : begin + count] = np.clip((db + 80) / 80 * 255, 0, 255).astype(np.uint8).T
        if progress:
            progress(0.20 + 0.68 * (begin + count) / frame_count)
    spectrum0.flush()

    spectrogram_levels = 1
    previous_spectrum = spectrum0
    while previous_spectrum.shape[1] > 2048:
        columns = math.ceil(previous_spectrum.shape[1] / 2)
        target = np.lib.format.open_memmap(
            _spectrum_path(root, spectrogram_levels), mode="w+", dtype=np.uint8,
            shape=(output_bins, columns),
        )
        usable = previous_spectrum[:, : columns * 2]
        if usable.shape[1] % 2:
            usable = np.pad(usable, ((0, 0), (0, 1)), mode="edge")
        target[:] = usable.reshape(output_bins, -1, 2).max(axis=2)
        target.flush()
        previous_spectrum = target
        spectrogram_levels += 1

    metadata = AudioVisualizationMetadata(
        visualization_version=VISUALIZATION_CACHE_VERSION,
        sample_rate=rate,
        duration_ms=int(samples.size / max(1, rate) * 1000),
        window_size=window_size,
        hop_size=hop_size,
        frequency_bins=output_bins,
        frame_count=frame_count,
        waveform_bucket_samples=bucket_samples,
        waveform_bucket_count=bucket_count,
        spectrogram_levels=spectrogram_levels,
        waveform_levels=waveform_levels,
    )
    (root / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    if progress:
        progress(1.0)
    return root


class AudioVisualizationCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        data = json.loads((self.root / "metadata.json").read_text("utf-8"))
        if data.get("visualization_version") != VISUALIZATION_CACHE_VERSION:
            raise ValueError("可视化缓存版本过旧，需要重新生成")
        self.metadata = AudioVisualizationMetadata(**data)
        self._spectrum_levels: dict[int, np.ndarray] = {}
        self._waveform_levels: dict[int, np.ndarray] = {}

    def _level(self, kind: str, number: int) -> np.ndarray:
        count = self.metadata.spectrogram_levels if kind == "spectrogram" else self.metadata.waveform_levels
        number = max(0, min(count - 1, number))
        collection = self._spectrum_levels if kind == "spectrogram" else self._waveform_levels
        if number not in collection:
            path = _spectrum_path(self.root, number) if kind == "spectrogram" else _waveform_path(self.root, number)
            collection[number] = np.load(path, mmap_mode="r")
        return collection[number]

    def level(self, number: int) -> np.ndarray:
        """Compatibility accessor for the spectrogram pyramid."""
        return self._level("spectrogram", number)

    def spectrogram_slice(self, start_ms: int, end_ms: int, pixel_width: int) -> tuple[np.ndarray, int, int]:
        start_ms, end_ms = self._clamp_range(start_ms, end_ms)
        if end_ms <= start_ms or self.metadata.frame_count <= 0:
            return np.empty((self.metadata.frequency_bins, 0), dtype=np.uint8), start_ms, end_ms
        base_start = int(start_ms / max(1, self.metadata.duration_ms) * self.metadata.frame_count)
        base_end = max(base_start + 1, math.ceil(end_ms / max(1, self.metadata.duration_ms) * self.metadata.frame_count))
        level_number = self._choose_level(base_end - base_start, pixel_width, self.metadata.spectrogram_levels)
        data = self._level("spectrogram", level_number)
        divisor = 2**level_number
        start = max(0, base_start // divisor)
        end = min(data.shape[1], max(start + 1, math.ceil(base_end / divisor)))
        return np.asarray(data[:, start:end]), start_ms, end_ms

    def waveform_slice(self, start_ms: int, end_ms: int, pixel_width: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        start_ms, end_ms = self._clamp_range(start_ms, end_ms)
        if end_ms <= start_ms or self.metadata.waveform_bucket_count <= 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty, start_ms, end_ms
        base_start = int(start_ms / max(1, self.metadata.duration_ms) * self.metadata.waveform_bucket_count)
        base_end = max(base_start + 1, math.ceil(end_ms / max(1, self.metadata.duration_ms) * self.metadata.waveform_bucket_count))
        level_number = self._choose_level(base_end - base_start, pixel_width, self.metadata.waveform_levels)
        data = self._level("waveform", level_number)
        divisor = 2**level_number
        start = max(0, base_start // divisor)
        end = min(data.shape[1], max(start + 1, math.ceil(base_end / divisor)))
        values = np.asarray(data[:, start:end], dtype=np.float32) / 32767.0
        return values[0], values[1], start_ms, end_ms

    def slice(self, start_ms: int, end_ms: int, pixel_width: int) -> tuple[np.ndarray, int, int]:
        return self.spectrogram_slice(start_ms, end_ms, pixel_width)

    def _clamp_range(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        duration = max(0, self.metadata.duration_ms)
        requested_start = int(start_ms)
        requested_end = int(end_ms)
        if duration <= 0 or requested_end <= 0 or requested_start >= duration or requested_end <= requested_start:
            edge = max(0, min(duration, requested_start))
            return edge, edge
        start = max(0, min(duration, requested_start))
        end = max(start, min(duration, requested_end))
        return start, end

    @staticmethod
    def _choose_level(needed: int, pixel_width: int, levels: int) -> int:
        return min(levels - 1, max(0, int(math.log2(max(1, needed / max(1, pixel_width))))))

    def close(self) -> None:
        for collection in (self._spectrum_levels, self._waveform_levels):
            for array in collection.values():
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            collection.clear()

    def __enter__(self) -> "AudioVisualizationCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


# Compatibility names keep exporters/tests and third-party imports working
# while the GUI moves to the unified visualization terminology.
SpectrogramCache = AudioVisualizationCache
build_spectrogram_cache = build_audio_visualization_cache
build_spectrogram_cache_from_slices = build_audio_visualization_cache_from_slices

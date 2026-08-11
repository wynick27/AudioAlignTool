from __future__ import annotations

import json
import math
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dataclasses import dataclass, field

from .models import AudioChapterMarker, BoundaryCandidate, SilenceSettings


@dataclass(slots=True)
class AudioProbe:
    duration_ms: int
    sample_rate: int
    channels: int
    format: str
    title: str = ""
    chapters: list[tuple[str, int, int]] = field(default_factory=list)


def decode_audio_mono(
    path: str | Path,
    target_rate: int = 16000,
    *,
    start_ms: int = 0,
    end_ms: int | None = None,
) -> tuple[np.ndarray, int]:
    """Decode to float32 mono. PyAV is preferred; PCM WAV has a stdlib fallback."""
    source = Path(path)
    try:
        import av  # type: ignore

        container = av.open(str(source))
        stream = container.streams.audio[0]
        if start_ms > 0 and stream.time_base:
            container.seek(max(0, int(start_ms / 1000 / float(stream.time_base))), stream=stream, backward=True)
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=target_rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            frame_start_ms = int(float(frame.time or 0) * 1000)
            frame_end_ms = frame_start_ms + round(frame.samples / max(1, frame.sample_rate) * 1000)
            if end_ms is not None and frame_start_ms >= end_ms:
                break
            if frame_end_ms <= start_ms:
                continue
            converted = resampler.resample(frame)
            frames = converted if isinstance(converted, list) else [converted]
            for output in frames:
                values = output.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                if frame_start_ms < start_ms:
                    trim = round((start_ms - frame_start_ms) * target_rate / 1000)
                    values = values[min(values.size, trim) :]
                if end_ms is not None and frame_end_ms > end_ms:
                    keep = round(max(0, end_ms - max(start_ms, frame_start_ms)) * target_rate / 1000)
                    values = values[:keep]
                if values.size:
                    chunks.append(values)
        container.close()
        samples = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
        expected = None if end_ms is None else max(0, int((end_ms - start_ms) * target_rate / 1000))
        if expected is not None and samples.size > expected:
            samples = samples[:expected]
        return samples, target_rate
    except ImportError:
        if source.suffix.lower() != ".wav":
            raise RuntimeError("读取非 WAV 音频需要安装 PyAV：pip install av")

    with wave.open(str(source), "rb") as handle:
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        width = handle.getsampwidth()
        frame_count = handle.getnframes()
        source_start = max(0, min(frame_count, int(start_ms * source_rate / 1000)))
        source_end = frame_count if end_ms is None else max(
            source_start, min(frame_count, int(end_ms * source_rate / 1000)),
        )
        # Reading the complete WAV for every ASR chunk made long book jobs retain
        # several full-file buffers until Python's collector ran. Decode only the
        # requested source-frame interval.
        handle.setpos(source_start)
        raw = handle.readframes(source_end - source_start)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"不支持的 WAV 位深：{width * 8}")
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        samples = (samples - 128.0) / 128.0
    else:
        samples /= float(2 ** (width * 8 - 1))
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if source_rate != target_rate and samples.size:
        source_x = np.arange(samples.size, dtype=np.float64)
        target_length = int(round(samples.size * target_rate / source_rate))
        target_x = np.linspace(0, samples.size - 1, target_length)
        samples = np.interp(target_x, source_x, samples).astype(np.float32)
    return samples, target_rate


def audio_metadata(path: str | Path) -> tuple[int, int, int]:
    probe = probe_audio(path)
    return probe.duration_ms, probe.sample_rate, probe.channels


def probe_audio(path: str | Path) -> AudioProbe:
    source = Path(path)
    try:
        import av  # type: ignore

        container = av.open(str(source))
        try:
            stream = container.streams.audio[0]
            rate = int(stream.codec_context.sample_rate or 0)
            channels = int(stream.codec_context.channels or 0)
            if stream.duration is not None:
                duration = stream.duration * stream.time_base
            elif container.duration is not None:
                duration = container.duration / 1_000_000
            else:
                duration = 0
            chapter_data: list[tuple[str, int, int]] = []
            raw_chapters = container.chapters() if callable(container.chapters) else container.chapters
            for index, chapter in enumerate(raw_chapters):
                time_base = float(chapter.time_base)
                title = chapter.metadata.get("title") or f"章节 {index + 1}"
                chapter_data.append((title, int(chapter.start * time_base * 1000), int(chapter.end * time_base * 1000)))
            title = container.metadata.get("title", "")
            format_name = source.suffix.lower().lstrip(".") or container.format.name
            return AudioProbe(int(float(duration or 0) * 1000), rate, channels, format_name, title, chapter_data)
        finally:
            container.close()
    except ImportError:
        with wave.open(str(source), "rb") as handle:
            rate = handle.getframerate()
            return AudioProbe(int(handle.getnframes() / rate * 1000), rate, handle.getnchannels(), "wav", source.stem)


def create_m4a_proxy(source: str | Path, destination: str | Path) -> Path:
    """Remux the first audio stream without re-encoding for Qt playback fallback."""
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise RuntimeError("M4B 播放代理需要 PyAV") from exc
    source_path, target = Path(source), Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    input_container = av.open(str(source_path))
    output_container = av.open(str(temporary), mode="w", format="ipod")
    input_stream = input_container.streams.audio[0]
    if hasattr(output_container, "add_stream_from_template"):
        output_stream = output_container.add_stream_from_template(input_stream)
    else:  # PyAV < 14
        output_stream = output_container.add_stream(template=input_stream)
    try:
        for packet in input_container.demux(input_stream):
            if packet.dts is None:
                continue
            packet.stream = output_stream
            output_container.mux(packet)
    finally:
        output_container.close()
        input_container.close()
    temporary.replace(target)
    return target


def detect_silence_candidates(
    samples: np.ndarray,
    sample_rate: int,
    settings: SilenceSettings,
) -> list[BoundaryCandidate]:
    settings.validate()
    if samples.size == 0:
        return []
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps  # type: ignore

        speech = get_speech_timestamps(
            samples,
            VadOptions(
                threshold=settings.vad_threshold,
                min_silence_duration_ms=settings.min_silence_ms,
                speech_pad_ms=0,
            ),
        )
        speech_ranges = [
            (max(0, int(item["start"])), min(samples.size, int(item["end"])))
            for item in speech
        ]
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in speech_ranges:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < samples.size:
            gaps.append((cursor, samples.size))
        minimum_samples = max(1, round(sample_rate * settings.min_silence_ms / 1000))
        global_rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
        result: list[BoundaryCandidate] = []
        for start, end in gaps:
            if end - start < minimum_samples:
                continue
            gap = samples[start:end]
            frame_size = max(1, round(sample_rate * 0.02))
            frame_count = max(1, math.ceil(gap.size / frame_size))
            padded = np.pad(gap, (0, frame_count * frame_size - gap.size))
            rms = np.sqrt(np.mean(np.square(padded.reshape(frame_count, frame_size)), axis=1) + 1e-12)
            valley_frame = int(np.argmin(rms))
            valley_sample = min(end - 1, start + valley_frame * frame_size + frame_size // 2)
            duration_ms = round((end - start) * 1000 / sample_rate)
            quietness = 1.0 - min(1.0, float(np.min(rms)) / max(global_rms, 1e-7))
            duration_score = min(1.0, duration_ms / max(settings.min_silence_ms * 3, 1))
            result.append(BoundaryCandidate(
                time_ms=round(valley_sample * 1000 / sample_rate),
                score=max(0.0, min(1.0, duration_score * 0.65 + quietness * 0.35)),
                kind="vad-silence",
                start_ms=round(start * 1000 / sample_rate),
                end_ms=round(end * 1000 / sample_rate),
            ))
        return result
    except (ImportError, RuntimeError, ValueError):
        # Keep an explicitly labelled energy fallback so a missing VAD model
        # never turns the UI threshold into a no-op.
        pass
    frame_ms = 20
    frame_size = max(1, sample_rate * frame_ms // 1000)
    frame_count = math.ceil(samples.size / frame_size)
    padded = np.pad(samples, (0, frame_count * frame_size - samples.size))
    rms = np.sqrt(np.mean(np.square(padded.reshape(frame_count, frame_size)), axis=1) + 1e-12)
    threshold = float(np.percentile(rms, settings.energy_percentile))
    silent = rms <= threshold
    minimum_frames = max(1, math.ceil(settings.min_silence_ms / frame_ms))
    result: list[BoundaryCandidate] = []
    start: int | None = None
    for index, value in enumerate(np.append(silent, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum_frames:
                segment = rms[start:index]
                valley = start + int(np.argmin(segment))
                start_ms = start * frame_ms
                end_ms = index * frame_ms
                score = min(1.0, (index - start) / max(1, minimum_frames * 3))
                result.append(
                    BoundaryCandidate(
                        time_ms=valley * frame_ms,
                        score=score,
                        kind="energy-silence",
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
            start = None
    return result


def analyze_silence_file(
    audio_path: str | Path,
    settings: SilenceSettings,
    output: str | Path | None = None,
) -> list[BoundaryCandidate]:
    samples, rate = decode_audio_mono(audio_path)
    candidates = detect_silence_candidates(samples, rate, settings)
    if output:
        Path(output).write_text(
            json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return candidates

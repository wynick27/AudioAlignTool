from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SegmentStatus(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"
    LOCKED = "locked"
    LOW_CONFIDENCE = "low_confidence"
    UNMATCHED = "unmatched"


class AlignmentMode(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    PRECISE = "precise"
    QWEN_FORCED = "qwen_forced"


class ASRBackendId(StrEnum):
    FASTER_WHISPER = "faster-whisper"
    QWEN3_ASR = "qwen3-asr"


class AudioVisualizationMode(StrEnum):
    NONE = "none"
    WAVEFORM = "waveform"
    SPECTROGRAM = "spectrogram"
    COMBINED = "combined"


class PlaybackFollowState(StrEnum):
    DISABLED = "disabled"
    FOLLOWING = "following"
    SUSPENDED = "suspended"


class SegmentOverlapPolicy(StrEnum):
    """How an interactive edit behaves when it reaches an adjacent cue."""

    CLAMP_CURRENT = "clamp_current"
    TRIM_NEIGHBORS = "trim_neighbors"
    ALLOW_OVERLAP = "allow_overlap"


class SegmentOrigin(StrEnum):
    SOURCE = "source"
    ASR = "asr"
    USER = "user"


class TaskLane(StrEnum):
    INFERENCE = "inference"
    MEDIA = "media"


@dataclass(slots=True, frozen=True)
class TaskHandle:
    task_id: int
    lane: TaskLane
    name: str


@dataclass(slots=True)
class TaskProgressSnapshot:
    lane: TaskLane
    name: str
    fraction: float = 0.0
    stage: str = ""
    chapter_index: int = 0
    chapter_total: int = 0
    chunk_index: int = 0
    chunk_total: int = 0
    processed_ms: int = 0
    total_ms: int = 0
    cache_hits: int = 0


@dataclass(slots=True)
class BookWorkPlan:
    chapter_total: int
    chunk_total: int
    total_audio_ms: int
    cache_hits: int = 0


@dataclass(slots=True)
class SilenceSettings:
    vad_threshold: float = 0.5
    min_silence_ms: int = 350
    boundary_padding_ms: int = 80
    snap_window_ms: int = 250
    energy_percentile: float = 18.0

    def validate(self) -> None:
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("VAD threshold must be between 0 and 1")
        if self.min_silence_ms < 20 or self.snap_window_ms < 0:
            raise ValueError("Invalid silence timing settings")
        if not 0.0 <= self.energy_percentile <= 100.0:
            raise ValueError("Energy percentile must be between 0 and 100")


@dataclass(slots=True)
class Chapter:
    id: int | None
    title: str
    position: int
    source_html: str = ""


@dataclass(slots=True)
class AudioAsset:
    id: int | None
    absolute_path: str
    relative_path: str | None = None
    fingerprint: str = ""
    duration_ms: int = 0
    sample_rate: int = 0
    channels: int = 0
    format: str = ""
    title: str = ""

    @property
    def path(self) -> Path:
        return Path(self.absolute_path)


@dataclass(slots=True)
class TextSegment:
    id: int | None
    chapter_id: int
    position: int
    text: str
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 0.0
    status: SegmentStatus = SegmentStatus.UNMATCHED
    locked: bool = False
    origin: SegmentOrigin = SegmentOrigin.SOURCE
    source_fragment_id: int | None = None
    source_start_char: int | None = None
    source_end_char: int | None = None

    def validate(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("Invalid segment time range")


@dataclass(slots=True)
class SourceFragment:
    id: int | None
    chapter_id: int
    position: int
    kind: str
    text: str
    source_start_char: int
    source_end_char: int


@dataclass(slots=True)
class ASRToken:
    id: int | None
    chapter_id: int
    position: int
    text: str
    start_ms: int
    end_ms: int
    probability: float = 0.0


@dataclass(slots=True)
class InferenceDeviceInfo:
    backend: str
    model: str
    requested_device: str = "auto"
    actual_device: str = "cpu"
    device_index: int | None = None
    device_name: str = ""
    compute_type: str = "int8"
    fallback_reason: str = ""
    runtime_paths: tuple[str, ...] = ()

    @property
    def display_text(self) -> str:
        if self.actual_device == "cuda":
            device = f"GPU {self.device_index or 0}"
            if self.device_name:
                device += f" {self.device_name}"
        else:
            device = "CPU"
        text = f"{self.backend} {self.model} · {device} · {self.compute_type}"
        if self.fallback_reason:
            text += f" · CUDA 回退：{self.fallback_reason}"
        return text


@dataclass(slots=True)
class RecognitionRun:
    id: int | None
    chapter_id: int
    cache_key: str
    backend: str
    model: str
    language: str
    audio_signature: str
    parameters_json: str
    status: str = "pending"
    actual_device: str = ""
    compute_type: str = ""
    device_name: str = ""
    fallback_reason: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class RecognitionChunk:
    id: int | None
    run_id: int
    position: int
    source_start_ms: int
    source_end_ms: int
    core_start_ms: int
    core_end_ms: int
    status: str = "pending"
    transcript: str = ""
    elapsed_ms: int = 0
    error: str = ""


@dataclass(slots=True)
class AudioChapterMarker:
    id: int | None
    audio_id: int
    position: int
    title: str
    start_ms: int
    end_ms: int


@dataclass(slots=True)
class ChapterAudioLink:
    id: int | None
    chapter_id: int
    audio_id: int
    position: int = 0
    source_start_ms: int = 0
    source_end_ms: int = 0
    confidence: float = 0.0


@dataclass(slots=True)
class TextAudioAnchor:
    id: int | None
    chapter_id: int
    segment_id: int | None
    source_start_char: int
    source_end_char: int
    start_ms: int
    end_ms: int
    confidence: float = 0.0
    method: str = "asr"


@dataclass(slots=True)
class SelectionRange:
    start_ms: int
    end_ms: int

    def normalized(self) -> "SelectionRange":
        return SelectionRange(min(self.start_ms, self.end_ms), max(self.start_ms, self.end_ms))


@dataclass(slots=True)
class BoundaryCandidate:
    time_ms: int
    score: float
    kind: str = "silence"
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(slots=True)
class SilenceAlignmentOptions:
    start_segment_index: int
    start_ms: int
    end_ms: int
    padding_ms: int = 80
    low_confidence: float = 0.58


@dataclass(slots=True)
class ProjectManifest:
    schema_version: int = 2
    project_id: str = ""
    title: str = "未命名项目"
    language: str = "auto"
    whisper_model: str = "small"
    asr_backend: ASRBackendId = ASRBackendId.FASTER_WHISPER
    qwen_model: str = "Qwen3-ASR-0.6B"
    alignment_mode: AlignmentMode = AlignmentMode.BALANCED
    segment_overlap_policy: SegmentOverlapPolicy = SegmentOverlapPolicy.CLAMP_CURRENT
    silence: SilenceSettings = field(default_factory=SilenceSettings)
    created_at: str = ""
    updated_at: str = ""
    source_name: str = ""
    media_references: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["alignment_mode"] = self.alignment_mode.value
        data["asr_backend"] = self.asr_backend.value
        data["segment_overlap_policy"] = self.segment_overlap_policy.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectManifest":
        payload = dict(data)
        payload["alignment_mode"] = AlignmentMode(payload.get("alignment_mode", "balanced"))
        payload["asr_backend"] = ASRBackendId(payload.get("asr_backend", ASRBackendId.FASTER_WHISPER.value))
        payload["segment_overlap_policy"] = SegmentOverlapPolicy(
            payload.get("segment_overlap_policy", SegmentOverlapPolicy.CLAMP_CURRENT.value)
        )
        payload["silence"] = SilenceSettings(**payload.get("silence", {}))
        return cls(**payload)

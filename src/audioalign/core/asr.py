from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Protocol, Sequence

from .models import ASRBackendId, ASRToken, AlignmentMode, BoundaryCandidate, InferenceDeviceInfo
from .audio import decode_audio_mono
from .runtime import bootstrap_native_runtime


ProgressCallback = Callable[[float, str], None]


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class ASROptions:
    backend: ASRBackendId = ASRBackendId.FASTER_WHISPER
    model: str = "small"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    mode: AlignmentMode = AlignmentMode.BALANCED
    vad_threshold: float = 0.5
    min_silence_ms: int = 350
    model_root: str | None = None
    clip_start_ms: int = 0
    clip_end_ms: int | None = None

    def parameters(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "model": self.model,
            "language": self.language or "auto",
            "device": self.device,
            "compute_type": self.compute_type,
            "mode": self.mode.value,
            "vad_threshold": round(self.vad_threshold, 4),
            "min_silence_ms": self.min_silence_ms,
            "chunker": "vad-core-v1",
        }


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    runtime_available: bool
    model_available: bool
    whisperx_available: bool
    cuda_available: bool
    message: str
    backend: str = ASRBackendId.FASTER_WHISPER.value
    compute_types: tuple[str, ...] = ()
    runtime_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AudioChunkPlan:
    position: int
    start_ms: int
    end_ms: int
    core_start_ms: int
    core_end_ms: int


def plan_audio_chunks(
    start_ms: int,
    end_ms: int,
    candidates: Sequence[BoundaryCandidate] = (),
    *,
    target_ms: int = 120_000,
    maximum_ms: int = 180_000,
    overlap_ms: int = 1_500,
) -> list[AudioChunkPlan]:
    """Plan bounded chunks whose non-overlapping cores meet at long pauses."""
    start_ms, end_ms = max(0, int(start_ms)), max(0, int(end_ms))
    if end_ms <= start_ms:
        return []
    pauses = sorted(
        candidate.time_ms for candidate in candidates
        if start_ms + 20_000 < candidate.time_ms < end_ms - 20_000
    )
    boundaries = [start_ms]
    cursor = start_ms
    while end_ms - cursor > maximum_ms:
        target = cursor + target_ms
        upper = min(end_ms, cursor + maximum_ms)
        lower = cursor + max(30_000, target_ms // 2)
        choices = [value for value in pauses if lower <= value <= upper]
        boundary = min(choices, key=lambda value: abs(value - target)) if choices else upper
        if boundary <= cursor:
            break
        boundaries.append(boundary)
        cursor = boundary
    boundaries.append(end_ms)
    result: list[AudioChunkPlan] = []
    for index, (core_start, core_end) in enumerate(zip(boundaries, boundaries[1:])):
        raw_start = max(start_ms, core_start - (overlap_ms if index else 0))
        raw_end = min(end_ms, core_end + (overlap_ms if index + 1 < len(boundaries) - 1 else 0))
        result.append(AudioChunkPlan(index, raw_start, raw_end, core_start, core_end))
    return result


def recognition_cache_key(audio_signature: str, options: ASROptions) -> tuple[str, str]:
    parameters = json.dumps(options.parameters(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{audio_signature}|{parameters}".encode("utf-8")).hexdigest()
    return digest, parameters


def cuda_runtime_available() -> bool:
    """Return true only when CTranslate2 and its CUDA runtime can both load."""
    try:
        bootstrap_native_runtime()
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        if __import__("os").name == "nt":
            import ctypes

            # Current CTranslate2 Windows wheels use CUDA 12 and cuDNN 9.
            # Device enumeration succeeds even when these runtime DLLs are absent.
            ctypes.WinDLL("cublas64_12.dll")
            ctypes.WinDLL("cudnn64_9.dll")
        return True
    except (ImportError, OSError, RuntimeError):
        return False


def _is_cuda_runtime_error(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return any(marker in message for marker in ("cublas", "cudnn", "cuda", "cufft", "cannot be loaded"))


def runtime_status(
    model: str,
    model_root: str | Path,
    backend: ASRBackendId | str = ASRBackendId.FASTER_WHISPER,
) -> RuntimeStatus:
    import importlib.util

    backend = ASRBackendId(backend)
    runtime_module = "qwen_asr" if backend == ASRBackendId.QWEN3_ASR else "faster_whisper"
    runtime = importlib.util.find_spec(runtime_module) is not None
    precise = importlib.util.find_spec("whisperx") is not None
    cuda = False
    compute_types: tuple[str, ...] = ()
    paths = bootstrap_native_runtime()
    if runtime and backend == ASRBackendId.FASTER_WHISPER:
        cuda = cuda_runtime_available()
        if cuda:
            try:
                import ctranslate2  # type: ignore
                compute_types = tuple(sorted(ctranslate2.get_supported_compute_types("cuda", 0)))
            except (ImportError, RuntimeError):
                pass
    elif runtime:
        try:
            import torch  # type: ignore
            cuda = bool(torch.cuda.is_available())
            compute_types = ("bfloat16", "float16") if cuda else ("float32",)
        except ImportError:
            pass
    root = Path(model_root)
    if backend == ASRBackendId.QWEN3_ASR:
        local_name = model.casefold().replace("qwen/", "").replace("-", "_").replace(".", "_")
        available = _qwen_model_is_ready(root / local_name)
    else:
        available = (root / model / "model.bin").is_file() and (root / model / "config.json").is_file()
    if not runtime:
        message = f"{runtime_module} 运行库缺失"
    elif not available:
        device_preview = ""
        if backend == ASRBackendId.QWEN3_ASR:
            if cuda:
                device_name = ""
                try:
                    import torch  # type: ignore

                    device_name = torch.cuda.get_device_name(0)
                except (ImportError, RuntimeError):
                    pass
                device_preview = f" · 将使用 GPU 0 {device_name} · bfloat16".rstrip()
            else:
                reason = "CPU 版 PyTorch"
                try:
                    import torch  # type: ignore

                    if torch.version.cuda is not None:
                        reason = "CUDA 不可用"
                except ImportError:
                    reason = "PyTorch 缺失"
                device_preview = f" · 将使用 CPU · float32 · {reason}"
        message = f"模型 {model} 尚未下载{device_preview}"
    elif cuda:
        device_name = ""
        if backend == ASRBackendId.QWEN3_ASR:
            try:
                import torch  # type: ignore

                device_name = torch.cuda.get_device_name(0)
            except (ImportError, RuntimeError):
                pass
        device_suffix = f" 0 {device_name}".rstrip()
        compute = "bfloat16" if backend == ASRBackendId.QWEN3_ASR else "CUDA"
        message = f"模型 {model} 已就绪 · GPU{device_suffix} · {compute}"
    else:
        reason = ""
        if backend == ASRBackendId.QWEN3_ASR:
            try:
                import torch  # type: ignore

                reason = " · CPU 版 PyTorch" if torch.version.cuda is None else " · CUDA 不可用"
            except ImportError:
                reason = " · PyTorch 缺失"
        compute = "float32" if backend == ASRBackendId.QWEN3_ASR else "INT8"
        message = f"模型 {model} 已就绪 · CPU · {compute}{reason}"
    return RuntimeStatus(runtime, available, precise, cuda, message, backend.value, compute_types, paths)


class Transcriber(Protocol):
    def transcribe(self, path: str | Path, chapter_id: int, options: ASROptions, progress: ProgressCallback | None = None) -> list[ASRToken]: ...


class FasterWhisperTranscriber:
    def __init__(self) -> None:
        self.last_device_info = InferenceDeviceInfo(ASRBackendId.FASTER_WHISPER.value, "")
        self._model = None
        self._model_key: tuple[str, str, str] | None = None
        self._forced_cpu_reason = ""

    def transcribe(
        self,
        path: str | Path,
        chapter_id: int,
        options: ASROptions,
        progress: ProgressCallback | None = None,
    ) -> list[ASRToken]:
        try:
            runtime_paths = bootstrap_native_runtime()
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise BackendUnavailableError("faster-whisper 运行库缺失；请使用项目 .venv 启动程序") from exc
        device = options.device
        if device == "auto":
            if self._forced_cpu_reason:
                device = "cpu"
            else:
                try:
                    device = "cuda" if cuda_runtime_available() else "cpu"
                except Exception:
                    device = "cpu"
        compute = options.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        if progress:
            progress(-1.0, f"加载模型 {options.model}")
        model_root = Path(options.model_root) if options.model_root else None
        local_model = model_root / options.model if model_root else None
        if local_model and not ((local_model / "model.bin").is_file() and (local_model / "config.json").is_file()):
            from faster_whisper.utils import download_model  # type: ignore

            local_model.mkdir(parents=True, exist_ok=True)
            if progress:
                progress(-1.0, f"正在下载模型 {options.model} 到 {local_model}")
            download_model(options.model, output_dir=str(local_model))
        model_name = str(local_model) if local_model else options.model
        audio_input: str | object = str(path)
        if options.clip_start_ms or options.clip_end_ms is not None:
            audio_input, _ = decode_audio_mono(
                path,
                16000,
                start_ms=options.clip_start_ms,
                end_ms=options.clip_end_ms,
            )
        def collect(active_model) -> list[ASRToken]:
            segments, info = active_model.transcribe(
                audio_input, language=options.language, word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "threshold": options.vad_threshold,
                    "min_silence_duration_ms": options.min_silence_ms,
                },
            )
            duration = max(1.0, float(info.duration))
            result: list[ASRToken] = []
            for segment in segments:
                for word in segment.words or []:
                    result.append(
                        ASRToken(
                            id=None, chapter_id=chapter_id, position=len(result), text=word.word,
                            start_ms=int(word.start * 1000), end_ms=int(word.end * 1000),
                            probability=float(word.probability),
                        )
                    )
                if progress:
                    progress(min(0.98, float(segment.end) / duration), segment.text.strip())
            return result

        model_key = (model_name, device, compute)
        try:
            if self._model is None or self._model_key != model_key:
                self._model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute,
                    download_root=str(model_root) if model_root else None,
                )
                self._model_key = model_key
            model = self._model
            tokens = collect(model)
            fallback_reason = self._forced_cpu_reason
        except RuntimeError as exc:
            if device != "cuda" or options.device != "auto" or not _is_cuda_runtime_error(exc):
                raise
            self._model = None
            self._model_key = None
            if progress:
                progress(-1.0, "CUDA 运行库不完整，已自动回退到 CPU INT8")
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(model_root) if model_root else None,
            )
            self._model = model
            self._model_key = (model_name, "cpu", "int8")
            tokens = collect(model)
            fallback_reason = str(exc)
            self._forced_cpu_reason = fallback_reason
            device, compute = "cpu", "int8"
        self.last_device_info = InferenceDeviceInfo(
            ASRBackendId.FASTER_WHISPER.value,
            options.model,
            options.device,
            device,
            0 if device == "cuda" else None,
            "",
            compute,
            fallback_reason,
            runtime_paths,
        )
        if progress:
            progress(1.0, "识别完成")
        return tokens


QWEN_ASR_LANGUAGE_NAMES = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "de": "German",
    "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "ko": "Korean", "ja": "Japanese", "ar": "Arabic",
    "id": "Indonesian", "th": "Thai", "vi": "Vietnamese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian",
    "hu": "Hungarian", "mk": "Macedonian",
}

QWEN_FORCED_LANGUAGE_CODES = (
    "zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es",
)


def _qwen_repo_id(model: str) -> str:
    return model if "/" in model else f"Qwen/{model}"


def _qwen_local_directory(model_root: str | Path | None, model: str) -> Path | None:
    if not model_root:
        return None
    name = model.casefold().replace("qwen/", "").replace("-", "_").replace(".", "_")
    return Path(model_root) / name


def _qwen_model_is_ready(local: Path) -> bool:
    """Reject half-downloaded snapshots even when config.json arrived first."""
    if not (local / "config.json").is_file():
        return False
    index_files = list(local.glob("*.safetensors.index.json"))
    if index_files:
        try:
            weight_map = json.loads(index_files[0].read_text(encoding="utf-8")).get("weight_map", {})
            return bool(weight_map) and all((local / name).is_file() for name in set(weight_map.values()))
        except (OSError, ValueError, TypeError, AttributeError):
            return False
    return any(local.glob("*.safetensors")) or (local / "pytorch_model.bin").is_file()


def _download_error_summary(error: Exception) -> str:
    message = " ".join(str(error).split())
    if len(message) > 240:
        message = message[:237] + "..."
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _ensure_qwen_model(model_root: str | Path | None, model: str, progress: ProgressCallback | None) -> str:
    local = _qwen_local_directory(model_root, model)
    if local is None:
        return _qwen_repo_id(model)
    if not _qwen_model_is_ready(local):
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:
            raise BackendUnavailableError("下载 Qwen 模型需要 huggingface-hub") from exc
        local.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(-1.0, f"正在通过 Hugging Face 下载 {model}")
        huggingface_error: Exception | None = None
        try:
            snapshot_download(repo_id=_qwen_repo_id(model), local_dir=str(local))
        except Exception as exc:
            huggingface_error = exc
            if progress:
                progress(-1.0, f"Hugging Face 下载失败，正在切换 ModelScope · {_download_error_summary(exc)}")
            try:
                from modelscope import snapshot_download as modelscope_snapshot_download  # type: ignore
            except ImportError as import_error:
                raise BackendUnavailableError(
                    "Hugging Face 下载失败，且未安装 ModelScope 回退运行库；请安装 modelscope 后重试。"
                    f"\nHugging Face: {_download_error_summary(exc)}"
                ) from import_error
            try:
                modelscope_snapshot_download(_qwen_repo_id(model), local_dir=str(local))
            except Exception as modelscope_error:
                raise BackendUnavailableError(
                    "Hugging Face 和 ModelScope 均无法下载模型。"
                    f"\nHugging Face: {_download_error_summary(exc)}"
                    f"\nModelScope: {_download_error_summary(modelscope_error)}"
                ) from modelscope_error
        if not _qwen_model_is_ready(local):
            source = "ModelScope" if huggingface_error is not None else "Hugging Face"
            raise BackendUnavailableError(f"{source} 下载结束，但模型文件不完整：{local}")
        if progress:
            source = "ModelScope" if huggingface_error is not None else "Hugging Face"
            progress(-1.0, f"{model} 下载完成 · {source}")
    return str(local)


def _qwen_device(options: ASROptions):
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise BackendUnavailableError("Qwen3-ASR 需要 PyTorch 运行库") from exc
    device = options.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return torch, str(device), dtype


def _qwen_timestamp_value(value) -> int:
    numeric = float(value)
    return round(numeric * 1000)


class Qwen3ASRTranscriber:
    def __init__(self) -> None:
        self.last_device_info = InferenceDeviceInfo(ASRBackendId.QWEN3_ASR.value, "")
        self._model = None
        self._model_key: tuple[str, str] | None = None

    def transcribe(
        self,
        path: str | Path,
        chapter_id: int,
        options: ASROptions,
        progress: ProgressCallback | None = None,
    ) -> list[ASRToken]:
        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore
        except ImportError as exc:
            raise BackendUnavailableError("Qwen3-ASR 运行库缺失；请在项目 Python 3.14 环境安装 qwen-asr") from exc
        torch, device, dtype = _qwen_device(options)
        model_path = _ensure_qwen_model(options.model_root, options.model, progress)
        aligner_path = _ensure_qwen_model(options.model_root, "Qwen3-ForcedAligner-0.6B", progress)
        if progress:
            progress(-1.0, f"加载 {options.model} · {'GPU' if device.startswith('cuda') else 'CPU'}")
        model_key = (model_path, device)
        if self._model is None or self._model_key != model_key:
            self._model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=1,
                max_new_tokens=4096,
                forced_aligner=aligner_path,
                forced_aligner_kwargs={"dtype": dtype, "device_map": device},
            )
            self._model_key = model_key
        model = self._model
        samples, sample_rate = decode_audio_mono(
            path, 16000, start_ms=options.clip_start_ms, end_ms=options.clip_end_ms,
        )
        language = QWEN_ASR_LANGUAGE_NAMES.get((options.language or "").casefold(), options.language)
        results = model.transcribe(
            audio=(samples, sample_rate), language=language, return_time_stamps=True,
        )
        record = results[0]
        tokens: list[ASRToken] = []
        for item in getattr(record, "time_stamps", None) or []:
            text = str(getattr(item, "text", ""))
            start = getattr(item, "start_time", None)
            end = getattr(item, "end_time", None)
            if start is None or end is None:
                continue
            tokens.append(ASRToken(
                None, chapter_id, len(tokens), text,
                _qwen_timestamp_value(start), _qwen_timestamp_value(end), 0.8,
            ))
        if not tokens and getattr(record, "text", ""):
            duration = max(0, (options.clip_end_ms or options.clip_start_ms) - options.clip_start_ms)
            tokens.append(ASRToken(None, chapter_id, 0, str(record.text), 0, duration, 0.45))
        device_name = ""
        if device.startswith("cuda"):
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        self.last_device_info = InferenceDeviceInfo(
            ASRBackendId.QWEN3_ASR.value, options.model, options.device,
            "cuda" if device.startswith("cuda") else "cpu",
            0 if device.startswith("cuda") else None, device_name,
            "bfloat16" if device.startswith("cuda") else "float32",
        )
        if progress:
            progress(1.0, "Qwen3-ASR 识别完成")
        return tokens


class Qwen3ForcedAligner:
    """Direct known-text alignment for selected regions up to four minutes."""

    SUPPORTED_LANGUAGES = frozenset(QWEN_FORCED_LANGUAGE_CODES)

    def __init__(self) -> None:
        self.last_device_info = InferenceDeviceInfo("qwen3-forced-aligner", "Qwen3-ForcedAligner-0.6B")
        self._model = None
        self._model_key: tuple[str, str, str] | None = None

    def align(
        self,
        path: str | Path,
        text: str,
        language: str,
        chapter_id: int,
        options: ASROptions,
        progress: ProgressCallback | None = None,
    ) -> list[ASRToken]:
        duration = None if options.clip_end_ms is None else options.clip_end_ms - options.clip_start_ms
        if duration is not None and duration > 240_000:
            raise ValueError("Qwen ForcedAligner 单个选区不能超过 240 秒")
        try:
            from qwen_asr import Qwen3ForcedAligner as QwenAlignerModel  # type: ignore
        except ImportError as exc:
            raise BackendUnavailableError("Qwen ForcedAligner 运行库缺失") from exc
        torch, device, dtype = _qwen_device(options)
        model_path = _ensure_qwen_model(options.model_root, "Qwen3-ForcedAligner-0.6B", progress)
        model_key = (model_path, device, str(dtype))
        if self._model is None or self._model_key != model_key:
            self._model = QwenAlignerModel.from_pretrained(model_path, dtype=dtype, device_map=device)
            self._model_key = model_key
        model = self._model
        samples, sample_rate = decode_audio_mono(
            path, 16000, start_ms=options.clip_start_ms, end_ms=options.clip_end_ms,
        )
        language_name = QWEN_ASR_LANGUAGE_NAMES.get(language.casefold(), language)
        results = model.align(audio=(samples, sample_rate), text=text, language=language_name)
        tokens: list[ASRToken] = []
        for item in results[0]:
            tokens.append(ASRToken(
                None, chapter_id, len(tokens), str(item.text),
                _qwen_timestamp_value(item.start_time), _qwen_timestamp_value(item.end_time), 0.9,
            ))
        device_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else ""
        self.last_device_info = InferenceDeviceInfo(
            "qwen3-forced-aligner", "Qwen3-ForcedAligner-0.6B", options.device,
            "cuda" if device.startswith("cuda") else "cpu", 0 if device.startswith("cuda") else None,
            device_name, "bfloat16" if device.startswith("cuda") else "float32",
        )
        return tokens


class WhisperXTranscriber:
    """Optional full-chapter precise alignment backend."""

    def __init__(self) -> None:
        self.last_device_info = InferenceDeviceInfo("whisperx", "")

    def transcribe(
        self,
        path: str | Path,
        chapter_id: int,
        options: ASROptions,
        progress: ProgressCallback | None = None,
    ) -> list[ASRToken]:
        try:
            import whisperx  # type: ignore
        except ImportError as exc:
            raise BackendUnavailableError("WhisperX 识别并精确对齐工作流需要安装 WhisperX 组件") from exc
        device = options.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if progress:
            progress(-1.0, "加载 WhisperX")
        audio = whisperx.load_audio(str(path))
        if options.clip_start_ms or options.clip_end_ms is not None:
            start = round(options.clip_start_ms * 16000 / 1000)
            end = None if options.clip_end_ms is None else round(options.clip_end_ms * 16000 / 1000)
            audio = audio[start:end]
        compute_type = "float16" if device == "cuda" else "int8"
        model = whisperx.load_model(options.model, device, compute_type=compute_type)
        result = model.transcribe(audio, language=options.language)
        language = result.get("language") or options.language
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)
        tokens: list[ASRToken] = []
        for word in aligned.get("word_segments", []):
            if "start" not in word or "end" not in word:
                continue
            tokens.append(
                ASRToken(
                    id=None, chapter_id=chapter_id, position=len(tokens), text=word.get("word", ""),
                    start_ms=int(word["start"] * 1000), end_ms=int(word["end"] * 1000),
                    probability=float(word.get("score", 0.0)),
                )
            )
        if progress:
            progress(1.0, "精确对齐完成")
        self.last_device_info = InferenceDeviceInfo(
            "whisperx", options.model, options.device, device,
            0 if device == "cuda" else None, "", compute_type,
        )
        return tokens


def transcriber_for_mode(mode: AlignmentMode) -> Transcriber:
    return WhisperXTranscriber() if mode == AlignmentMode.PRECISE else FasterWhisperTranscriber()


def transcriber_for_options(options: ASROptions) -> Transcriber:
    if options.backend == ASRBackendId.QWEN3_ASR:
        return Qwen3ASRTranscriber()
    return transcriber_for_mode(options.mode)

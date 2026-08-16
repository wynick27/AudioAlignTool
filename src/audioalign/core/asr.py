from __future__ import annotations

from dataclasses import dataclass
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable, Protocol, Sequence

from .models import (
    ASRBackendId, ASRToken, AlignmentMode, BoundaryCandidate,
    InferenceDeviceInfo, SilenceSettings,
)
from .audio import decode_audio_mono, detect_silence_candidates
from .runtime import bootstrap_native_runtime
from .runtime_addons import active_runtime_manifest


ProgressCallback = Callable[[float, str], None]
_qwen_cuda_disabled_reason = ""

# Qwen's timestamp path internally splits at 180 seconds. Passing one of our
# 180-second cores with 1.5-second overlap on both sides produces a 183-second
# request and a tiny, context-free 3-second tail inside Qwen. Split only inputs
# that exceed the upstream limit, preferably at a VAD silence near 150 seconds.
QWEN_MAX_INPUT_MS = 180_000
QWEN_SPLIT_TARGET_MS = 150_000
QWEN_MIN_SUBCHUNK_MS = 30_000
QWEN_MAX_NEW_TOKENS = 1_024


class BackendUnavailableError(RuntimeError):
    pass


class InferenceMemoryPressureError(RuntimeError):
    """Raised before the operating system has to terminate an inference process."""


@dataclass(frozen=True, slots=True)
class InferenceMemoryStatus:
    available_bytes: int
    total_bytes: int
    gpu_available_bytes: int | None = None
    gpu_total_bytes: int | None = None


def inference_memory_status() -> InferenceMemoryStatus:
    """Read memory counters without adding a psutil dependency or importing torch."""
    available = total = 0
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MemoryStatusEx()
        state.dwLength = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            available, total = int(state.ullAvailPhys), int(state.ullTotalPhys)
    elif hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
            total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            pass

    gpu_available = gpu_total = None
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                gpu_available, gpu_total = (int(value) for value in torch.cuda.mem_get_info())
        except Exception:
            pass
    return InferenceMemoryStatus(available, total, gpu_available, gpu_total)


def release_inference_memory(transcriber=None, *, aggressive: bool = False) -> None:
    """Release per-chunk objects while deliberately retaining the loaded model."""
    release = getattr(transcriber, "release_temporary_memory", None)
    if callable(release):
        release(aggressive=aggressive)
        return
    if aggressive:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def ensure_inference_memory_headroom(transcriber=None) -> InferenceMemoryStatus:
    """Clean up early and stop safely if either host RAM or VRAM is critically low."""
    status = inference_memory_status()
    cleanup_limit = max(768 * 1024**2, round(status.total_bytes * 0.06)) if status.total_bytes else 0
    gpu_cleanup_limit = 768 * 1024**2
    needs_cleanup = (
        (cleanup_limit and status.available_bytes < cleanup_limit)
        or (status.gpu_available_bytes is not None and status.gpu_available_bytes < gpu_cleanup_limit)
    )
    if needs_cleanup:
        release_inference_memory(transcriber, aggressive=True)
        status = inference_memory_status()
    stop_limit = max(384 * 1024**2, round(status.total_bytes * 0.03)) if status.total_bytes else 0
    if stop_limit and status.available_bytes < stop_limit:
        raise InferenceMemoryPressureError(
            f"系统可用内存只剩 {status.available_bytes / 1024**2:.0f} MB"
        )
    if status.gpu_available_bytes is not None and status.gpu_available_bytes < 256 * 1024**2:
        raise InferenceMemoryPressureError(
            f"GPU 可用显存只剩 {status.gpu_available_bytes / 1024**2:.0f} MB"
        )
    return status


def is_inference_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, MemoryError):
        return True
    message = str(error).casefold()
    return any(marker in message for marker in (
        "out of memory", "cuda out of memory", "not enough memory",
        "cannot allocate memory", "bad allocation", "cublas_status_alloc_failed",
    ))


def _is_fatal_torch_cuda_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".casefold()
    return is_inference_out_of_memory(error) or any(marker in message for marker in (
        "cuda error",
        "illegal memory access",
        "device-side assert",
        "unspecified launch failure",
        "acceleratorerror",
        "cudnn_status",
        "cublas_status",
    ))


def qwen_cuda_disabled_reason() -> str:
    return _qwen_cuda_disabled_reason


def _disable_qwen_cuda(error: BaseException) -> str:
    global _qwen_cuda_disabled_reason
    summary = " ".join(str(error).split())
    if len(summary) > 300:
        summary = summary[:297] + "..."
    _qwen_cuda_disabled_reason = summary or type(error).__name__
    return _qwen_cuda_disabled_reason


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
    minimum_tail_ms: int = 0,
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
        tail_limit = end_ms - max(0, minimum_tail_ms) if minimum_tail_ms else end_ms
        upper = min(tail_limit, cursor + maximum_ms)
        target = min(upper, cursor + target_ms)
        lower = cursor + max(30_000, target_ms // 2)
        choices = [value for value in pauses if lower <= value <= upper]
        boundary = min(choices, key=lambda value: abs(value - target)) if choices else (
            target if minimum_tail_ms else upper
        )
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


def plan_recognition_chunks(
    start_ms: int,
    end_ms: int,
    candidates: Sequence[BoundaryCandidate],
    options: ASROptions,
    *,
    preserve_existing_plan: bool = False,
) -> list[AudioChunkPlan]:
    """Plan new Qwen runs below its limit while retaining resumable v1 plans."""
    if options.backend == ASRBackendId.QWEN3_ASR and not preserve_existing_plan:
        # Interior chunks receive 1.5 seconds of context on both sides, so a
        # 177-second core is the largest one whose actual request stays within
        # Qwen's 180-second forced-alignment limit.
        return plan_audio_chunks(
            start_ms,
            end_ms,
            candidates,
            target_ms=QWEN_SPLIT_TARGET_MS,
            maximum_ms=QWEN_MAX_INPUT_MS - 3_000,
            overlap_ms=1_500,
            minimum_tail_ms=QWEN_MIN_SUBCHUNK_MS,
        )
    return plan_audio_chunks(start_ms, end_ms, candidates)


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
    *,
    probe_device: bool = True,
) -> RuntimeStatus:
    import importlib.util

    backend = ASRBackendId(backend)
    runtime_module = "qwen_asr" if backend == ASRBackendId.QWEN3_ASR else "faster_whisper"
    runtime = importlib.util.find_spec(runtime_module) is not None
    precise = (
        importlib.util.find_spec("whisperx") is not None
        or active_runtime_manifest("whisperx") is not None
    )
    cuda = False
    compute_types: tuple[str, ...] = ()
    paths = bootstrap_native_runtime()
    if probe_device and runtime and backend == ASRBackendId.FASTER_WHISPER:
        cuda = cuda_runtime_available()
        if cuda:
            try:
                import ctranslate2  # type: ignore
                compute_types = tuple(sorted(ctranslate2.get_supported_compute_types("cuda", 0)))
            except (ImportError, RuntimeError):
                pass
    elif probe_device and runtime:
        try:
            import torch  # type: ignore
            cuda = bool(torch.cuda.is_available()) and not bool(qwen_cuda_disabled_reason())
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
        if probe_device and backend == ASRBackendId.QWEN3_ASR:
            if cuda:
                device_name = ""
                try:
                    import torch  # type: ignore

                    device_name = torch.cuda.get_device_name(0)
                except (ImportError, RuntimeError):
                    pass
                device_preview = f" · 将使用 GPU 0 {device_name} · bfloat16".rstrip()
            else:
                reason = (
                    f"本次运行 CUDA 已熔断：{qwen_cuda_disabled_reason()}"
                    if qwen_cuda_disabled_reason() else "CPU 版 PyTorch"
                )
                try:
                    import torch  # type: ignore

                    if torch.version.cuda is not None and not qwen_cuda_disabled_reason():
                        reason = "CUDA 不可用"
                except ImportError:
                    reason = "PyTorch 缺失"
                device_preview = f" · 将使用 CPU · float32 · {reason}"
        message = f"模型 {model} 尚未下载{device_preview}"
    elif not probe_device:
        message = f"模型 {model} 已就绪 · 设备状态检测中（不阻塞界面）"
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
            if qwen_cuda_disabled_reason():
                reason = f" · CUDA 已熔断并回退 CPU：{qwen_cuda_disabled_reason()}"
            else:
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
    def release_temporary_memory(self, *, aggressive: bool = False) -> None: ...


class FasterWhisperTranscriber:
    def __init__(self) -> None:
        self.last_device_info = InferenceDeviceInfo(ASRBackendId.FASTER_WHISPER.value, "")
        self._model = None
        self._model_key: tuple[str, str, str] | None = None
        self._forced_cpu_reason = ""

    def release_temporary_memory(self, *, aggressive: bool = False) -> None:
        # The CTranslate2 model is intentionally retained; decoded NumPy arrays and
        # generator frames are ordinary Python objects and can be reclaimed here.
        if aggressive:
            gc.collect()

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
            segments = None
            try:
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
            finally:
                close = getattr(segments, "close", None)
                if callable(close):
                    close()

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
        device = (
            "cuda:0"
            if torch.cuda.is_available() and not qwen_cuda_disabled_reason()
            else "cpu"
        )
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return torch, str(device), dtype


def _load_qwen_with_cpu_fallback(
    loader,
    *,
    torch,
    device: str,
    dtype,
    progress: ProgressCallback | None,
):
    """Load on CUDA once, then permanently use CPU after a fatal CUDA failure."""
    try:
        return loader(device, dtype), device, dtype, ""
    except Exception as exc:
        # Retrying a CUDA OOM by loading the same model on CPU can double host
        # memory pressure. On Windows that path can terminate in torch_cpu.dll
        # before Python gets a chance to report the exception.
        if device.startswith("cuda") and is_inference_out_of_memory(exc):
            raise InferenceMemoryPressureError(
                f"Qwen GPU model load ran out of memory: {exc}"
            ) from exc
        if not device.startswith("cuda") or not _is_fatal_torch_cuda_error(exc):
            raise
        reason = _disable_qwen_cuda(exc)
        if progress:
            progress(
                -1.0,
                "Qwen CUDA 模型加载失败，已切换 CPU float32；"
                f"本次运行不再重试 GPU · {reason}",
            )
        exc.__traceback__ = None
        # Do not call any CUDA cleanup API here. After an illegal memory access
        # even mem_get_info()/empty_cache() may report the previous async error.
        gc.collect()
        cpu_device, cpu_dtype = "cpu", torch.float32
        return loader(cpu_device, cpu_dtype), cpu_device, cpu_dtype, reason


def _qwen_timestamp_value(value) -> int:
    numeric = float(value)
    return round(numeric * 1000)


def _qwen_vad_subchunk_ranges(
    samples,
    sample_rate: int,
    options: ASROptions,
) -> list[tuple[int, int]]:
    """Split only over-limit Qwen inputs, preferring a real VAD silence."""
    total_samples = len(samples)
    maximum_samples = max(1, round(sample_rate * QWEN_MAX_INPUT_MS / 1000))
    if total_samples <= maximum_samples:
        return [(0, total_samples)] if total_samples else []

    settings = SilenceSettings(
        vad_threshold=options.vad_threshold,
        min_silence_ms=options.min_silence_ms,
    )
    candidates = detect_silence_candidates(samples, sample_rate, settings)
    candidate_samples = sorted(
        round(candidate.time_ms * sample_rate / 1000)
        for candidate in candidates
    )
    minimum_samples = max(1, round(sample_rate * QWEN_MIN_SUBCHUNK_MS / 1000))
    target_samples = max(1, round(sample_rate * QWEN_SPLIT_TARGET_MS / 1000))

    boundaries = [0]
    cursor = 0
    while total_samples - cursor > maximum_samples:
        # Keep enough material for the final part. This avoids recreating the
        # pathological 180-second + 3-second split inside Qwen.
        latest = min(cursor + maximum_samples, total_samples - minimum_samples)
        earliest = cursor + minimum_samples
        target = min(cursor + target_samples, latest)
        nearby = [value for value in candidate_samples if earliest <= value <= latest]
        boundary = min(nearby, key=lambda value: abs(value - target)) if nearby else target
        if boundary <= cursor:
            boundary = latest
        boundaries.append(boundary)
        cursor = boundary
    boundaries.append(total_samples)
    return list(zip(boundaries, boundaries[1:]))


class Qwen3ASRTranscriber:
    def __init__(self) -> None:
        self.last_device_info = InferenceDeviceInfo(ASRBackendId.QWEN3_ASR.value, "")
        self._model = None
        self._model_key: tuple[str, str] | None = None

    def release_temporary_memory(self, *, aggressive: bool = False) -> None:
        if aggressive:
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None:
                try:
                    if torch.cuda.is_available() and torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

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
            raise BackendUnavailableError("Qwen3-ASR 运行库缺失；请安装 qwen-asr 或使用运行时组件管理器") from exc
        torch, device, dtype = _qwen_device(options)
        model_path = _ensure_qwen_model(options.model_root, options.model, progress)
        aligner_path = _ensure_qwen_model(options.model_root, "Qwen3-ForcedAligner-0.6B", progress)
        if progress:
            progress(-1.0, f"加载 {options.model} · {'GPU' if device.startswith('cuda') else 'CPU'}")
        model_key = (model_path, device)
        fallback_reason = qwen_cuda_disabled_reason() if device == "cpu" else ""

        def load_model(target_device: str, target_dtype):
            return Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=target_dtype,
                device_map=target_device,
                max_inference_batch_size=1,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                forced_aligner=aligner_path,
                forced_aligner_kwargs={"dtype": target_dtype, "device_map": target_device},
            )

        if self._model is None or self._model_key != model_key:
            self._model, device, dtype, load_fallback = _load_qwen_with_cpu_fallback(
                load_model, torch=torch, device=device, dtype=dtype, progress=progress,
            )
            fallback_reason = load_fallback or fallback_reason
            self._model_key = (model_path, device)
        model = self._model
        samples = results = record = part_samples = None
        tokens: list[ASRToken] = []
        try:
            samples, sample_rate = decode_audio_mono(
                path, 16000, start_ms=options.clip_start_ms, end_ms=options.clip_end_ms,
            )
            language = QWEN_ASR_LANGUAGE_NAMES.get((options.language or "").casefold(), options.language)
            part_ranges = _qwen_vad_subchunk_ranges(samples, sample_rate, options)
            part_total = max(1, len(part_ranges))
            for part_index, (sample_start, sample_end) in enumerate(part_ranges):
                part_samples = samples[sample_start:sample_end]
                offset_ms = round(sample_start * 1000 / sample_rate)
                duration_ms = round((sample_end - sample_start) * 1000 / sample_rate)
                if progress and part_total > 1:
                    progress(
                        part_index / part_total,
                        f"Qwen 安全子块 {part_index + 1}/{part_total}",
                    )
                try:
                    with torch.inference_mode():
                        results = model.transcribe(
                            audio=(part_samples, sample_rate),
                            language=language,
                            return_time_stamps=True,
                        )
                except Exception as exc:
                    if device.startswith("cuda") and is_inference_out_of_memory(exc):
                        raise InferenceMemoryPressureError(
                            "Qwen GPU 推理显存不足；已完成的识别块仍会保留。"
                            f"失败的安全子块：{part_index + 1}/{part_total}。"
                        ) from exc
                    if not device.startswith("cuda") or not _is_fatal_torch_cuda_error(exc):
                        raise
                    fallback_reason = _disable_qwen_cuda(exc)
                    if progress:
                        progress(
                            -1.0,
                            "Qwen CUDA 推理失败，正在用 CPU float32 重试当前安全子块；"
                            f"本次运行不再重试 GPU · {fallback_reason}",
                        )
                    exc.__traceback__ = None
                    self._model = None
                    self._model_key = None
                    model = None
                    gc.collect()
                    device, dtype = "cpu", torch.float32
                    self._model = load_model(device, dtype)
                    self._model_key = (model_path, device)
                    model = self._model
                    with torch.inference_mode():
                        results = model.transcribe(
                            audio=(part_samples, sample_rate),
                            language=language,
                            return_time_stamps=True,
                        )
                record = results[0]
                part_had_timestamps = False
                for item in getattr(record, "time_stamps", None) or []:
                    text = str(getattr(item, "text", ""))
                    start = getattr(item, "start_time", None)
                    end = getattr(item, "end_time", None)
                    if start is None or end is None:
                        continue
                    part_had_timestamps = True
                    tokens.append(ASRToken(
                        None, chapter_id, len(tokens), text,
                        offset_ms + _qwen_timestamp_value(start),
                        offset_ms + _qwen_timestamp_value(end),
                        0.8,
                    ))
                if not part_had_timestamps and getattr(record, "text", ""):
                    tokens.append(ASRToken(
                        None, chapter_id, len(tokens), str(record.text),
                        offset_ms, offset_ms + duration_ms, 0.45,
                    ))
                record = None
                results = None
                part_samples = None
        finally:
            # Some qwen-asr result objects retain tensors through their timestamp
            # records. Break those references before the next book chunk starts.
            record = None
            results = None
            part_samples = None
            samples = None
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
            fallback_reason,
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

    def release_temporary_memory(self, *, aggressive: bool = False) -> None:
        if aggressive:
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None:
                try:
                    if torch.cuda.is_available() and torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

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
        fallback_reason = qwen_cuda_disabled_reason() if device == "cpu" else ""

        def load_model(target_device: str, target_dtype):
            return QwenAlignerModel.from_pretrained(
                model_path, dtype=target_dtype, device_map=target_device,
            )

        if self._model is None or self._model_key != model_key:
            self._model, device, dtype, load_fallback = _load_qwen_with_cpu_fallback(
                load_model, torch=torch, device=device, dtype=dtype, progress=progress,
            )
            fallback_reason = load_fallback or fallback_reason
            self._model_key = (model_path, device, str(dtype))
        model = self._model
        samples = results = None
        try:
            samples, sample_rate = decode_audio_mono(
                path, 16000, start_ms=options.clip_start_ms, end_ms=options.clip_end_ms,
            )
            language_name = QWEN_ASR_LANGUAGE_NAMES.get(language.casefold(), language)
            try:
                with torch.inference_mode():
                    results = model.align(audio=(samples, sample_rate), text=text, language=language_name)
            except Exception as exc:
                if not device.startswith("cuda") or not _is_fatal_torch_cuda_error(exc):
                    raise
                fallback_reason = _disable_qwen_cuda(exc)
                if progress:
                    progress(
                        -1.0,
                        "Qwen ForcedAligner CUDA 失败，正在用 CPU float32 重试；"
                        f"本次运行不再重试 GPU · {fallback_reason}",
                    )
                exc.__traceback__ = None
                self._model = None
                self._model_key = None
                model = None
                gc.collect()
                device, dtype = "cpu", torch.float32
                self._model = load_model(device, dtype)
                self._model_key = (model_path, device, str(dtype))
                model = self._model
                with torch.inference_mode():
                    results = model.align(
                        audio=(samples, sample_rate), text=text, language=language_name,
                    )
            tokens: list[ASRToken] = []
            for item in results[0]:
                tokens.append(ASRToken(
                    None, chapter_id, len(tokens), str(item.text),
                    _qwen_timestamp_value(item.start_time), _qwen_timestamp_value(item.end_time), 0.9,
                ))
        finally:
            results = None
            samples = None
        device_name = ""
        if device.startswith("cuda"):
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        self.last_device_info = InferenceDeviceInfo(
            "qwen3-forced-aligner", "Qwen3-ForcedAligner-0.6B", options.device,
            "cuda" if device.startswith("cuda") else "cpu", 0 if device.startswith("cuda") else None,
            device_name, "bfloat16" if device.startswith("cuda") else "float32",
            fallback_reason,
        )
        return tokens


class WhisperXTranscriber:
    """Optional full-chapter precise alignment backend."""

    def __init__(self) -> None:
        self.last_device_info = InferenceDeviceInfo("whisperx", "")
        self._model = None
        self._model_key: tuple[str, str, str] | None = None
        self._align_model = None
        self._align_metadata = None
        self._align_key: tuple[str, str] | None = None

    def release_temporary_memory(self, *, aggressive: bool = False) -> None:
        if aggressive:
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None:
                try:
                    if torch.cuda.is_available() and torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                except (AttributeError, RuntimeError):
                    pass

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
            raise BackendUnavailableError(
                "WhisperX 运行库缺失或未能加载；请在“选项 → 运行时组件”中安装后重启"
            ) from exc
        device = options.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if progress:
            progress(-1.0, "加载 WhisperX")
        if options.clip_start_ms or options.clip_end_ms is not None:
            audio, _sample_rate = decode_audio_mono(
                path, 16000, start_ms=options.clip_start_ms, end_ms=options.clip_end_ms,
            )
        else:
            audio = whisperx.load_audio(str(path))
        compute_type = "float16" if device == "cuda" else "int8"
        model_key = (options.model, device, compute_type)
        if self._model is None or self._model_key != model_key:
            self._model = whisperx.load_model(options.model, device, compute_type=compute_type)
            self._model_key = model_key
        result = aligned = None
        try:
            result = self._model.transcribe(audio, language=options.language)
            language = result.get("language") or options.language
            align_key = (str(language), device)
            if self._align_model is None or self._align_key != align_key:
                self._align_model, self._align_metadata = whisperx.load_align_model(
                    language_code=language, device=device,
                )
                self._align_key = align_key
            aligned = whisperx.align(
                result["segments"], self._align_model, self._align_metadata, audio, device,
            )
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
        finally:
            aligned = None
            result = None
            audio = None
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

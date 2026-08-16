from __future__ import annotations

import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audioalign.core.asr import (
    ASROptions,
    FasterWhisperTranscriber,
    InferenceMemoryPressureError,
    QWEN_MAX_NEW_TOKENS,
    Qwen3ASRTranscriber,
    _bounded_forced_alignment_tokens,
    _ensure_qwen_model,
    _qwen_device,
    plan_audio_chunks,
    plan_recognition_chunks,
    recognition_cache_key,
)
from audioalign.core.models import ASRBackendId, BoundaryCandidate


class _Info:
    duration = 1.0


class _Word:
    word = "hello"
    start = 0.1
    end = 0.5
    probability = 0.9


class _Segment:
    words = [_Word()]
    end = 0.5
    text = "hello"


class _AlignedItem:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start_time = start
        self.end_time = end


class ASRTests(unittest.TestCase):
    def test_forced_alignment_tail_is_bounded_to_requested_m4b_clip(self) -> None:
        tokens = _bounded_forced_alignment_tokens([
            _AlignedItem("last", 29.4, 30.24),
            _AlignedItem("outside", 30.0, 30.3),
            _AlignedItem("zero", 11.2, 11.2),
        ], 7, 29_808)
        self.assertEqual(1, len(tokens))
        self.assertEqual((29_400, 29_808), (tokens[0].start_ms, tokens[0].end_ms))

    def test_qwen_splits_over_limit_outer_chunk_at_vad_silence(self) -> None:
        calls: list[int] = []
        load_options: list[dict] = []

        class FakeQwenModel:
            @classmethod
            def from_pretrained(cls, _path, **kwargs):
                load_options.append(kwargs)
                return cls()

            def transcribe(self, *, audio, **_kwargs):
                samples, sample_rate = audio
                calls.append(len(samples))
                duration = len(samples) / sample_rate
                stamp = types.SimpleNamespace(text="part", start_time=0.0, end_time=duration)
                return [types.SimpleNamespace(text="part", time_stamps=[stamp])]

        fake_qwen = types.ModuleType("qwen_asr")
        fake_qwen.Qwen3ASRModel = FakeQwenModel
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.bfloat16 = object()
        fake_torch.inference_mode = nullcontext
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            is_initialized=lambda: False,
        )
        with (
            patch.dict(sys.modules, {"qwen_asr": fake_qwen, "torch": fake_torch}),
            patch("audioalign.core.asr._ensure_qwen_model", return_value="model"),
            patch(
                "audioalign.core.asr.decode_audio_mono",
                return_value=(np.zeros(183 * 16_000, dtype=np.float32), 16_000),
            ),
            patch(
                "audioalign.core.asr.detect_silence_candidates",
                return_value=[BoundaryCandidate(150_000, 0.95, "vad-silence")],
            ),
        ):
            tokens = Qwen3ASRTranscriber().transcribe(
                "audio.wav", 1, ASROptions(model="Qwen3-ASR-0.6B"),
            )

        self.assertEqual([150 * 16_000, 33 * 16_000], calls)
        self.assertEqual(QWEN_MAX_NEW_TOKENS, load_options[0]["max_new_tokens"])
        self.assertEqual([0, 150_000], [token.start_ms for token in tokens])
        self.assertEqual(183_000, tokens[-1].end_ms)

    def test_qwen_fatal_cuda_load_error_falls_back_to_cpu_for_the_task(self) -> None:
        load_devices: list[str] = []

        class FakeQwenModel:
            @classmethod
            def from_pretrained(cls, _path, **kwargs):
                device = kwargs["device_map"]
                load_devices.append(device)
                if str(device).startswith("cuda"):
                    raise RuntimeError("CUDA error: an illegal memory access was encountered")
                return cls()

            def transcribe(self, **_kwargs):
                stamp = types.SimpleNamespace(text="hello", start_time=0.1, end_time=0.5)
                return [types.SimpleNamespace(text="hello", time_stamps=[stamp])]

        fake_qwen = types.ModuleType("qwen_asr")
        fake_qwen.Qwen3ASRModel = FakeQwenModel
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.bfloat16 = object()
        fake_torch.inference_mode = nullcontext
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_device_name=lambda _index: "Fake GPU",
        )
        messages: list[str] = []
        with (
            patch.dict(sys.modules, {"qwen_asr": fake_qwen, "torch": fake_torch}),
            patch("audioalign.core.asr._qwen_cuda_disabled_reason", ""),
            patch("audioalign.core.asr._ensure_qwen_model", return_value="model"),
            patch(
                "audioalign.core.asr.decode_audio_mono",
                return_value=(np.zeros(1600, dtype=np.float32), 16000),
            ),
        ):
            transcriber = Qwen3ASRTranscriber()
            tokens = transcriber.transcribe(
                "audio.wav",
                1,
                ASROptions(model="Qwen3-ASR-0.6B"),
                lambda _value, message: messages.append(message),
            )
            self.assertNotEqual("", __import__("audioalign.core.asr", fromlist=[""]).qwen_cuda_disabled_reason())

        self.assertEqual(["cuda:0", "cpu"], load_devices)
        self.assertEqual("hello", tokens[0].text)
        self.assertEqual("cpu", transcriber.last_device_info.actual_device)
        self.assertIn("illegal memory access", transcriber.last_device_info.fallback_reason)
        self.assertTrue(any("CPU float32" in message for message in messages))

    def test_qwen_cuda_oom_stops_without_loading_cpu_copy(self) -> None:
        load_devices: list[str] = []

        class FakeQwenModel:
            @classmethod
            def from_pretrained(cls, _path, **kwargs):
                load_devices.append(kwargs["device_map"])
                return cls()

            def transcribe(self, **_kwargs):
                raise RuntimeError("CUDA out of memory. Tried to allocate 10.83 GiB")

        fake_qwen = types.ModuleType("qwen_asr")
        fake_qwen.Qwen3ASRModel = FakeQwenModel
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.bfloat16 = object()
        fake_torch.inference_mode = nullcontext
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_device_name=lambda _index: "Fake GPU",
        )
        with (
            patch.dict(sys.modules, {"qwen_asr": fake_qwen, "torch": fake_torch}),
            patch("audioalign.core.asr._qwen_cuda_disabled_reason", ""),
            patch("audioalign.core.asr._ensure_qwen_model", return_value="model"),
            patch(
                "audioalign.core.asr.decode_audio_mono",
                return_value=(np.zeros(1600, dtype=np.float32), 16_000),
            ),
        ):
            with self.assertRaises(InferenceMemoryPressureError):
                Qwen3ASRTranscriber().transcribe(
                    "audio.wav", 1, ASROptions(model="Qwen3-ASR-0.6B"),
                )

        self.assertEqual(["cuda:0"], load_devices)

    def test_long_audio_chunk_plan_uses_pause_and_bounded_overlap(self) -> None:
        chunks = plan_audio_chunks(
            0, 400_000, [BoundaryCandidate(118_000, 0.9), BoundaryCandidate(241_000, 0.8)],
        )
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(0, chunks[0].core_start_ms)
        self.assertEqual(400_000, chunks[-1].core_end_ms)
        self.assertTrue(all(chunk.end_ms - chunk.start_ms <= 183_000 for chunk in chunks))
        self.assertEqual(chunks[0].core_end_ms - 1_500, chunks[1].start_ms)

    def test_new_qwen_plan_accounts_for_overlap_and_avoids_short_tail(self) -> None:
        options = ASROptions(backend=ASRBackendId.QWEN3_ASR)
        chunks = plan_recognition_chunks(
            0, 400_000,
            [BoundaryCandidate(147_450, 0.95), BoundaryCandidate(299_000, 0.9)],
            options,
        )
        self.assertEqual(147_450, chunks[0].core_end_ms)
        self.assertTrue(all(chunk.end_ms - chunk.start_ms <= 180_000 for chunk in chunks))
        self.assertGreaterEqual(chunks[-1].core_end_ms - chunks[-1].core_start_ms, 30_000)

    def test_resumed_qwen_run_keeps_original_chunk_plan(self) -> None:
        options = ASROptions(backend=ASRBackendId.QWEN3_ASR)
        resumed = plan_recognition_chunks(
            0, 400_000, [], options, preserve_existing_plan=True,
        )
        original = plan_audio_chunks(0, 400_000, [])
        self.assertEqual(original, resumed)

    def test_cache_key_changes_with_recognition_parameters(self) -> None:
        options = ASROptions(model="small", vad_threshold=0.5)
        first, _ = recognition_cache_key("audio-a", options)
        options.vad_threshold = 0.7
        second, _ = recognition_cache_key("audio-a", options)
        self.assertNotEqual(first, second)

    def test_missing_cuda_dll_falls_back_to_cpu_int8(self) -> None:
        calls: list[tuple[str, str]] = []

        class FakeModel:
            def __init__(self, _model, *, device, compute_type, **_kwargs):
                self.device = device
                calls.append((device, compute_type))

            def transcribe(self, *_args, **_kwargs):
                if self.device == "cuda":
                    def failed():
                        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                        yield
                    return failed(), _Info()
                return iter([_Segment()]), _Info()

        fake_module = types.ModuleType("faster_whisper")
        fake_module.WhisperModel = FakeModel
        messages: list[str] = []
        with patch.dict(sys.modules, {"faster_whisper": fake_module}), patch(
            "audioalign.core.asr.cuda_runtime_available", return_value=True
        ):
            tokens = FasterWhisperTranscriber().transcribe(
                "audio.opus",
                1,
                ASROptions(device="auto", compute_type="auto"),
                lambda _value, message: messages.append(message),
            )
        self.assertEqual([("cuda", "float16"), ("cpu", "int8")], calls)
        self.assertEqual("hello", tokens[0].text)
        self.assertTrue(any("回退到 CPU INT8" in message for message in messages))

    def test_qwen_cpu_accepts_long_ranges(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.bfloat16 = object()
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            _torch, device, dtype = _qwen_device(ASROptions(clip_start_ms=0, clip_end_ms=180_000))
        self.assertEqual("cpu", device)
        self.assertIs(fake_torch.float32, dtype)

    def test_qwen_download_falls_back_to_modelscope(self) -> None:
        messages: list[str] = []
        huggingface = types.ModuleType("huggingface_hub")
        modelscope = types.ModuleType("modelscope")

        def failed_huggingface(**_kwargs):
            raise TimeoutError("huggingface.co timed out")

        def successful_modelscope(_model_id, *, local_dir):
            directory = Path(local_dir)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"weights")
            return local_dir

        huggingface.snapshot_download = failed_huggingface
        modelscope.snapshot_download = successful_modelscope
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            sys.modules, {"huggingface_hub": huggingface, "modelscope": modelscope},
        ):
            result = _ensure_qwen_model(folder, "Qwen3-ASR-0.6B", lambda _value, text: messages.append(text))
            self.assertTrue((Path(result) / "config.json").is_file())
        self.assertTrue(any("切换 ModelScope" in message for message in messages))
        self.assertTrue(any("下载完成 · ModelScope" in message for message in messages))


if __name__ == "__main__":
    unittest.main()

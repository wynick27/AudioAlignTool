from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from audioalign.core.asr import (
    ASROptions,
    FasterWhisperTranscriber,
    _ensure_qwen_model,
    _qwen_device,
    plan_audio_chunks,
    recognition_cache_key,
)
from audioalign.core.models import BoundaryCandidate


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


class ASRTests(unittest.TestCase):
    def test_long_audio_chunk_plan_uses_pause_and_bounded_overlap(self) -> None:
        chunks = plan_audio_chunks(
            0, 400_000, [BoundaryCandidate(118_000, 0.9), BoundaryCandidate(241_000, 0.8)],
        )
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(0, chunks[0].core_start_ms)
        self.assertEqual(400_000, chunks[-1].core_end_ms)
        self.assertTrue(all(chunk.end_ms - chunk.start_ms <= 183_000 for chunk in chunks))
        self.assertEqual(chunks[0].core_end_ms - 1_500, chunks[1].start_ms)

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

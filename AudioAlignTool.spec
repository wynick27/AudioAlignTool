# Run through build-portable.ps1 so local and CI builds use identical options.
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, copy_metadata

datas, binaries, hiddenimports = [], [], []
for package in (
    "av",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "huggingface_hub",
    "pip",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# Silero VAD only needs ONNX Runtime's inference API.  collect_all() also tries
# importing quantization/training helpers (and their optional `onnx` package),
# producing warnings and collecting modules the editor never uses.
datas += collect_data_files("onnxruntime")
binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += [
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_pybind11_state",
]

# Styled source-book reader dependencies. Explicit entries make collection of
# the WebEngine helper/resources deterministic in local and CI builds.
hiddenimports += [
    "markdown_it",
    "pyqtgraph",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
]

for distribution in (
    "faster-whisper",
    "ctranslate2",
    "huggingface-hub",
    "tokenizers",
):
    datas += copy_metadata(distribution)

include_qwen = os.environ.get("AAT_INCLUDE_QWEN", "").strip() == "1"
if include_qwen:
    package_datas, package_binaries, package_hidden = collect_all("qwen_asr")
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
    # Our fallback imports only modelscope.snapshot_download.  Static analysis
    # follows that path; collecting every trainer/parallel plugin pulled in
    # optional `addict` and a large unrelated training stack.
    hiddenimports += ["modelscope", "modelscope.hub.snapshot_download"]
    for distribution in ("qwen-asr", "modelscope"):
        datas += copy_metadata(distribution)

excludes = [] if include_qwen else [
    "qwen_asr",
    "modelscope",
    "torch",
    "transformers",
    "accelerate",
    "scipy",
    "sklearn",
    "pandas",
    "numba",
    "llvmlite",
]

a = Analysis(
    ["src/audioalign/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AudioAlignTool", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AudioAlignTool")

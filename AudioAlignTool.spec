# Run through build-portable.ps1 so local and CI builds use identical options.
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

excludes = [
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
    # pip's vendored distlib enumerates its Windows launcher resources through
    # the package loader.  Keeping pip outside PYZ gives it a standard file
    # loader and leaves the bundled --runtime-pip entry point self-contained.
    module_collection_mode={"pip": "py"},
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AudioAlignTool", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AudioAlignTool")

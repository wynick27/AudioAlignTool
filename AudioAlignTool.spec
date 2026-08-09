# Run through build-portable.ps1 so local and CI builds use identical options.
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas, binaries, hiddenimports = [], [], []
for package in (
    "av",
    "pyqtgraph",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "onnxruntime",
    "huggingface_hub",
    "qwen_asr",
    "modelscope",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

for distribution in (
    "faster-whisper",
    "ctranslate2",
    "huggingface-hub",
    "tokenizers",
    "qwen-asr",
    "modelscope",
):
    datas += copy_metadata(distribution)

a = Analysis(
    ["src/audioalign/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AudioAlignTool", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AudioAlignTool")

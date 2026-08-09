# AudioAlignTool

AudioAlignTool 是用于电子书文本与有声书时间轴对齐的 Windows 桌面编辑器。它也支持完全没有原文的工作流：直接导入音频或 M4B，由 Whisper 生成文本与句段，再人工校正。

## 当前实现

- PySide6 桌面界面；字幕表格与“文章频谱”双视图。
- Subtitle Edit 风格的时频谱主编辑器：框选时间、拖动句段边界、静音吸附、播放/循环选区、水平滚动和缩放、频率轴缩放、总览跳转、右键编辑。
- 文章按窗口宽度自然换行，每个视觉文字行下方显示其时间范围的频谱；有 ASR 锚点时使用锚点，没有时明确标记为估算时间。
- TXT/EPUB，以及 MP3、M4A、M4B、AAC、WAV、FLAC、OGG、Opus。M4B 内嵌章节作为同一资源中的时间切片参与配对。
- 章节—音频配对管理器：重新指定、取消、整体 ±1 偏移、自动匹配和一次事务应用。
- 句子—时间关系编辑：选区绑定、平均分配、开始/结束、拆分、合并、清除和删除；重叠以红色显示，不会静默移动其他句段。
- faster-whisper 是基础识别后对齐后端；WhisperX 会在 Whisper 识别后再使用声学对齐模型细化时间。CPU 默认 INT8，CTranslate2 检测到 CUDA 时自动使用 GPU。
- 静音检测为时间边界候选，Whisper 用于确定文本位置；人工锁定内容不会被自动流程覆盖。
- HTML、SRT、WebVTT 和 schema v2 JSON 导出。

## 项目与程序数据

源码运行时，仓库根目录就是程序目录；打包后，可执行文件所在目录就是程序目录。程序必须位于可写位置，不会回退到 LocalAppData。

```text
AudioAlignTool/
  projects/<项目名>/
  .work/<压缩项目名>/
  models/<Whisper 模型名>/
  logs/
  settings.json
```

项目目录名就是经过 Windows 文件名校验的项目名，不使用随机 UUID。同名项目不会被覆盖。

项目格式只支持 schema v2：

```text
manifest.json
project.sqlite3
source/
media/
cache/
```

`.aatproj` 是 ZIP64 单文件项目；编辑时解压到程序目录下 `.work/`，保存时使用临时文件原子替换。普通项目文件夹直接事务化保存。

## 安装与启动

要求 Python 3.14。首次准备源码环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

之后始终使用项目自己的解释器启动：

```powershell
.\start.bat
```

Whisper 模型在首次确认下载后写入 `models/<模型名>/`，之后可离线运行。WhisperX 工作流另行安装可选组件：

```powershell
.\.venv\Scripts\python.exe -m pip install whisperx
```

## 基本流程

1. 新建以项目名命名的项目，导入 TXT/EPUB 和音频；也可只导入音频。
2. 在“章节与音频配对”中确认文件或 M4B 内嵌章节的对应关系。
3. 选择对齐方式、模型和语言。可选 faster-whisper/WhisperX/Qwen3-ASR 的识别后对齐，或 Qwen ForcedAligner 的已知文本强制对齐。
4. 检测静音，在频谱上框选、试听并拖动句段边界；按 Shift 拖动可临时关闭静音吸附。
5. 在字幕表格或文章频谱视图中检查内容，保存项目并导出所需格式。

音频图操作：普通滚轮水平滚动，Ctrl+滚轮以鼠标位置为中心缩放时间轴，Alt+滚轮按播放跟随中心缩放，Shift+滚轮缩放波形振幅或频谱频率轴。Escape 清除选区，Delete 删除所选句段，Enter 将选区绑定到当前句。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Windows 便携版打包

项目只发布免安装便携版，不生成安装器。请在 Windows 上使用 Python 3.14：

```powershell
.\build-portable.ps1 -Clean
```

脚本会创建或复用程序目录内的 `.venv`，安装锁定依赖，运行测试，调用
PyInstaller 生成 onedir 目录，最后输出：

```text
artifacts/AudioAlignTool-<版本>-Windows-x64-portable.zip
artifacts/SHA256SUMS.txt
```

解压 ZIP 后直接运行 `AudioAlignTool.exe`。不能只复制 EXE，必须保留同目录下的
`_internal`。项目、模型、日志和设置都会写在解压后的程序目录，因此应解压到用户
有写权限的位置。

常用参数：

```powershell
# 使用指定的 Python 3.14
.\build-portable.ps1 -Python C:\Python314\python.exe -Clean

# 已经安装依赖时跳过安装；仅在确认环境完整时使用
.\build-portable.ps1 -SkipInstall

# 临时跳过测试
.\build-portable.ps1 -SkipTests
```

GitHub Actions 配置位于 `.github/workflows/windows-portable.yml`。推送到 `main`、
创建针对 `main` 的拉取请求、推送 `v*` 标签或手动触发时都会构建，并上传 ZIP 与
SHA-256 校验文件。CI 与本机均调用同一个 `build-portable.ps1`，不会调用 Inno Setup。

## 识别缓存与 CUDA

- faster-whisper 与 Qwen3-ASR 使用统一的分块识别流程；默认目标块为 120 秒，优先在 VAD 停顿处分块。
- 完成的识别块保存在项目数据库中。修改原文只会重新匹配，不会重新运行 ASR；可在界面中强制刷新或清除当前模型缓存。
- Windows 启动时会注册 Python 环境中 `nvidia/cublas/bin` 与 `nvidia/cudnn/bin`，不会永久修改系统 PATH。
- 模型状态栏在实际推理后显示真正使用的 CPU/GPU、计算类型以及 CUDA 回退原因。
- Qwen3-ForcedAligner 用于已知文本的句子或选区；GPU 优先，CPU 允许继续但会明确提示速度很慢，单次模型输入硬上限为 240 秒。
- Qwen 强制对齐入口集中在一个下拉菜单：可处理当前句、连续所选句子与明确音频选区、从当前句/播放头向后，以及整个章节。音频存在片头或文本版本差异时，优先使用“所选句子 ↔ 音频选区”建立准确局部范围，或用“从当前句/时间向后”跳过不一致的开头。
- Qwen 没有可用 GPU 时不会禁止任务，而是显示 CPU/float32 状态并警告可能很慢，由用户确认是否继续。
- Qwen 模型优先从 Hugging Face 下载；连接或下载失败时自动切换到 ModelScope。只有配置和全部权重完整时才会把模型标记为可用，未完成文件可供下次续传。

首次启动默认播放速度为 `1.00×`；以后恢复上次使用的倍率。若开启“启动时使用 1.0×”，则每次启动都忽略历史倍率。

# BoneAgeVision

Deep-learning bone age assessment with YOLOv5 localization, ResNet stage classification, and RUS-CHN scoring.  
基于 YOLOv5 骨骼定位、ResNet 骨骺分级与 RUS-CHN 计分的深度学习骨龄评估系统。

> This project is intended for research, teaching, and engineering demonstration. It is not a clinical diagnostic device.  
> 本项目仅用于科研、教学与工程演示，不构成医疗诊断或临床决策依据。

## English

### Overview

BoneAgeVision provides an end-to-end desktop workflow for pediatric hand X-ray bone-age assessment. A custom YOLOv5 detector localizes the required skeletal regions, dedicated ResNet classifiers estimate maturity stages, and the RUS-CHN score table converts the 13 key regions into a total score and estimated bone age.

The repository has been reorganized as a standard Python `src`-layout project. Model checkpoints are isolated under `weights/`, inference logic is separated from GUI code, invalid or incomplete detections fail explicitly instead of silently producing default scores, and large model files are prepared for Git LFS.

### Pipeline

```text
Hand X-ray
   │
   ▼
YOLOv5 detector (best.pt)
   │  13 required anatomical regions
   ▼
Bone-specific ResNet classifiers
   │  maturity stage per region
   ▼
RUS-CHN score lookup
   │
   ▼
Total score → polynomial bone-age estimation
   │
   ▼
Annotated image + structured report
```

### Repository structure

```text
BoneAgeVision/
├── .github/workflows/ci.yml
├── .gitattributes
├── .gitignore
├── README.md
├── pyproject.toml
├── src/
│   └── bone_age_vision/
│       ├── __init__.py
│       ├── __main__.py
│       ├── paths.py
│       ├── core/
│       │   ├── analyzer.py
│       │   ├── classifier.py
│       │   ├── image_processing.py
│       │   ├── resnet.py
│       │   └── scoring.py
│       └── gui/
│           ├── main_window.py
│           ├── styles/theme.py
│           └── widgets/
│               ├── image_viewer.py
│               └── result_display.py
├── tests/
│   └── test_scoring.py
└── weights/
    ├── best.pt
    ├── Resnet_*.pt
    └── README.md
```

### Requirements

- Python 3.10+
- Git LFS for versioning model checkpoints
- Internet access on the first detector initialization so PyTorch Hub can fetch YOLOv5 commit `4add2aff6e3d`
- NVIDIA CUDA is optional; CUDA is selected automatically when available
- Tkinter must be available in the local Python installation

### Installation

Clone the repository and initialize Git LFS:

```bash
git clone <your-repository-url>
cd BoneAgeVision
git lfs install
git lfs pull
```

Create a virtual environment and install the project in editable mode. Editable installation is intentional because the large checkpoints remain in the repository-level `weights/` directory.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

On Linux, install Tkinter through the operating system if needed, for example `sudo apt install python3-tk` on Debian/Ubuntu.

### Run

```bash
bone-age-vision
```

or:

```bash
python -m bone_age_vision
```

Select a hand X-ray image, choose the subject sex, and start the analysis. The first run may take longer because the pinned YOLOv5 source is fetched and cached by PyTorch Hub.

### Engineering decisions

The refactor intentionally removes compatibility code and obsolete artifacts instead of preserving the original layout. The application now has a single dependency definition in `pyproject.toml`; IDE metadata, Python bytecode caches, and interrupted download files are excluded. The broken duplicate ResNet definition was replaced by one checkpoint-compatible model whose classifier input dimension matches the supplied weights (`2048`). The actual checkpoint filename convention (`Resnet_*.pt`) is used consistently on case-sensitive systems.

The detector no longer falls back to an unrelated pretrained COCO model. Missing or corrupt checkpoints stop execution with a clear error. Likewise, incomplete localization no longer assigns zero scores to undetected bones, because that would yield a numerically valid but medically meaningless result.

GUI inference runs in a worker thread so model initialization and inference do not freeze the Tkinter event loop. The core inference, scoring, model architecture, image utilities, GUI styling, and widgets remain separate modules with narrowly defined responsibilities.

### Model files and GitHub

The supplied checkpoints are large. `.gitattributes` configures `weights/*.pt` for Git LFS. Before the first push:

```bash
git lfs install
git add .
git commit -m "Initial BoneAgeVision release"
git push -u origin main
```

Do not upload the checkpoint files with GitHub's browser file uploader; use Git with Git LFS.

### Quality checks

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

GitHub Actions runs lightweight linting and scoring tests on pushes and pull requests without downloading the large model checkpoints.

---

## 中文

### 项目简介

BoneAgeVision 提供一套面向儿童手部 X 光片的端到端骨龄评估流程。系统首先使用自定义 YOLOv5 模型定位关键骨骼区域，再通过不同骨骼对应的 ResNet 分类器判断骨骺成熟分级，最后依据 RUS-CHN 评分表计算 13 个关键区域的总分，并通过多项式模型估算骨龄。

本仓库已按照标准 Python `src` 布局重新整理。模型权重统一放在 `weights/`，核心推理逻辑与 GUI 解耦；检测不完整或模型缺失时会明确报错，不再通过默认 0 分生成表面上“正常”的错误结果；大模型文件则使用 Git LFS 管理。

### 处理流程

```text
手部 X 光片
   │
   ▼
YOLOv5 目标检测（best.pt）
   │  定位 13 个必需骨骼区域
   ▼
骨骼专用 ResNet 分级模型
   │  输出各区域成熟分级
   ▼
RUS-CHN 评分表
   │
   ▼
总分 → 多项式骨龄估算
   │
   ▼
标注图像 + 检测报告
```

### 环境要求

- Python 3.10 及以上
- Git LFS，用于管理模型权重
- 首次初始化检测器时需要联网，由 PyTorch Hub 获取固定 YOLOv5 提交 `4add2aff6e3d`
- CUDA 非必需；检测到可用 CUDA 时自动使用 GPU，否则使用 CPU
- 本机 Python 需要具备 Tkinter

### 安装

克隆仓库并初始化 Git LFS：

```bash
git clone <你的仓库地址>
cd BoneAgeVision
git lfs install
git lfs pull
```

建议使用虚拟环境，并采用 editable 模式安装。这样模型权重可以继续保留在仓库根目录的 `weights/` 中，不需要塞进 Python 安装包。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Linux 若缺少 Tkinter，可通过系统包管理器安装，例如 Debian/Ubuntu 使用 `sudo apt install python3-tk`。

### 运行

```bash
bone-age-vision
```

或：

```bash
python -m bone_age_vision
```

选择手部 X 光图片、设置检测对象性别，然后执行分析。首次运行时 PyTorch Hub 需要拉取并缓存固定提交的 YOLOv5 源码，因此启动时间会长于后续运行。

### 本次工程化重构

本次重构不保留旧目录兼容层，而是直接删除过时结构和无效文件。依赖声明统一收敛到 `pyproject.toml`，不再同时维护 `setup.py` 与 `requirements.txt`；`.idea`、`__pycache__` 和未完成下载的 `.partial` 文件均被清除。

原始 `resnet.py` 存在两个同名 `ResNet` 类，后一个类还依赖未定义的 `one_hot_dic_grade`，代码实际无法正常初始化。同时原代码将全连接层输入写为 `512 × 4 × 4`，而现有权重的真实尺寸为 `1024 × 2048`，因此本项目已按照权重 state dict 恢复为 `512 × 2 × 2 = 2048` 的正确输入维度。原代码查找 `ResNet_*.pt`，实际权重名称则为 `Resnet_*.pt`，这一问题在 Linux 等大小写敏感系统上会导致全部分类模型加载失败，现已统一修正。

检测器只使用项目提供的 `best.pt`，不再在加载失败后悄悄退回与任务无关的 COCO 预训练 YOLOv5s。分类模型缺失、损坏或关键骨骼检测不完整时同样直接报错，避免缺失骨骼被默认赋 0 分后继续计算骨龄。

GUI 推理被放入后台工作线程，避免模型初始化和推理过程阻塞 Tkinter 主事件循环。检测、分级、评分、图像处理、界面样式和组件分别维护，模块职责更加明确。

### GitHub 大文件管理

模型权重总量较大，仓库已通过 `.gitattributes` 将 `weights/*.pt` 配置为 Git LFS 文件。首次推送前执行：

```bash
git lfs install
git add .
git commit -m "Initial BoneAgeVision release"
git push -u origin main
```

不要直接使用 GitHub 网页上传这些权重文件，应使用 Git + Git LFS 推送。

### 代码质量检查

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

GitHub Actions 会在 push 和 pull request 时运行轻量级 lint 与评分单元测试，不需要下载或加载大模型权重。

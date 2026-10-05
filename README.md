# BoneAgeVision

Deep-learning bone age assessment with YOLOv5 localization, ResNet stage classification, and RUS-CHN scoring.  
基于 YOLOv5 骨骼定位、ResNet 骨骺分级与 RUS-CHN 计分的深度学习骨龄评估系统。

> For research, teaching, and engineering demonstration only. It is not a clinical diagnostic device.  
> 本项目仅用于科研、教学与工程演示，不构成医疗诊断或临床决策依据。

## English

### Overview

BoneAgeVision is a desktop pipeline for pediatric hand X-ray bone-age assessment. A custom YOLOv5 detector localizes the required skeletal regions, bone-specific ResNet classifiers estimate maturity stages, and the RUS-CHN scoring tables convert the 13 required regions into a total score and estimated bone age.

The repository follows a standard Python `src` layout. Detection, region selection, stage classification, scoring, model-asset validation, and GUI code are separated by responsibility. Invalid input, missing model files, incomplete localization, and invalid score indices fail explicitly instead of producing silent fallback results.

### Pipeline

```text
Hand X-ray
   │
   ▼
Custom YOLOv5 detector
   │
   ▼
13-region deterministic selector
   │
   ▼
Bone-specific ResNet classifiers
   │
   ▼
RUS-CHN score lookup
   │
   ▼
Total score → polynomial bone-age estimate
   │
   ▼
Annotated image + report
```

### Repository structure

```text
BoneAgeVision/
├── .github/workflows/ci.yml
├── .gitignore
├── README.md
├── pyproject.toml
├── src/bone_age_vision/
│   ├── __main__.py
│   ├── paths.py
│   ├── core/
│   │   ├── analyzer.py
│   │   ├── assets.py
│   │   ├── classifier.py
│   │   ├── detector.py
│   │   ├── domain.py
│   │   ├── image_processing.py
│   │   ├── regions.py
│   │   ├── resnet.py
│   │   └── scoring.py
│   └── gui/
│       ├── main_window.py
│       ├── styles/theme.py
│       └── widgets/
├── tests/
└── weights/
    └── README.md
```

### Model checkpoints

Trained checkpoints are intentionally **not distributed in this public repository**. This keeps the source repository lightweight and avoids publishing private model artifacts. Git ignores `weights/*.pt` to prevent accidental commits.

The application requires these local files:

```text
best.pt
Resnet_DIP.pt
Resnet_DIPFirst.pt
Resnet_MCP.pt
Resnet_MCPFirst.pt
Resnet_MIP.pt
Resnet_PIP.pt
Resnet_PIPFirst.pt
Resnet_Radius.pt
Resnet_Ulna.pt
```

You may place them in `./weights`, or keep them in any private directory and pass that path at runtime.

### Installation

Python 3.10+ is required. CUDA is optional; the application automatically uses CUDA when available and otherwise runs on CPU. Tkinter must be available in the local Python installation.

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

### Run

With checkpoints in `./weights`:

```bash
bone-age-vision
```

With checkpoints stored elsewhere:

```bash
bone-age-vision --weights /path/to/private/weights
```

The first detector initialization requires internet access because PyTorch Hub fetches the pinned YOLOv5 source revision `4add2aff6e3d`.

### Engineering design

The codebase intentionally avoids compatibility shims for the original project layout. Dependencies have one source of truth in `pyproject.toml`, generated and IDE artifacts are excluded, and the supplied ResNet architecture uses the checkpoint-compatible `2048`-feature classifier input.

The detector adapter converts YOLO tensors into framework-independent domain objects before anatomical selection. Region selection is therefore deterministic and independently testable. The analyzer only orchestrates detection, classification, scoring, and rendering. All ten required checkpoint files are validated together at startup so a missing-model error reports the complete problem instead of failing one file at a time.

The detector never falls back to an unrelated COCO model. Incomplete localization also stops the assessment rather than assigning zero scores to missing bones. GUI inference runs in a worker thread so model loading and inference do not block the Tkinter event loop.

### Quality checks

```bash
pip install -e ".[dev]"
python -m compileall -q src
ruff check src tests
pytest
```

GitHub Actions runs compilation, Ruff, and the lightweight test suite on every push to `main` and every pull request. ResNet forward-pass tests run automatically when PyTorch is available and otherwise skip cleanly in lightweight CI.

---

## 中文

### 项目简介

BoneAgeVision 是一套面向儿童手部 X 光片的骨龄评估桌面程序。系统使用自定义 YOLOv5 模型定位骨骼区域，根据固定解剖位置筛选 RUS-CHN 所需的 13 个区域，再通过骨骼专用 ResNet 分类器判断成熟分级，最终依据 RUS-CHN 评分表计算总分并估算骨龄。

项目采用标准 Python `src` 布局。目标检测、区域筛选、骨骼分级、评分、模型文件校验和 GUI 分别承担单一职责。输入无效、权重缺失、关键区域检测不完整或评分索引异常时均明确报错，不使用静默回退结果。

### 模型权重

训练权重**不在本公开仓库中发布**。这样可以保持源码仓库轻量，并避免将私有模型文件误提交到 GitHub。仓库已经通过 `.gitignore` 忽略 `weights/*.pt`。

运行时需要以下本地文件：

```text
best.pt
Resnet_DIP.pt
Resnet_DIPFirst.pt
Resnet_MCP.pt
Resnet_MCPFirst.pt
Resnet_MIP.pt
Resnet_PIP.pt
Resnet_PIPFirst.pt
Resnet_Radius.pt
Resnet_Ulna.pt
```

可以将它们放在项目根目录的 `weights/` 中，也可以保存在任意私有目录，通过 `--weights` 指定。

### 安装

要求 Python 3.10 及以上。CUDA 非必需；检测到 CUDA 时自动使用 GPU，否则使用 CPU。本机 Python 还需要具备 Tkinter。

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

### 运行

权重位于 `./weights` 时：

```bash
bone-age-vision
```

权重位于其他私有目录时：

```bash
bone-age-vision --weights /path/to/private/weights
```

首次初始化检测器时需要联网，PyTorch Hub 会拉取固定的 YOLOv5 源码提交 `4add2aff6e3d`。

### 软件工程重构

本项目不为旧目录结构保留兼容层。依赖声明统一由 `pyproject.toml` 管理，IDE 元数据、缓存、构建产物和模型权重均不会进入源码版本控制。ResNet 结构按照现有权重恢复为正确的 `2048` 维分类器输入，并统一使用实际的 `Resnet_*.pt` 文件命名。

YOLO 检测器已经从总流程中独立出来，检测输出先转换为与 PyTorch 无关的领域对象，再进入固定的 13 区域筛选逻辑，因此该部分可以脱离模型单独测试。`BoneAgeAnalyzer` 只负责流程编排。程序启动模型时会一次检查全部 10 个必需权重文件，避免逐个失败造成定位困难。

检测器不会在加载失败时退回与任务无关的 COCO 通用模型；关键骨骼检测不完整时也不会默认补 0 分继续计算。GUI 推理在后台工作线程执行，避免模型加载和推理阻塞 Tkinter 主事件循环。

### 质量检查

```bash
pip install -e ".[dev]"
python -m compileall -q src
ruff check src tests
pytest
```

GitHub Actions 会在每次推送到 `main` 以及 Pull Request 时执行编译检查、Ruff 和轻量单元测试。本机存在 PyTorch 时还会执行 ResNet 前向维度测试；轻量 CI 未安装 PyTorch 时该测试会自动跳过。

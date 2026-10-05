# BoneAgeVision

Deep-learning bone age assessment with YOLOv5 localization, ResNet stage classification, and RUS-CHN scoring.  
基于 YOLOv5 骨骼定位、ResNet 骨骺分级与 RUS-CHN 计分的深度学习骨龄评估系统。

> For research, teaching, and engineering demonstration only. It is not a clinical diagnostic device.  
> 本项目仅用于科研、教学与工程演示，不构成医疗诊断或临床决策依据。

## English

### Overview

BoneAgeVision is a desktop application for hand X-ray bone-age assessment. It combines a custom YOLOv5 detector, bone-specific ResNet classifiers, and the RUS-CHN scoring method to produce skeletal maturity stages, a total score, an estimated bone age, and an annotated result image.

### Pipeline

```text
Hand X-ray
   │
   ▼
YOLOv5 bone localization
   │
   ▼
13 required RUS-CHN regions
   │
   ▼
Bone-specific ResNet classification
   │
   ▼
RUS-CHN scoring
   │
   ▼
Bone-age estimation
   │
   ▼
Annotated image + assessment report
```

### Main features

- Custom YOLOv5 hand-bone localization
- ResNet-based skeletal maturity classification
- RUS-CHN scoring for 13 anatomical regions
- Automatic bone-age estimation
- Annotated image preview
- Desktop GUI built with Tkinter
- CUDA acceleration when available

### Repository structure

```text
BoneAgeVision/
├── .github/
│   └── workflows/
├── src/
│   └── bone_age_vision/
│       ├── core/
│       │   ├── analyzer.py
│       │   ├── assets.py
│       │   ├── classifier.py
│       │   ├── detector.py
│       │   ├── domain.py
│       │   ├── image_processing.py
│       │   ├── regions.py
│       │   ├── resnet.py
│       │   └── scoring.py
│       ├── gui/
│       ├── __main__.py
│       └── paths.py
├── tests/
├── weights/
│   └── README.md
├── pyproject.toml
└── README.md
```

### Requirements

- Python 3.10+
- PyTorch
- TorchVision
- NumPy
- Pillow
- OpenCV
- Tkinter
- CUDA-compatible GPU optional

### Installation

```bash
git clone https://github.com/ZJY-HSBL/BoneAgeVision.git
cd BoneAgeVision

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

### Model files

Model checkpoints are not included in the repository. To run inference, prepare the following files locally:

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

Place them in the project-level `weights/` directory, or provide another directory with `--weights`.

### Run

Default `./weights` directory:

```bash
bone-age-vision
```

Custom model directory:

```bash
bone-age-vision --weights /path/to/weights
```

The first detector initialization requires internet access because PyTorch Hub loads the pinned YOLOv5 source revision used by the project.

---

## 中文

### 项目简介

BoneAgeVision 是一套面向手部 X 光图像的骨龄评估桌面程序。系统结合自定义 YOLOv5 骨骼定位模型、骨骼专用 ResNet 成熟度分类模型与 RUS-CHN 计分方法，对关键骨骼区域进行识别、分级和评分，并输出估算骨龄、分析报告及标注结果图。

### 处理流程

```text
手部 X 光图像
   │
   ▼
YOLOv5 骨骼定位
   │
   ▼
提取 RUS-CHN 所需 13 个关键区域
   │
   ▼
ResNet 骨骼成熟度分级
   │
   ▼
RUS-CHN 计分
   │
   ▼
骨龄估算
   │
   ▼
标注图像 + 评估报告
```

### 主要功能

- YOLOv5 手部骨骼目标定位
- ResNet 骨骼成熟度分级
- 13 个关键区域 RUS-CHN 评分
- 自动骨龄估算
- 检测结果图像标注
- Tkinter 桌面图形界面
- 可用时自动启用 CUDA 加速

### 项目结构

```text
BoneAgeVision/
├── .github/
│   └── workflows/
├── src/
│   └── bone_age_vision/
│       ├── core/
│       │   ├── analyzer.py
│       │   ├── assets.py
│       │   ├── classifier.py
│       │   ├── detector.py
│       │   ├── domain.py
│       │   ├── image_processing.py
│       │   ├── regions.py
│       │   ├── resnet.py
│       │   └── scoring.py
│       ├── gui/
│       ├── __main__.py
│       └── paths.py
├── tests/
├── weights/
│   └── README.md
├── pyproject.toml
└── README.md
```

### 环境要求

- Python 3.10 及以上
- PyTorch
- TorchVision
- NumPy
- Pillow
- OpenCV
- Tkinter
- CUDA GPU 可选

### 安装

```bash
git clone https://github.com/ZJY-HSBL/BoneAgeVision.git
cd BoneAgeVision

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

### 模型文件

仓库不包含模型权重。运行推理前，需要在本地准备以下文件：

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

可以将其放入项目根目录的 `weights/` 中，也可以通过 `--weights` 指定其他目录。

### 运行

使用默认 `./weights` 目录：

```bash
bone-age-vision
```

指定其他权重目录：

```bash
bone-age-vision --weights /path/to/weights
```

首次初始化检测器时需要联网，以便 PyTorch Hub 加载项目所使用的固定 YOLOv5 源码版本。

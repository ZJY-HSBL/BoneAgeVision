# Model checkpoints / 模型权重

This directory contains the detector and nine skeletal-stage classifier checkpoints required by BoneAgeVision. The files are intentionally kept outside the Python package and tracked with Git LFS.

本目录保存 BoneAgeVision 运行所需的目标检测权重与 9 个骨骼分级权重。大文件不放入 Python 包内部，并通过 Git LFS 管理。

Required files / 必需文件：

- `best.pt`
- `Resnet_DIP.pt`
- `Resnet_DIPFirst.pt`
- `Resnet_MCP.pt`
- `Resnet_MCPFirst.pt`
- `Resnet_MIP.pt`
- `Resnet_PIP.pt`
- `Resnet_PIPFirst.pt`
- `Resnet_Radius.pt`
- `Resnet_Ulna.pt`

Before the first Git push / 首次推送前：

```bash
git lfs install
git add .gitattributes weights/*.pt
git add .
git commit -m "Initial BoneAgeVision release"
git push -u origin main
```

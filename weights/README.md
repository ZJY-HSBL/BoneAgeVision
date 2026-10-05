# Model checkpoints / 模型权重

BoneAgeVision does **not** publish trained model checkpoints in this repository. Keep the private checkpoint files in this directory for local development, or store them elsewhere and pass that directory with `--weights`.

BoneAgeVision **不在本仓库公开训练权重**。本地开发时可将私有权重放在此目录，也可以保存在其他位置，并通过 `--weights` 指定目录。

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

All `weights/*.pt` files are ignored by Git to prevent accidental publication.

所有 `weights/*.pt` 均已加入 `.gitignore`，避免误提交模型文件。

"""Residual network used to classify skeletal maturity stages."""

from torch import Tensor, nn

NUM_CLASSES = {
    "DIP": 11,
    "DIPFirst": 11,
    "MCP": 10,
    "MCPFirst": 11,
    "MIP": 12,
    "PIP": 12,
    "PIPFirst": 12,
    "Radius": 14,
    "Ulna": 12,
}


class BoneStageResNet(nn.Module):
    """ResNet-like classifier matching the supplied checkpoint state dictionaries."""

    def __init__(self, bone_type: str) -> None:
        super().__init__()
        if bone_type not in NUM_CLASSES:
            raise ValueError(f"unsupported bone type: {bone_type}")

        self.bone_type = bone_type
        self.relu = nn.ReLU()
        self.layer1_conv64_and_maxPool = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer2_conv64 = self._same_width_block(64)
        self.layer3_conv64 = self._same_width_block(64)
        self.layer4_conv64_to_conv128 = self._downsample_block(64, 128)
        self.layer4_res128 = nn.Conv2d(64, 128, 1, 2, 0, bias=False)
        self.layer5_conv128 = self._same_width_block(128)
        self.layer6_conv128_to_conv256 = self._downsample_block(128, 256)
        self.layer6_res256 = nn.Conv2d(128, 256, 1, 2, 0, bias=False)
        self.layer7_conv256 = self._same_width_block(256)
        self.layer8_conv256_to_conv512 = self._downsample_block(256, 512)
        self.layer8_res512 = nn.Conv2d(256, 512, 1, 2, 0, bias=False)
        self.layer9_conv512 = self._same_width_block(512)
        self.layer10_axgPool = nn.AvgPool2d(2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, NUM_CLASSES[bone_type]),
        )

    @staticmethod
    def _same_width_block(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    @staticmethod
    def _downsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1_conv64_and_maxPool(x)

        residual = x
        x = self.relu(self.layer2_conv64(x) + residual)

        residual = x
        x = self.relu(self.layer3_conv64(x) + residual)

        residual = self.layer4_res128(x)
        x = self.relu(self.layer4_conv64_to_conv128(x) + residual)

        residual = x
        x = self.relu(self.layer5_conv128(x) + residual)

        residual = self.layer6_res256(x)
        x = self.relu(self.layer6_conv128_to_conv256(x) + residual)

        residual = x
        x = self.relu(self.layer7_conv256(x) + residual)

        residual = self.layer8_res512(x)
        x = self.relu(self.layer8_conv256_to_conv512(x) + residual)

        residual = x
        x = self.relu(self.layer9_conv512(x) + residual)

        x = self.layer10_axgPool(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

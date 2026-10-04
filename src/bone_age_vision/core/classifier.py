"""Bone-stage classification using the supplied ResNet checkpoints."""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from bone_age_vision.core.image_processing import extract_region
from bone_age_vision.core.resnet import BoneStageResNet

Sex = Literal["boy", "girl"]

MODEL_FOR_BONE = {
    "DIPFifth": "DIP",
    "DIPThird": "DIP",
    "DIPFirst": "DIPFirst",
    "MCPFifth": "MCP",
    "MCPThird": "MCP",
    "MCPFirst": "MCPFirst",
    "MIPFifth": "MIP",
    "MIPThird": "MIP",
    "PIPFifth": "PIP",
    "PIPThird": "PIP",
    "PIPFirst": "PIPFirst",
    "Radius": "Radius",
    "Ulna": "Ulna",
}

MODEL_TYPES = tuple(dict.fromkeys(MODEL_FOR_BONE.values()))


class BoneClassifier:
    """Load all stage classifiers once and run inference on detected bone regions."""

    def __init__(self, weights_dir: Path, device: torch.device) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((96, 96), antialias=True),
                transforms.ToTensor(),
            ]
        )
        self.models = self._load_models()

    def _load_models(self) -> dict[str, nn.Module]:
        models: dict[str, nn.Module] = {}
        for model_type in MODEL_TYPES:
            checkpoint = self.weights_dir / f"Resnet_{model_type}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing classifier checkpoint: {checkpoint}")

            model = BoneStageResNet(model_type).to(self.device)
            state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            models[model_type] = model
        return models

    def classify(self, image_rgb: np.ndarray, bone_name: str, box: list[float]) -> int:
        model_type = MODEL_FOR_BONE.get(bone_name)
        if model_type is None:
            raise ValueError(f"unsupported bone name: {bone_name}")

        region = extract_region(image_rgb, box)
        image = Image.fromarray(region.astype(np.uint8), mode="RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.models[model_type](tensor)
        return int(logits.argmax(dim=1).item())

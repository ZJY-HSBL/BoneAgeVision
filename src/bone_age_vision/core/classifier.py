"""Bone-stage classification using the external ResNet checkpoints."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from bone_age_vision.core.domain import BoneName, Box
from bone_age_vision.core.image_processing import extract_region
from bone_age_vision.core.resnet import BoneStageResNet

MODEL_FOR_BONE: dict[BoneName, str] = {
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
    """Load stage classifiers once and run inference on selected bone regions."""

    def __init__(self, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((96, 96), antialias=True),
                transforms.ToTensor(),
            ]
        )
        self.models = self._load_models(weights_dir)

    def _load_models(self, weights_dir: Path) -> dict[str, nn.Module]:
        models: dict[str, nn.Module] = {}
        for model_type in MODEL_TYPES:
            checkpoint = weights_dir / f"Resnet_{model_type}.pt"
            model = BoneStageResNet(model_type).to(self.device)
            state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            models[model_type] = model
        return models

    def classify(self, image_rgb: np.ndarray, bone_name: BoneName, box: Box) -> int:
        """Return the zero-based maturity-stage index for one selected region."""
        model_type = MODEL_FOR_BONE[bone_name]
        region = extract_region(image_rgb, box)
        image = Image.fromarray(region.astype(np.uint8), mode="RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.models[model_type](tensor)
        return int(logits.argmax(dim=1).item())

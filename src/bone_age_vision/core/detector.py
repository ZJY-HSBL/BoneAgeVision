"""YOLOv5 hand-bone detector adapter."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from bone_age_vision.core.assets import DETECTOR_CHECKPOINT
from bone_age_vision.core.domain import Detection

YOLOV5_REPOSITORY = "ultralytics/yolov5:4add2aff6e3d926586a3eab4659f3b498dd444a1"


class BoneDetector:
    """Load the custom YOLOv5 detector and expose framework-independent detections."""

    def __init__(self, weights_dir: Path, device: torch.device, confidence: float = 0.60) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in the interval (0, 1]")

        checkpoint = weights_dir / DETECTOR_CHECKPOINT
        self.device = device
        self.model: Any = torch.hub.load(
            YOLOV5_REPOSITORY,
            "custom",
            path=str(checkpoint),
            trust_repo=True,
            _verbose=False,
            device=str(device),
        )
        self.model.conf = confidence
        self.model.eval()

    def detect(self, image_path: Path) -> tuple[np.ndarray, tuple[Detection, ...]]:
        """Load an image, run detection, and convert tensor rows to domain objects."""
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")

        with Image.open(image_path) as source:
            image_rgb = np.array(source.convert("RGB"), copy=True)

        with torch.inference_mode():
            result = self.model(str(image_path))

        rows = result.xyxy[0].detach().cpu().tolist()
        detections = tuple(
            Detection(
                box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                confidence=float(row[4]),
                class_id=int(row[5]),
            )
            for row in rows
        )
        return image_rgb, detections

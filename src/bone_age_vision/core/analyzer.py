"""End-to-end bone-age inference pipeline."""

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from bone_age_vision.core.classifier import BoneClassifier, Sex
from bone_age_vision.core.scoring import calculate_bone_age, format_report, score_for_prediction

LOGGER = logging.getLogger(__name__)
YOLOV5_REPOSITORY = "ultralytics/yolov5:4add2aff6e3d926586a3eab4659f3b498dd444a1"

# YOLO class -> x-sorted detection index -> anatomical target.
DETECTION_LAYOUT = (
    (6, ((0, "DIPFifth"), (2, "DIPThird"))),
    (5, ((0, "MIPFifth"), (2, "MIPThird"))),
    (4, ((0, "PIPFifth"), (2, "PIPThird"), (4, "PIPFirst"))),
    (3, ((0, "MCPFifth"), (2, "MCPThird"), (4, "MCPFirst"))),
    (2, ((0, "DIPFirst"),)),
    (1, ((0, "Ulna"),)),
    (0, ((0, "Radius"),)),
)

BOX_COLORS = (
    "#2563EB",
    "#16A34A",
    "#CA8A04",
    "#DB2777",
    "#EA580C",
    "#7C3AED",
    "#DC2626",
)


class IncompleteDetectionError(RuntimeError):
    """Raised when a complete 13-region RUS-CHN assessment cannot be formed."""


class BoneAgeAnalyzer:
    """Coordinate YOLO localization, stage classification, scoring and rendering."""

    def __init__(self, weights_dir: Path, confidence: float = 0.60) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = BoneClassifier(self.weights_dir, self.device)
        self.detector = self._load_detector(confidence)
        LOGGER.info("inference device: %s", self.device)

    def _load_detector(self, confidence: float):
        checkpoint = self.weights_dir / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing detector checkpoint: {checkpoint}")

        model = torch.hub.load(
            YOLOV5_REPOSITORY,
            "custom",
            path=str(checkpoint),
            trust_repo=True,
            _verbose=False,
            device=str(self.device),
        )
        model.conf = confidence
        model.eval()
        return model

    def detect(self, image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")

        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        with torch.inference_mode():
            result = self.detector(str(image_path))
        boxes = result.xyxy[0].detach().cpu()
        return image_rgb, boxes

    @staticmethod
    def _select_regions(boxes: torch.Tensor) -> list[tuple[str, list[float]]]:
        selected: list[tuple[str, list[float]]] = []
        missing: list[str] = []

        for class_index, targets in DETECTION_LAYOUT:
            class_boxes = boxes[boxes[:, 5] == class_index]
            if len(class_boxes):
                class_boxes = class_boxes[class_boxes[:, 0].argsort()]

            for sorted_index, bone_name in targets:
                if sorted_index >= len(class_boxes):
                    missing.append(bone_name)
                    continue
                coordinates = [float(value) for value in class_boxes[sorted_index, :4].tolist()]
                selected.append((bone_name, coordinates))

        if missing:
            raise IncompleteDetectionError(
                "关键骨骼检测不完整，无法可靠计算骨龄。缺失区域：" + ", ".join(missing)
            )
        return selected

    @staticmethod
    def _load_font(size: int = 24) -> ImageFont.ImageFont:
        candidates: Iterable[str] = (
            "msyh.ttc",
            "simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def analyze(self, image_path: str | Path, sex: Sex) -> tuple[str, Image.Image]:
        if sex not in ("boy", "girl"):
            raise ValueError("sex must be 'boy' or 'girl'")

        path = Path(image_path)
        image_rgb, boxes = self.detect(path)
        regions = self._select_regions(boxes)

        annotated = Image.fromarray(image_rgb.copy())
        draw = ImageDraw.Draw(annotated)
        font = self._load_font()
        assessments: dict[str, tuple[int, int]] = {}

        for index, (bone_name, box) in enumerate(regions):
            prediction_index = self.classifier.classify(image_rgb, bone_name, box)
            stage = prediction_index + 1
            score = score_for_prediction(sex, bone_name, prediction_index)
            assessments[bone_name] = (stage, score)

            x1, y1, x2, y2 = box
            color = BOX_COLORS[index % len(BOX_COLORS)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            text_y = max(0, y1 - 26)
            draw.text((x1, text_y), bone_name, fill=color, font=font)

        total_score = sum(score for _, score in assessments.values())
        bone_age = calculate_bone_age(total_score, sex)
        report = format_report(assessments, total_score, bone_age)
        return report, annotated

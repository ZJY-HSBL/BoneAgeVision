"""End-to-end bone-age inference pipeline."""

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from bone_age_vision.core.assets import validate_weights_dir
from bone_age_vision.core.classifier import BoneClassifier
from bone_age_vision.core.detector import BoneDetector
from bone_age_vision.core.domain import Assessment, BoneName, BoneRegion, Sex
from bone_age_vision.core.regions import select_required_regions
from bone_age_vision.core.scoring import calculate_bone_age, format_report, score_for_prediction

LOGGER = logging.getLogger(__name__)

BOX_COLORS = (
    "#2563EB",
    "#16A34A",
    "#CA8A04",
    "#DB2777",
    "#EA580C",
    "#7C3AED",
    "#DC2626",
)


class BoneAgeAnalyzer:
    """Coordinate detection, stage classification, scoring, and rendering."""

    def __init__(self, weights_dir: str | Path, confidence: float = 0.60) -> None:
        directory = validate_weights_dir(weights_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = BoneDetector(directory, self.device, confidence)
        self.classifier = BoneClassifier(directory, self.device)
        LOGGER.info("inference device: %s", self.device)

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

    @classmethod
    def _render_regions(cls, image_rgb: np.ndarray, regions: tuple[BoneRegion, ...]) -> Image.Image:
        annotated = Image.fromarray(image_rgb.copy())
        draw = ImageDraw.Draw(annotated)
        font = cls._load_font()

        for index, region in enumerate(regions):
            x1, y1, x2, y2 = region.box
            color = BOX_COLORS[index % len(BOX_COLORS)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1, max(0, y1 - 26)), region.bone_name, fill=color, font=font)
        return annotated

    def analyze(self, image_path: str | Path, sex: Sex) -> tuple[str, Image.Image]:
        """Run the complete assessment and return a report plus annotated image."""
        if sex not in ("boy", "girl"):
            raise ValueError("sex must be 'boy' or 'girl'")

        image_rgb, detections = self.detector.detect(Path(image_path))
        regions = select_required_regions(detections)
        assessments: dict[BoneName, Assessment] = {}

        for region in regions:
            prediction_index = self.classifier.classify(
                image_rgb,
                region.bone_name,
                region.box,
            )
            assessments[region.bone_name] = Assessment(
                stage=prediction_index + 1,
                score=score_for_prediction(sex, region.bone_name, prediction_index),
            )

        total_score = sum(assessment.score for assessment in assessments.values())
        bone_age = calculate_bone_age(total_score, sex)
        report = format_report(assessments, total_score, bone_age)
        return report, self._render_regions(image_rgb, regions)

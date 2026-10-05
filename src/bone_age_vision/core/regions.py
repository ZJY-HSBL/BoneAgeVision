"""Map detector outputs to the 13 anatomical regions required by RUS-CHN."""

from collections.abc import Iterable

from bone_age_vision.core.domain import BoneName, BoneRegion, Detection

# YOLO class -> x-sorted detection index -> anatomical target.
DETECTION_LAYOUT: tuple[tuple[int, tuple[tuple[int, BoneName], ...]], ...] = (
    (6, ((0, "DIPFifth"), (2, "DIPThird"))),
    (5, ((0, "MIPFifth"), (2, "MIPThird"))),
    (4, ((0, "PIPFifth"), (2, "PIPThird"), (4, "PIPFirst"))),
    (3, ((0, "MCPFifth"), (2, "MCPThird"), (4, "MCPFirst"))),
    (2, ((0, "DIPFirst"),)),
    (1, ((0, "Ulna"),)),
    (0, ((0, "Radius"),)),
)


class IncompleteDetectionError(RuntimeError):
    """Raised when a complete 13-region RUS-CHN assessment cannot be formed."""


def select_required_regions(detections: Iterable[Detection]) -> tuple[BoneRegion, ...]:
    """Select and order the 13 required regions from detector results."""
    grouped: dict[int, list[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_id, []).append(detection)

    selected: list[BoneRegion] = []
    missing: list[BoneName] = []

    for class_id, targets in DETECTION_LAYOUT:
        class_detections = sorted(grouped.get(class_id, ()), key=lambda item: item.box[0])
        for sorted_index, bone_name in targets:
            if sorted_index >= len(class_detections):
                missing.append(bone_name)
                continue
            selected.append(BoneRegion(bone_name=bone_name, box=class_detections[sorted_index].box))

    if missing:
        missing_text = ", ".join(missing)
        raise IncompleteDetectionError(
            "关键骨骼检测不完整，无法可靠计算骨龄。"
            f"缺失区域：{missing_text}"
        )
    return tuple(selected)

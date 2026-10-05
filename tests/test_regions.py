import pytest

from bone_age_vision.core.domain import Detection
from bone_age_vision.core.regions import IncompleteDetectionError, select_required_regions


def detection(class_id: int, x: float) -> Detection:
    return Detection(box=(x, 10.0, x + 5.0, 20.0), confidence=0.9, class_id=class_id)


def test_select_required_regions_uses_x_sorted_anatomical_positions() -> None:
    detections = [
        detection(6, 30), detection(6, 10), detection(6, 20),
        detection(5, 30), detection(5, 20), detection(5, 10),
        detection(4, 50), detection(4, 10), detection(4, 40), detection(4, 30), detection(4, 20),
        detection(3, 40), detection(3, 20), detection(3, 50), detection(3, 10), detection(3, 30),
        detection(2, 12),
        detection(1, 14),
        detection(0, 16),
    ]

    regions = select_required_regions(detections)

    assert len(regions) == 13
    selected = {region.bone_name: region.box[0] for region in regions}
    assert selected["DIPFifth"] == 10
    assert selected["DIPThird"] == 30
    assert selected["PIPFirst"] == 50
    assert selected["MCPThird"] == 30
    assert selected["Radius"] == 16


def test_select_required_regions_fails_when_required_regions_are_missing() -> None:
    with pytest.raises(IncompleteDetectionError, match="DIPFifth"):
        select_required_regions(())

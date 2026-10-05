import numpy as np
import pytest

from bone_age_vision.core.image_processing import extract_region


def test_extract_region_clips_box_to_image_bounds() -> None:
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)

    crop = extract_region(image, (-2.0, 1.0, 99.0, 4.0))

    assert crop.shape == (3, 6, 3)
    np.testing.assert_array_equal(crop, image[1:4, 0:6])


@pytest.mark.parametrize(
    "box",
    [
        (1.0, 1.0, 1.0, 4.0),
        (4.0, 3.0, 2.0, 4.0),
        (0.0, 9.0, 2.0, 10.0),
    ],
)
def test_extract_region_rejects_empty_or_inverted_boxes(box: tuple[float, ...]) -> None:
    image = np.zeros((5, 6, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="invalid or empty crop"):
        extract_region(image, box)


def test_extract_region_rejects_invalid_image_rank() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        extract_region(np.zeros((1, 2, 3, 4), dtype=np.uint8), (0, 0, 1, 1))

"""Image utilities used by the inference pipeline."""

from collections.abc import Sequence

import numpy as np


def extract_region(image: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Return a clipped image crop for an ``[x1, y1, x2, y2]`` box."""
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D NumPy array")
    if len(box) != 4:
        raise ValueError("box must contain exactly four coordinates")

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(float(value)) for value in box)
    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid or empty crop after clipping: {(x1, y1, x2, y2)}")

    return image[y1:y2, x1:x2]

"""Validation for external model checkpoints."""

from pathlib import Path

DETECTOR_CHECKPOINT = "best.pt"
CLASSIFIER_CHECKPOINTS = (
    "Resnet_DIP.pt",
    "Resnet_DIPFirst.pt",
    "Resnet_MCP.pt",
    "Resnet_MCPFirst.pt",
    "Resnet_MIP.pt",
    "Resnet_PIP.pt",
    "Resnet_PIPFirst.pt",
    "Resnet_Radius.pt",
    "Resnet_Ulna.pt",
)
REQUIRED_CHECKPOINTS = (DETECTOR_CHECKPOINT, *CLASSIFIER_CHECKPOINTS)


class MissingWeightsError(FileNotFoundError):
    """Raised when one or more required model checkpoints are unavailable."""


def validate_weights_dir(weights_dir: str | Path) -> Path:
    """Return a resolved weights directory after checking every required checkpoint."""
    directory = Path(weights_dir).expanduser().resolve()
    if not directory.is_dir():
        raise MissingWeightsError(f"model weights directory does not exist: {directory}")

    missing = [name for name in REQUIRED_CHECKPOINTS if not (directory / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise MissingWeightsError(
            f"model weights directory is incomplete: {directory}; missing: {missing_text}"
        )
    return directory

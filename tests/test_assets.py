from pathlib import Path

import pytest

from bone_age_vision.core.assets import (
    REQUIRED_CHECKPOINTS,
    MissingWeightsError,
    validate_weights_dir,
)


def test_validate_weights_dir_accepts_complete_directory(tmp_path: Path) -> None:
    for filename in REQUIRED_CHECKPOINTS:
        (tmp_path / filename).touch()

    assert validate_weights_dir(tmp_path) == tmp_path.resolve()


def test_validate_weights_dir_reports_all_missing_files(tmp_path: Path) -> None:
    (tmp_path / REQUIRED_CHECKPOINTS[0]).touch()

    with pytest.raises(MissingWeightsError) as exc_info:
        validate_weights_dir(tmp_path)

    message = str(exc_info.value)
    assert REQUIRED_CHECKPOINTS[1] in message
    assert REQUIRED_CHECKPOINTS[-1] in message


def test_validate_weights_dir_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(MissingWeightsError, match="does not exist"):
        validate_weights_dir(tmp_path / "missing")

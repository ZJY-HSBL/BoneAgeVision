import pytest

from bone_age_vision.core.domain import Assessment
from bone_age_vision.core.scoring import (
    BONE_ORDER,
    calculate_bone_age,
    format_report,
    score_for_prediction,
)


def test_score_lookup_uses_zero_based_prediction_index() -> None:
    assert score_for_prediction("boy", "Radius", 0) == 8
    assert score_for_prediction("girl", "MCPFirst", 10) == 66


def test_score_lookup_rejects_unknown_bone_and_out_of_range_stage() -> None:
    with pytest.raises(ValueError, match="unsupported sex/bone"):
        score_for_prediction("boy", "Unknown", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        score_for_prediction("girl", "Radius", 14)


def test_bone_age_polynomial_matches_known_baseline() -> None:
    assert calculate_bone_age(0, "boy") == 2.02
    assert calculate_bone_age(0, "girl") == 5.81


def test_bone_age_rejects_negative_score() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_bone_age(-1, "boy")


def test_format_report_requires_all_thirteen_regions() -> None:
    assessments = {bone: Assessment(stage=1, score=1) for bone in BONE_ORDER}
    report = format_report(assessments, total_score=13, bone_age=8.5)

    assert "CHN 总得分 13 分" in report
    assert "8.5 岁" in report

    assessments.pop("Radius")
    with pytest.raises(ValueError, match="Radius"):
        format_report(assessments, total_score=12, bone_age=8.0)

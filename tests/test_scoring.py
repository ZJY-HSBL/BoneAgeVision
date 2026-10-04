from bone_age_vision.core.scoring import calculate_bone_age, score_for_prediction


def test_score_lookup_uses_zero_based_prediction_index() -> None:
    assert score_for_prediction("boy", "Radius", 0) == 8
    assert score_for_prediction("girl", "MCPFirst", 10) == 66


def test_bone_age_polynomial_matches_known_baseline() -> None:
    assert calculate_bone_age(0, "boy") == 2.02
    assert calculate_bone_age(0, "girl") == 5.81

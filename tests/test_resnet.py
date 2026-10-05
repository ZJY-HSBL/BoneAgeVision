import pytest

torch = pytest.importorskip("torch")

from bone_age_vision.core.resnet import BoneStageResNet, NUM_CLASSES  # noqa: E402


@pytest.mark.parametrize("bone_type, classes", NUM_CLASSES.items())
def test_resnet_output_shape_matches_checkpoint_class_count(bone_type: str, classes: int) -> None:
    model = BoneStageResNet(bone_type).eval()

    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 96, 96))

    assert output.shape == (1, classes)

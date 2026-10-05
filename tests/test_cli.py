from pathlib import Path

from bone_age_vision.__main__ import build_parser


def test_parser_accepts_external_weights_directory() -> None:
    weights = Path("private-models")

    args = build_parser().parse_args(["--weights", str(weights)])

    assert args.weights == weights

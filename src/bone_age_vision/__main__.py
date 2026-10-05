"""Application entry point."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from bone_age_vision.paths import DEFAULT_WEIGHTS_DIR


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without starting the GUI."""
    parser = argparse.ArgumentParser(description="BoneAgeVision desktop application")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="directory containing the private model checkpoint files (default: ./weights)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Configure logging, parse runtime paths, and launch the desktop application."""
    args = build_parser().parse_args(argv)

    from bone_age_vision.gui.main_window import MainWindow

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    MainWindow(weights_dir=args.weights).run()


if __name__ == "__main__":
    main()

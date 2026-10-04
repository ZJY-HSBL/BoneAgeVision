"""Application entry point."""

import logging

from bone_age_vision.gui.main_window import MainWindow


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    MainWindow().run()


if __name__ == "__main__":
    main()

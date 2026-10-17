"""GUI smoke tests with a virtual display and no inference weights."""

import os
from pathlib import Path
from tkinter import filedialog

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")
if not os.environ.get("DISPLAY"):
    pytest.skip("Desktop display unavailable", allow_module_level=True)

from PIL import Image  # noqa: E402

from bone_age_vision.core.domain import AnalysisResult, Assessment  # noqa: E402
from bone_age_vision.core.scoring import BONE_ORDER  # noqa: E402
from bone_age_vision.gui.main_window import MainWindow  # noqa: E402


def test_desktop_preview_results_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "hand.png"
    Image.new("RGB", (80, 120), "black").save(image_path)
    app = MainWindow(tmp_path)
    try:
        app.root.update()
        assert app.result is None
        assert len(app.table.get_children()) == 13
        assert app.export_report_button["state"] == "disabled"

        monkeypatch.setattr(filedialog, "askopenfilename", lambda **_: str(image_path))
        app._choose_image()
        assert app.source_path == image_path
        assert app.image_viewer.image is not None

        assessments = {bone: Assessment(stage=1, score=4) for bone in BONE_ORDER}
        result = AnalysisResult(
            report="Mock report",
            annotated_image=Image.new("RGB", (80, 120), "gray"),
            total_score=52,
            bone_age=8.5,
            assessments=assessments,
        )
        app._show_result(result)
        assert app.age_label["text"] == "8.50 岁"
        assert app.export_image_button["state"] == "normal"
        assert len(app.table.get_children()) == 13

        report_path = tmp_path / "result.txt"
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_: str(report_path))
        app._export_report()
        assert report_path.read_text(encoding="utf-8") == "Mock report\n"

        picture_path = tmp_path / "annotated.png"
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_: str(picture_path))
        app._export_image()
        assert picture_path.is_file()

        app._clear_result()
        assert app.result is None
        assert app.export_image_button["state"] == "disabled"
    finally:
        app._on_close()

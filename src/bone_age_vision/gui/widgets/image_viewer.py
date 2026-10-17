"""Resize-aware image preview widget, preserving original aspect ratio."""

import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


class ImageViewer(tk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, bg="#15243D")
        self.image: Image.Image | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.canvas = tk.Canvas(self, bg="#15243D", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._refresh)
        self._refresh()

    def show_image(self, source: str | Path | Image.Image) -> None:
        if isinstance(source, (str, Path)):
            with Image.open(source) as opened:
                image = opened.convert("RGB")
        else:
            image = source.convert("RGB")
        self.image = image.copy()
        self._refresh()

    def _refresh(self, _event: object = None) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if self.image is None:
            self.canvas.create_text(
                width // 2, height // 2 - 10,
                text="尚未选择影像", font=("Microsoft YaHei UI", 16, "bold"),
                fill="#DDE7F6",
            )
            self.canvas.create_text(
                width // 2, height // 2 + 25,
                text="请选择手部 X 光片开始预览", font=("Microsoft YaHei UI", 10),
                fill="#879BB7",
            )
            return
        preview = self.image.copy()
        preview.thumbnail((max(1, width - 32), max(1, height - 32)), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(width // 2, height // 2, image=self.current_photo, anchor="center")

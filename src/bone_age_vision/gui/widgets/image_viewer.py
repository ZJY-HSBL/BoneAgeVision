"""Image preview widget."""

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk


class ImageViewer(tk.Frame):
    def __init__(self, parent, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)
        self.configure(bg="#F3F4F6", highlightthickness=1, highlightbackground="#E5E7EB")
        self.display_label = ttk.Label(self, text="请上传图片以预览", background="#F3F4F6")
        self.display_label.place(relx=0.5, rely=0.5, anchor="center")
        self.current_photo: ImageTk.PhotoImage | None = None

    def show_image(self, image_source: str | Path | Image.Image) -> None:
        if isinstance(image_source, (str, Path)):
            with Image.open(image_source) as source:
                image = source.convert("RGB")
        else:
            image = image_source.convert("RGB")

        width = max(self.winfo_width(), 600)
        height = max(self.winfo_height(), 500)
        preview = image.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(preview)
        self.display_label.configure(image=self.current_photo, text="")

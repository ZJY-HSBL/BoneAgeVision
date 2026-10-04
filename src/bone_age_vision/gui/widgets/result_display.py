"""Read-only analysis report widget."""

import tkinter as tk
from tkinter import ttk


class ResultDisplay(ttk.LabelFrame):
    def __init__(self, parent, *args, **kwargs) -> None:
        super().__init__(parent, text=" 分析报告 ", padding=15, *args, **kwargs)
        scrollbar = ttk.Scrollbar(self)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text = tk.Text(
            self,
            width=34,
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#1F2937",
            bd=0,
            padx=8,
            pady=8,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text.yview)
        self.update_result("等待检测……\n请先选择手部 X 光图像。")

    def update_result(self, value: str) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, value)
        self.text.configure(state=tk.DISABLED)

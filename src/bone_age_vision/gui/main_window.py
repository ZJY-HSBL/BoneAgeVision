"""BoneAgeVision desktop workspace: input, image, and assessment panels."""

from __future__ import annotations

import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import cast

from bone_age_vision.core.analyzer import BoneAgeAnalyzer
from bone_age_vision.core.domain import AnalysisResult, Sex
from bone_age_vision.core.scoring import BONE_LABELS_ZH, BONE_ORDER
from bone_age_vision.gui.styles.theme import COLORS, FONTS
from bone_age_vision.gui.widgets.image_viewer import ImageViewer


class MainWindow:
    """Single-window image assessment application with no inert navigation."""

    def __init__(self, weights_dir: Path) -> None:
        self.root = tk.Tk()
        self.weights_dir = Path(weights_dir).expanduser()
        self.analyzer: BoneAgeAnalyzer | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bone-age-analysis")
        self.sex = tk.StringVar(value="boy")
        self.source_path: Path | None = None
        self.result: AnalysisResult | None = None
        self._busy = False
        self._closed = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _label(self, parent: tk.Misc, text: str, *, size: str = "body", fg: str = "text",
               bg: str = "surface", **kwargs: object) -> tk.Label:
        return tk.Label(parent, text=text, font=FONTS[size], bg=COLORS[bg],
                        fg=COLORS[fg], anchor="w", **kwargs)

    def _button(self, parent: tk.Misc, text: str, command: object,
                *, primary: bool = False) -> tk.Button:
        key = "accent" if primary else "surface_alt"
        return tk.Button(parent, text=text, command=command, font=FONTS["button"],
                         bg=COLORS[key], fg=COLORS["white"] if primary else COLORS["text"],
                         activebackground=COLORS["accent_hover"] if primary else COLORS["border"],
                         activeforeground=COLORS["white"] if primary else COLORS["text"],
                         relief="flat", bd=0, cursor="hand2", padx=18, pady=12,
                         disabledforeground=COLORS["muted"])

    def _card(self, parent: tk.Misc, *, width: int | None = None) -> tk.Frame:
        options: dict[str, object] = {"bg": COLORS["surface"], "highlightthickness": 1,
                                      "highlightbackground": COLORS["border"]}
        if width is not None:
            options["width"] = width
        return tk.Frame(parent, **options)

    def _build(self) -> None:
        self.root.title("BoneAgeVision | 骨龄智能评估")
        self.root.geometry("1450x900")
        self.root.minsize(1120, 710)
        self.root.configure(bg=COLORS["canvas"])
        header = tk.Frame(self.root, bg=COLORS["navy"], height=94)
        header.pack(fill="x")
        header.pack_propagate(False)
        symbol = tk.Frame(header, bg=COLORS["accent"], width=44, height=44)
        symbol.pack(side="left", padx=(30, 15), pady=24)
        symbol.pack_propagate(False)
        tk.Label(symbol, text="B", fg="white", bg=COLORS["accent"],
                 font=(FONTS["title"][0], 22, "bold")).place(relx=.5, rely=.5, anchor="center")
        title_group = tk.Frame(header, bg=COLORS["navy"])
        title_group.pack(side="left", pady=20)
        tk.Label(title_group, text="BoneAgeVision", bg=COLORS["navy"], fg="white",
                 font=FONTS["title"]).pack(anchor="w")
        tk.Label(title_group, text="智能骨龄评估工作台  /  Bone Age Assessment",
                 bg=COLORS["navy"], fg=COLORS["header_sub"],
                 font=FONTS["small"]).pack(anchor="w", pady=(3, 0))
        tk.Label(header, text="YOLOv5  /  ResNet  /  RUS-CHN     •     v0.3.0",
                 bg=COLORS["navy"], fg=COLORS["header_sub"],
                 font=FONTS["small"]).pack(side="right", padx=30)

        workspace = tk.Frame(self.root, bg=COLORS["canvas"])
        workspace.pack(fill="both", expand=True, padx=24, pady=22)
        workspace.grid_columnconfigure(0, minsize=264, weight=0)
        workspace.grid_columnconfigure(1, minsize=360, weight=1)
        workspace.grid_columnconfigure(2, minsize=316, weight=0)
        workspace.grid_rowconfigure(0, weight=1)

        self._build_input(workspace)
        self._build_preview(workspace)
        self._build_results(workspace)
        footer = tk.Frame(self.root, bg=COLORS["canvas"], height=32)
        footer.pack(fill="x", padx=28, pady=(0, 7))
        self._label(footer, "研究用途 · 不用于临床诊断或医疗决策", size="small",
                    fg="muted", bg="canvas").pack(side="left")
        self._label(footer, "本地处理  •  Model files stay on your device", size="small",
                    fg="muted", bg="canvas").pack(side="right")

    def _build_input(self, workspace: tk.Frame) -> None:
        left = self._card(workspace, width=280)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_propagate(False)
        body = tk.Frame(left, bg=COLORS["surface"])
        body.pack(fill="both", expand=True, padx=20, pady=20)
        self._label(body, "01   影像与对象", size="section").pack(anchor="w")
        self._label(
            body, "选择手部 X 光片并设置评估对象", size="small", fg="muted"
        ).pack(anchor="w", pady=(6, 24))
        self._label(body, "检测对象", size="strong").pack(anchor="w")
        group = tk.Frame(body, bg=COLORS["surface"])
        group.pack(fill="x", pady=(10, 24))
        for sex, text in (("boy", "男童"), ("girl", "女童")):
            tk.Radiobutton(group, text=text, variable=self.sex, value=sex,
                           command=self._clear_result, bg=COLORS["surface"],
                           activebackground=COLORS["surface"], fg=COLORS["text"],
                           font=FONTS["body"], selectcolor=COLORS["surface"],
                           indicatoron=True).pack(side="left", padx=(0, 24))
        self._label(body, "影像文件", size="strong").pack(anchor="w")
        self.file_info = self._label(body, "尚未选择文件", fg="muted", wraplength=235,
                                     justify="left")
        self.file_info.pack(fill="x", pady=(10, 12))
        self.choose_button = self._button(body, "选择 X 光图像", self._choose_image)
        self.choose_button.pack(fill="x")
        self.analyze_button = self._button(
            body, "开始骨龄分析  →", self._start_analysis, primary=True
        )
        self.analyze_button.pack(fill="x", pady=(14, 0))
        self._label(
            body, "支持 PNG、JPG、BMP、TIFF", size="small", fg="muted"
        ).pack(anchor="w", pady=(15, 0))
        tk.Frame(body, bg=COLORS["border"], height=1).pack(fill="x", pady=(32, 25))
        self._label(body, "分析流程", size="strong").pack(anchor="w", pady=(0, 12))
        for index, name, sub in (
            ("01", "骨骼定位", "YOLOv5 · 目标检测"),
            ("02", "成熟度分级", "ResNet · 关键骨骼"),
            ("03", "骨龄评估", "RUS-CHN · 标准评分"),
        ):
            row = tk.Frame(body, bg=COLORS["surface"])
            row.pack(fill="x", pady=9)
            self._label(row, index, size="small", fg="accent").pack(side="left", padx=(0, 12))
            col = tk.Frame(row, bg=COLORS["surface"])
            col.pack(side="left")
            self._label(col, name, size="strong").pack(anchor="w")
            self._label(col, sub, size="small", fg="muted").pack(anchor="w", pady=(3, 0))
        self.status_label = self._label(body, "●  就绪，等待选择影像", size="small", fg="muted",
                                         wraplength=220)
        self.status_label.pack(side="bottom", fill="x", pady=(20, 0))

    def _build_preview(self, workspace: tk.Frame) -> None:
        center = self._card(workspace)
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 16))
        heading = tk.Frame(center, bg=COLORS["surface"])
        heading.pack(fill="x", padx=22, pady=(22, 15))
        self._label(heading, "02   医学影像", size="section").pack(side="left")
        self._label(heading, "IMAGE VIEWER", size="small", fg="muted").pack(side="right")
        self.image_viewer = ImageViewer(center)
        self.image_viewer.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        bottom = tk.Frame(center, bg=COLORS["surface"])
        bottom.pack(fill="x", padx=22, pady=(0, 22))
        self.image_hint = self._label(bottom, "上传影像后可预览骨骼区域", size="small", fg="muted")
        self.image_hint.pack(side="left")
        self.export_image_button = self._button(bottom, "保存标注图", self._export_image)
        self.export_image_button.configure(state="disabled")
        self.export_image_button.pack(side="right")

    def _build_results(self, workspace: tk.Frame) -> None:
        right = self._card(workspace, width=346)
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_propagate(False)
        body = tk.Frame(right, bg=COLORS["surface"])
        body.pack(fill="both", expand=True, padx=19, pady=20)
        self._label(body, "03   评估结果", size="section").pack(anchor="w")
        self._label(
            body, "RUS-CHN ASSESSMENT", size="small", fg="muted"
        ).pack(anchor="w", pady=(5, 16))
        summary = tk.Frame(body, bg=COLORS["surface_alt"], padx=16, pady=14)
        summary.pack(fill="x")
        self._label(summary, "预测骨龄 / Predicted bone age", size="small", fg="muted",
                    bg="surface_alt").pack(anchor="w")
        self.age_label = self._label(summary, "—", size="metric", bg="surface_alt")
        self.age_label.pack(anchor="w", pady=(6, 12))
        stats = tk.Frame(summary, bg=COLORS["surface_alt"])
        stats.pack(fill="x")
        self.score_label = self._label(stats, "总分   —", size="small", bg="surface_alt")
        self.score_label.pack(side="left")
        self.region_label = self._label(stats, "关键区域   0 / 13", size="small", bg="surface_alt")
        self.region_label.pack(side="right")
        self._label(body, "各区域评分", size="strong").pack(anchor="w", pady=(26, 12))
        table_frame = tk.Frame(body, bg=COLORS["surface"])
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=("bone", "stage", "score"),
                                  show="headings", height=13, selectmode="none")
        self.table.heading("bone", text="骨骼区域")
        self.table.heading("stage", text="分级")
        self.table.heading("score", text="得分")
        self.table.column("bone", width=135, stretch=True)
        self.table.column("stage", width=53, stretch=False, anchor="center")
        self.table.column("score", width=55, stretch=False, anchor="center")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Bone.Treeview", background=COLORS["surface"],
                        fieldbackground=COLORS["surface"], foreground=COLORS["text"],
                        rowheight=27, font=FONTS["small"], borderwidth=0)
        style.configure("Bone.Treeview.Heading", font=FONTS["strong"],
                        background=COLORS["surface_alt"], foreground=COLORS["text"],
                        relief="flat")
        self.table.configure(style="Bone.Treeview")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._fill_empty_rows()
        self.export_report_button = self._button(body, "导出评估报告", self._export_report)
        self.export_report_button.configure(state="disabled")
        self.export_report_button.pack(fill="x", pady=(20, 0))

    def _fill_empty_rows(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        for name in BONE_ORDER:
            self.table.insert("", "end", values=(BONE_LABELS_ZH[name], "—", "—"))

    def _clear_result(self) -> None:
        self.result = None
        self.age_label.configure(text="—")
        self.score_label.configure(text="总分   —")
        self.region_label.configure(text="关键区域   0 / 13")
        self.export_image_button.configure(state="disabled")
        self.export_report_button.configure(state="disabled")
        self._fill_empty_rows()

    def _choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root, title="选择手部 X 光图像",
            filetypes=[("影像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not filename:
            return
        try:
            self.image_viewer.show_image(filename)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法打开影像", str(exc), parent=self.root)
            return
        self.source_path = Path(filename)
        self.file_info.configure(text=self.source_path.name, fg=COLORS["text"])
        self.image_hint.configure(text="原始影像 · 已加载")
        self.status_label.configure(text="●  影像已就绪，可开始分析", fg=COLORS["accent"])
        self._clear_result()

    def _start_analysis(self) -> None:
        if self._busy:
            return
        if self.source_path is None or not self.source_path.is_file():
            messagebox.showwarning("请选择影像", "请先选择有效的手部 X 光图像。", parent=self.root)
            return
        self._clear_result()
        self._busy = True
        self.choose_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled", text="正在分析…")
        self.status_label.configure(text="●  正在加载模型并进行分析", fg=COLORS["accent"])
        future = self.executor.submit(self._analyze, self.source_path, self.sex.get())
        self._check_future(future)

    def _analyze(self, path: Path, sex: str) -> AnalysisResult:
        if sex not in ("boy", "girl"):
            raise ValueError("invalid subject sex")
        if self.analyzer is None:
            self.analyzer = BoneAgeAnalyzer(self.weights_dir)
        return self.analyzer.analyze(path, cast(Sex, sex))

    def _check_future(self, future: Future[AnalysisResult]) -> None:
        """Poll only from Tk's main thread; never call Tk from a worker callback."""
        if self._closed:
            return
        if not future.done():
            self.root.after(120, self._check_future, future)
            return
        self._busy = False
        self.choose_button.configure(state="normal")
        self.analyze_button.configure(state="normal", text="开始骨龄分析  →")
        try:
            result = future.result()
        except Exception as exc:
            self.status_label.configure(text="●  分析未完成", fg=COLORS["error"])
            messagebox.showerror("分析失败", str(exc), parent=self.root)
            return
        self._show_result(result)

    def _show_result(self, result: AnalysisResult) -> None:
        self.result = result
        self.image_viewer.show_image(result.annotated_image)
        self.age_label.configure(text=f"{result.bone_age:.2f} 岁")
        self.score_label.configure(text=f"总分   {result.total_score}")
        self.region_label.configure(text=f"关键区域   {len(result.assessments)} / 13")
        self._fill_empty_rows()
        for item, name in zip(self.table.get_children(), BONE_ORDER, strict=True):
            assess = result.assessments[name]
            self.table.item(item, values=(BONE_LABELS_ZH[name], assess.stage, assess.score))
        self.image_hint.configure(text="分析标注图 · 13 个关键区域")
        self.status_label.configure(text="●  分析完成，结果可导出", fg=COLORS["success"])
        self.export_image_button.configure(state="normal")
        self.export_report_button.configure(state="normal")

    def _export_image(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="保存标注图",
                                             defaultextension=".png",
                                             filetypes=[("PNG 图片", "*.png")])
        if path:
            try:
                self.result.annotated_image.save(path, format="PNG")
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=self.root)

    def _export_report(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="导出评估报告",
                                             defaultextension=".txt",
                                             filetypes=[("文本报告", "*.txt")])
        if path:
            try:
                Path(path).write_text(self.result.report + "\n", encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("导出失败", str(exc), parent=self.root)

    def _on_close(self) -> None:
        self._closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

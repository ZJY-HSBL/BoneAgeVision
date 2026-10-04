"""Main desktop window."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from bone_age_vision.core.analyzer import BoneAgeAnalyzer
from bone_age_vision.gui.styles.theme import COLORS, FONTS
from bone_age_vision.gui.widgets.image_viewer import ImageViewer
from bone_age_vision.gui.widgets.result_display import ResultDisplay
from bone_age_vision.paths import WEIGHTS_DIR


class MainWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.analyzer: BoneAgeAnalyzer | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bone-age-inference")
        self.file_path = tk.StringVar()
        self.sex = tk.StringVar(value="boy")

        self._setup_window()
        self._setup_styles()
        self._create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_window(self) -> None:
        self.root.title("BoneAgeVision · 骨龄智能评估")
        self.root.geometry("1280x850")
        self.root.minsize(1000, 700)
        self.root.configure(bg=COLORS["bg_main"])

    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TLabel",
            background=COLORS["bg_main"],
            foreground=COLORS["text_main"],
            font=FONTS["body"],
        )
        style.configure(
            "Title.TLabel",
            font=FONTS["h1"],
            foreground=COLORS["text_main"],
            background=COLORS["bg_main"],
        )
        style.configure(
            "Subtitle.TLabel",
            font=FONTS["body"],
            foreground=COLORS["text_sub"],
            background=COLORS["bg_main"],
        )
        style.configure("Panel.TFrame", background=COLORS["bg_panel"])
        style.configure("Main.TFrame", background=COLORS["bg_main"])
        style.configure(
            "Primary.TButton",
            font=FONTS["button"],
            background=COLORS["primary"],
            foreground="white",
            borderwidth=0,
            padding=(20, 10),
        )
        style.map(
            "Primary.TButton",
            background=[
                ("active", COLORS["primary_hover"]),
                ("pressed", COLORS["primary_hover"]),
            ],
        )
        style.configure("TRadiobutton", background=COLORS["bg_panel"], font=FONTS["body"])

    def _create_widgets(self) -> None:
        header = ttk.Frame(self.root, style="Main.TFrame")
        header.pack(fill=tk.X, padx=40, pady=(30, 20))
        ttk.Label(header, text="BoneAgeVision", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="YOLOv5 骨骼定位 · ResNet 分级 · RUS-CHN 计分",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(5, 0))

        controls = ttk.Frame(self.root, style="Panel.TFrame")
        controls.pack(fill=tk.X, padx=40, ipady=15)

        sex_frame = ttk.Frame(controls, style="Panel.TFrame")
        sex_frame.pack(side=tk.LEFT, padx=30)
        ttk.Label(
            sex_frame,
            text="检测对象",
            font=FONTS["h2"],
            background=COLORS["bg_panel"],
        ).pack(anchor="w", pady=(0, 5))
        radio_frame = ttk.Frame(sex_frame, style="Panel.TFrame")
        radio_frame.pack(anchor="w")
        ttk.Radiobutton(radio_frame, text="男童", variable=self.sex, value="boy").pack(
            side=tk.LEFT, padx=(0, 15)
        )
        ttk.Radiobutton(radio_frame, text="女童", variable=self.sex, value="girl").pack(
            side=tk.LEFT
        )

        ttk.Separator(controls, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        file_frame = ttk.Frame(controls, style="Panel.TFrame")
        file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        ttk.Label(
            file_frame,
            text="手部 X 光图像",
            font=FONTS["h2"],
            background=COLORS["bg_panel"],
        ).pack(anchor="w", pady=(0, 5))
        input_row = ttk.Frame(file_frame, style="Panel.TFrame")
        input_row.pack(fill=tk.X)
        ttk.Entry(input_row, textvariable=self.file_path, font=("Microsoft YaHei UI", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=3
        )
        ttk.Button(input_row, text="浏览文件", command=self._browse_file).pack(side=tk.LEFT)

        content = ttk.Frame(self.root, style="Main.TFrame")
        content.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        left = ttk.Frame(content, style="Main.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_viewer = ImageViewer(left)
        self.image_viewer.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(content, style="Main.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(30, 0))
        self.detect_button = ttk.Button(
            right,
            text="开始智能分析",
            style="Primary.TButton",
            command=self._start_detection,
        )
        self.detect_button.pack(fill=tk.X, pady=(0, 20))
        self.result_display = ResultDisplay(right)
        self.result_display.pack(fill=tk.BOTH, expand=True)

    def _browse_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择手部 X 光图像",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self.file_path.set(selected)
        try:
            self.image_viewer.show_image(selected)
            self.result_display.update_result("图像已加载。\n点击“开始智能分析”执行推理。")
        except Exception as exc:
            messagebox.showerror("图像加载失败", str(exc))

    def _start_detection(self) -> None:
        path = Path(self.file_path.get())
        if not path.is_file():
            messagebox.showwarning("无效输入", "请先选择有效的图像文件。")
            return

        self.detect_button.configure(state=tk.DISABLED, text="正在分析……")
        self.result_display.update_result("正在加载模型并执行推理……\n首次运行会下载固定版本的 YOLOv5 代码。")
        future = self.executor.submit(self._analyze, path, self.sex.get())
        future.add_done_callback(
            lambda completed: self.root.after(0, self._finish_detection, completed)
        )

    def _analyze(self, path: Path, sex: str):
        if self.analyzer is None:
            self.analyzer = BoneAgeAnalyzer(WEIGHTS_DIR)
        return self.analyzer.analyze(path, sex)  # type: ignore[arg-type]

    def _finish_detection(self, future) -> None:
        self.detect_button.configure(state=tk.NORMAL, text="开始智能分析")
        try:
            report, image = future.result()
        except Exception as exc:
            self.result_display.update_result(f"分析失败：\n{exc}")
            messagebox.showerror("分析失败", str(exc))
            return

        self.image_viewer.show_image(image)
        self.result_display.update_result("【检测完成】\n\n" + report)

    def _on_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

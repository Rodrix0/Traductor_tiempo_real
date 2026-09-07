"""Resizable, always-on-top captions with user-controlled readability."""
import tkinter as tk
from tkinter import ttk


class SubtitleOverlay:
    def __init__(self, root, text, original, font_size=24, opacity=0.95):
        self.window = tk.Toplevel(root)
        self.window.title("Subtítulos · arrastrá esta ventana para moverla")
        width = min(850, root.winfo_screenwidth() - 40)
        left = max(0, (root.winfo_screenwidth() - width) // 2)
        top = max(0, root.winfo_screenheight() - 280)
        self.window.geometry(f"{width}x210+{left}+{top}")
        self.window.minsize(380, 140)
        self.window.configure(bg="#151922")
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", opacity)
        self.font_size = tk.IntVar(value=font_size)
        self.opacity = tk.DoubleVar(value=opacity)
        self.show_original = tk.BooleanVar(value=False)
        toolbar = ttk.Frame(self.window, padding=5)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Tamaño").pack(side="left")
        size_control = ttk.Spinbox(toolbar, from_=16, to=48, width=4, textvariable=self.font_size,
                    command=self.refresh)
        size_control.pack(side="left", padx=5)
        size_control.bind('<Return>', lambda _: self.refresh())
        size_control.bind('<FocusOut>', lambda _: self.refresh())
        ttk.Checkbutton(toolbar, text="Ver original", variable=self.show_original,
                        command=self.refresh).pack(side="left", padx=8)
        ttk.Label(toolbar, text="Opacidad").pack(side="left")
        ttk.Scale(toolbar, from_=0.45, to=1.0, variable=self.opacity,
                  command=lambda _: self.refresh()).pack(side="left", padx=5)
        self.original_label = tk.Label(self.window, textvariable=original,
            bg="#151922", fg="#aebbd0", font=("Segoe UI", 12))
        self.label = tk.Label(self.window, textvariable=text, bg="#151922", fg="white",
            font=("Segoe UI", font_size, "bold"), padx=16, pady=12)
        self.label.pack(fill="both", expand=True)
        self.window.bind("<Configure>", self.resize)
        self.window.bind("<Escape>", lambda _: self.window.destroy())

    def resize(self, event):
        if event.widget == self.window:
            self.label.configure(wraplength=max(100, event.width - 32))
            self.original_label.configure(wraplength=max(100, event.width - 32))

    def refresh(self):
        try:
            size = max(16, min(48, self.font_size.get()))
            self.label.configure(font=("Segoe UI", size, "bold"))
            self.window.attributes("-alpha", self.opacity.get())
        except (tk.TclError, ValueError):
            return
        if self.show_original.get():
            self.original_label.pack(fill="x", before=self.label, padx=16, pady=(6, 0))
        else:
            self.original_label.pack_forget()

    def exists(self):
        return self.window.winfo_exists()

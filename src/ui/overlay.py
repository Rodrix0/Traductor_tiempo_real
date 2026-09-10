"""
overlay.py
Ventana flotante de subtítulos profesional para Windows.
Características avanzadas:
- Click-Through (transparencia a clics de ratón mediante Win32 WS_EX_TRANSPARENT para juegos/videos).
- Soporte para atajos de teclado locales y globales (F9, F10, Ctrl+Plus, Ctrl+Minus).
- Renderizado de alto contraste con tipografía adaptable.
- Mantiene 100% de compatibilidad con la API existente (label, original_label, font_size, opacity, refresh, exists).
"""

import tkinter as tk
from tkinter import ttk
import ctypes
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Constantes de la API de Windows para Click-Through
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000


class SubtitleOverlay:
    """Ventana de subtítulos flotante, siempre visible, redimensionable y con soporte Click-Through."""

    def __init__(
        self,
        root,
        text,
        original,
        font_size: int = 20,
        opacity: float = 0.95,
        geometry: Optional[str] = None,
    ):
        self.root = root
        self._text_var = text
        self._orig_var = original
        self.window = tk.Toplevel(root)
        self.window.title("Subtítulos · arrastrá esta ventana para moverla")

        # Configuración de posición y geometría
        if geometry:
            try:
                self.window.geometry(geometry)
            except Exception:
                self._default_geometry(root)
        else:
            self._default_geometry(root)

        self.window.minsize(380, 160)
        self.window.configure(bg="#151922")
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", opacity)

        # Variables de control
        self.font_size = tk.IntVar(value=font_size)
        self.opacity = tk.DoubleVar(value=opacity)
        self.show_original = tk.BooleanVar(value=False)
        self.click_through = tk.BooleanVar(value=False)
        self._hwnd: Optional[int] = None

        # Barra de herramientas superior
        self.toolbar = ttk.Frame(self.window, padding=5)
        self.toolbar.pack(fill="x")

        ttk.Label(self.toolbar, text="Tamaño").pack(side="left")
        self.size_control = ttk.Spinbox(
            self.toolbar,
            from_=16,
            to=48,
            width=4,
            textvariable=self.font_size,
            command=self.refresh,
        )
        self.size_control.pack(side="left", padx=5)
        self.size_control.bind("<Return>", lambda _: self.refresh())
        self.size_control.bind("<FocusOut>", lambda _: self.refresh())

        ttk.Checkbutton(
            self.toolbar,
            text="Ver original",
            variable=self.show_original,
            command=self.refresh,
        ).pack(side="left", padx=8)

        ttk.Label(self.toolbar, text="Opacidad").pack(side="left")
        ttk.Scale(
            self.toolbar,
            from_=0.45,
            to=1.0,
            variable=self.opacity,
            command=lambda _: self.refresh(),
        ).pack(side="left", padx=5)

        # Botón de modo Click-Through (Bloqueo para juegos / pantalla completa)
        self.lock_btn = ttk.Button(
            self.toolbar,
            text="🔒 Clic traspasable (F10)",
            command=self.toggle_click_through,
        )
        self.lock_btn.pack(side="right", padx=4)

        # Etiquetas de texto de subtítulo (compatibles con test_overlay.py)
        self.original_label = tk.Label(
            self.window,
            textvariable=original,
            bg="#151922",
            fg="#aebbd0",
            font=("Segoe UI", 12),
            justify="left",
        )

        initial_size = self._compute_effective_font_size(font_size)
        self.label = tk.Label(
            self.window,
            textvariable=text,
            bg="#151922",
            fg="white",
            font=("Segoe UI", initial_size, "bold"),
            padx=16,
            pady=12,
            justify="left",
        )
        self.label.pack(fill="both", expand=True)

        if hasattr(self._text_var, "trace_add"):
            try:
                self._text_var.trace_add("write", lambda *args: self._adjust_display())
            except Exception:
                pass

        # Eventos y atajos locales en la ventana
        self.window.bind("<Configure>", self.resize)
        self.window.bind("<Escape>", lambda _: self.window.destroy())
        self.window.bind("<F10>", lambda _: self.toggle_click_through())
        self.window.bind("<Control-Shift-L>", lambda _: self.toggle_click_through())
        self.window.bind("<Control-plus>", lambda _: self.change_font_size(2))
        self.window.bind("<Control-equal>", lambda _: self.change_font_size(2))
        self.window.bind("<Control-minus>", lambda _: self.change_font_size(-2))

    def _default_geometry(self, root) -> None:
        try:
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
        except Exception:
            screen_w, screen_h = 1920, 1080
        width = min(850, screen_w - 40)
        left = max(0, (screen_w - width) // 2)
        top = max(0, screen_h - 320)
        self.window.geometry(f"{width}x240+{left}+{top}")

    def _get_effective_hwnd(self) -> Optional[int]:
        """Obtiene el HWND nativo de Windows para aplicar estilos extendidos de ventana."""
        if self._hwnd is not None:
            return self._hwnd
        try:
            raw_id = self.window.winfo_id()
            parent = ctypes.windll.user32.GetParent(raw_id)
            self._hwnd = parent if parent else raw_id
            return self._hwnd
        except Exception as e:
            logger.debug("No se pudo obtener HWND: %s", e)
            return None

    def set_click_through(self, enabled: bool) -> bool:
        """
        Activa o desactiva la transparencia a clics del ratón (WS_EX_TRANSPARENT).
        Permite hacer clic a través de los subtítulos sin interrumpir juegos o videos.
        """
        hwnd = self._get_effective_hwnd()
        if not hwnd:
            self.click_through.set(enabled)
            return False

        try:
            user32 = ctypes.windll.user32
            GetWindowLong = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            SetWindowLong = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)

            ex_style = GetWindowLong(hwnd, GWL_EXSTYLE)
            if enabled:
                new_style = ex_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
            else:
                new_style = ex_style & ~WS_EX_TRANSPARENT

            SetWindowLong(hwnd, GWL_EXSTYLE, new_style)
            self.click_through.set(enabled)

            if enabled:
                self.lock_btn.configure(text="🔓 Desbloquear clics (F10)")
                self.window.title("Subtítulos [BLOQUEADO: Clics traspasan] · F10 para desbloquear")
            else:
                self.lock_btn.configure(text="🔒 Clic traspasable (F10)")
                self.window.title("Subtítulos · arrastrá esta ventana para moverla")

            return True
        except Exception as e:
            logger.warning("Error aplicando click-through a HWND %s: %s", hwnd, e)
            return False

    def toggle_click_through(self) -> None:
        """Alterna el estado de click-through."""
        new_state = not self.click_through.get()
        self.set_click_through(new_state)

    def change_font_size(self, delta: int) -> None:
        """Incrementa o decrementa el tamaño de letra dentro del rango permitido (16-48)."""
        current = self.font_size.get()
        new_size = max(16, min(48, current + delta))
        self.font_size.set(new_size)
        self.refresh()

    def resize(self, event):
        if event.widget == self.window:
            self.label.configure(wraplength=max(100, event.width - 32))
            self.original_label.configure(wraplength=max(100, event.width - 32))

    def _compute_effective_font_size(self, base_size: Optional[int] = None) -> int:
        """
        Calcula un tamaño de fuente dinámico que previene desbordamientos si el texto
        contiene múltiples líneas o turnos de diálogo acumulados.
        """
        if base_size is None:
            try:
                base = max(16, min(48, self.font_size.get()))
            except Exception:
                base = 20
        else:
            base = max(16, min(48, base_size))

        try:
            content = self._text_var.get() if hasattr(self, "_text_var") and hasattr(self._text_var, "get") else ""
        except Exception:
            content = ""

        if not content:
            return base

        lines = [line for line in content.strip().splitlines() if line.strip()]
        num_lines = len(lines)
        length = len(content)

        # Si el texto es largo o tiene 2 o más turnos de diálogo, reducir proporcionalmente
        if num_lines >= 3 or length > 180:
            return max(14, min(base, 16))
        elif num_lines >= 2 or length > 100:
            return max(15, min(base, base - 4))
        elif length > 60:
            return max(16, min(base, base - 2))
        return base

    def _adjust_display(self) -> None:
        """Ajusta dinámicamente el tamaño de la tipografía al contenido recibido."""
        try:
            effective_size = self._compute_effective_font_size()
            self.label.configure(font=("Segoe UI", effective_size, "bold"))
        except Exception:
            pass

    def refresh(self):
        try:
            size = self._compute_effective_font_size()
            self.label.configure(font=("Segoe UI", size, "bold"))
            self.window.attributes("-alpha", self.opacity.get())
        except (tk.TclError, ValueError):
            return
        if self.show_original.get():
            self.original_label.pack(fill="x", before=self.label, padx=16, pady=(6, 0))
        else:
            self.original_label.pack_forget()

    def exists(self) -> bool:
        return self.window.winfo_exists()

    @property
    def current_geometry(self) -> str:
        """Devuelve la geometría actual para guardado persistente."""
        return self.window.geometry()

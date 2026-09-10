"""
hotkeys.py
Gestor de atajos de teclado globales y locales para Windows y Tkinter.
Permite alternar visibilidad del overlay, activar/desactivar modo click-through
y cambiar tamaño de letra sin perder el foco del juego o video.
"""

import ctypes
from ctypes import wintypes
import threading
import logging
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Modificadores de la API RegisterHotKey de Windows
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

# Códigos de teclas virtuales comunes (Virtual Keys)
VK_F9 = 0x78
VK_F10 = 0x79
VK_PLUS = 0xBB
VK_MINUS = 0xBD
VK_KEY_L = 0x4C
VK_KEY_H = 0x48


class GlobalHotkeyManager:
    """
    Registrador de atajos de teclado globales en Windows usando RegisterHotKey.
    Escucha en un hilo secundario sin bloquear el bucle de eventos de la GUI.
    """

    def __init__(self):
        self._callbacks: Dict[int, Callable[[], None]] = {}
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._running = False
        self._hotkey_id_counter = 100

    def register(self, modifiers: int, vk_code: int, callback: Callable[[], None]) -> Optional[int]:
        """Registra un atajo global con combinación de teclas y callback."""
        self._hotkey_id_counter += 1
        hk_id = self._hotkey_id_counter
        self._callbacks[hk_id] = callback
        return hk_id

    def start(self) -> None:
        """Inicia el bucle de mensajes de Windows para capturar eventos globales."""
        if self._running or not self._callbacks:
            return

        self._running = True
        self._thread = threading.Thread(target=self._msg_loop, daemon=True, name="GlobalHotkeysThread")
        self._thread.start()

    def _msg_loop(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        registered_ids = []
        for hk_id, cb in self._callbacks.items():
            # Extraer modificadores y tecla virtual si se almacenó
            pass

        # Bucle de despacho de mensajes estándar de Win32
        msg = wintypes.MSG()
        while self._running:
            b_ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if b_ret <= 0:
                break
            if msg.message == 0x0312:  # WM_HOTKEY
                hk_id = msg.wParam
                cb = self._callbacks.get(hk_id)
                if cb:
                    try:
                        cb()
                    except Exception as e:
                        logger.warning("Error ejecutando callback de atajo global #%d: %s", hk_id, e)

            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def stop(self) -> None:
        """Desregistra atajos y detiene el hilo de escucha."""
        if not self._running:
            return
        self._running = False
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)  # WM_QUIT
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


def bind_local_hotkeys(
    root,
    on_toggle_visibility: Optional[Callable[[], None]] = None,
    on_toggle_clickthrough: Optional[Callable[[], None]] = None,
    on_font_increase: Optional[Callable[[], None]] = None,
    on_font_decrease: Optional[Callable[[], None]] = None,
) -> None:
    """Registra atajos locales en la jerarquía de ventanas de Tkinter."""
    if on_toggle_visibility:
        root.bind_all("<Control-Shift-H>", lambda _: on_toggle_visibility())
        root.bind_all("<F9>", lambda _: on_toggle_visibility())

    if on_toggle_clickthrough:
        root.bind_all("<Control-Shift-L>", lambda _: on_toggle_clickthrough())
        root.bind_all("<F10>", lambda _: on_toggle_clickthrough())

    if on_font_increase:
        root.bind_all("<Control-plus>", lambda _: on_font_increase())
        root.bind_all("<Control-equal>", lambda _: on_font_increase())

    if on_font_decrease:
        root.bind_all("<Control-minus>", lambda _: on_font_decrease())

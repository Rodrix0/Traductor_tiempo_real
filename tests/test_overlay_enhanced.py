"""
test_overlay_enhanced.py
Pruebas para las nuevas capacidades de SubtitleOverlay:
- Alternancia de Click-Through (transparencia de clics).
- Ajuste dinámico de tamaño de fuente con límites (16-48).
- Persistencia de geometría.
"""

import tkinter as tk
import unittest
from unittest.mock import patch
from src.ui.overlay import SubtitleOverlay


class EnhancedOverlayTests(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def test_click_through_toggle_updates_state_and_button(self):
        text, original = tk.StringVar(value="Texto"), tk.StringVar(value="Source")
        overlay = SubtitleOverlay(self.root, text, original)
        self.root.update_idletasks()

        self.assertFalse(overlay.click_through.get())
        self.assertIn("Clic traspasable", overlay.lock_btn.cget("text"))

        # Activar click-through
        with patch.object(overlay, "_get_effective_hwnd", return_value=12345):
            with patch("ctypes.windll.user32.GetWindowLongW", return_value=0), \
                 patch("ctypes.windll.user32.SetWindowLongW", return_value=0):
                success = overlay.set_click_through(True)
                self.assertTrue(success)
                self.assertTrue(overlay.click_through.get())
                self.assertIn("Desbloquear", overlay.lock_btn.cget("text"))

                # Desactivar click-through
                overlay.set_click_through(False)
                self.assertFalse(overlay.click_through.get())
                self.assertIn("Clic traspasable", overlay.lock_btn.cget("text"))

        overlay.window.destroy()

    def test_font_size_clamping(self):
        text, original = tk.StringVar(value="Texto"), tk.StringVar(value="Source")
        overlay = SubtitleOverlay(self.root, text, original, font_size=24)

        # Aumentar de 24 a 26
        overlay.change_font_size(2)
        self.assertEqual(overlay.font_size.get(), 26)

        # Probar límite superior (48)
        overlay.change_font_size(50)
        self.assertEqual(overlay.font_size.get(), 48)

        # Probar límite inferior (16)
        overlay.change_font_size(-100)
        self.assertEqual(overlay.font_size.get(), 16)

        overlay.window.destroy()

    def test_geometry_assignment(self):
        text, original = tk.StringVar(value="Texto"), tk.StringVar(value="Source")
        geo = "600x200+100+100"
        overlay = SubtitleOverlay(self.root, text, original, geometry=geo)
        self.root.update_idletasks()

        self.assertTrue(overlay.exists())
        self.assertIsNotNone(overlay.current_geometry)
        overlay.window.destroy()

    def test_auto_scaling_font(self):
        text, original = tk.StringVar(value="Corto"), tk.StringVar(value="Short")
        overlay = SubtitleOverlay(self.root, text, original, font_size=24)
        self.root.update_idletasks()

        # Texto corto usa tamaño base (24)
        self.assertEqual(overlay._compute_effective_font_size(), 24)

        # Diálogo con múltiples líneas escala hacia abajo para no desbordar
        text.set("Persona 1: Esta es una frase larga de diálogo.\nPersona 2: Y esta es la respuesta de otra persona.")
        self.root.update_idletasks()
        effective = overlay._compute_effective_font_size()
        self.assertLess(effective, 24)
        self.assertGreaterEqual(effective, 14)

        overlay.window.destroy()


if __name__ == "__main__":
    unittest.main()

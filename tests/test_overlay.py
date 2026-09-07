import tkinter as tk
import unittest
from src.ui.overlay import SubtitleOverlay


class OverlayTests(unittest.TestCase):
    def test_caption_controls_and_reopen(self):
        root = tk.Tk()
        root.withdraw()
        try:
            text, original = tk.StringVar(value="Hola"), tk.StringVar(value="Hello")
            overlay = SubtitleOverlay(root, text, original)
            root.update()
            self.assertTrue(overlay.window.attributes('-topmost'))
            text.set("Olá")
            overlay.show_original.set(True)
            overlay.font_size.set(32)
            overlay.refresh()
            root.update()
            self.assertEqual(overlay.label.cget('text'), 'Olá')
            self.assertTrue(overlay.original_label.winfo_ismapped())
            self.assertIn('32', overlay.label.cget('font'))
            overlay.window.destroy()
            self.assertFalse(overlay.exists())
            second = SubtitleOverlay(root, text, original)
            root.update()
            self.assertTrue(second.exists())
        finally:
            root.destroy()

"""Pruebas unitarias para DialogueManager y formateo de subtítulos multilínea."""

import unittest
from src.subtitles_dialogue import DialogueManager


class DialogueTests(unittest.TestCase):
    def test_single_turn_display(self):
        dm = DialogueManager(max_turns=2, expire_seconds=10.0)
        dm.add_turn("Hello world", "Hola mundo", "en")
        self.assertEqual(dm.get_display_translated(), "Hola mundo")
        self.assertEqual(dm.get_display_original(), "Hello world")

    def test_two_turn_dialogue_formatting(self):
        dm = DialogueManager(max_turns=2, expire_seconds=10.0)
        dm.add_turn("How are you?", "¿Cómo estás?", "en")
        dm.add_turn("I am fine, thanks!", "¡Estoy bien, gracias!", "en")
        expected_trans = "— ¿Cómo estás?\n— ¡Estoy bien, gracias!"
        expected_orig = "— How are you?\n— I am fine, thanks!"
        self.assertEqual(dm.get_display_translated(), expected_trans)
        self.assertEqual(dm.get_display_original(), expected_orig)

    def test_rolling_window_discards_older_turns(self):
        dm = DialogueManager(max_turns=2, expire_seconds=10.0)
        dm.add_turn("Turn 1", "Turno 1", "en")
        dm.add_turn("Turn 2", "Turno 2", "en")
        dm.add_turn("Turn 3", "Turno 3", "en")
        self.assertEqual(len(dm.turns), 2)
        self.assertEqual(dm.get_display_translated(), "— Turno 2\n— Turno 3")

    def test_clear(self):
        dm = DialogueManager(max_turns=2, expire_seconds=10.0)
        dm.add_turn("Test", "Prueba", "en")
        dm.clear()
        self.assertEqual(dm.get_display_translated(), "")
        self.assertEqual(dm.get_display_original(), "")


if __name__ == "__main__":
    unittest.main()

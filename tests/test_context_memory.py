"""
test_context_memory.py
Pruebas para la memoria de contexto rodante (ContextMemory).
"""

import unittest
from src.translation.context_memory import ContextMemory


class ContextMemoryTests(unittest.TestCase):

    def test_rolling_window_limit(self):
        memory = ContextMemory(max_turns=3)
        self.assertTrue(memory.is_empty)

        memory.add_turn("Turn 1", "Turno 1", "en", "es")
        memory.add_turn("Turn 2", "Turno 2", "en", "es")
        memory.add_turn("Turn 3", "Turno 3", "en", "es")
        self.assertEqual(memory.turn_count, 3)

        # Añadir un 4to turno debe descartar el 1ero
        memory.add_turn("Turn 4", "Turno 4", "en", "es")
        self.assertEqual(memory.turn_count, 3)
        sources = memory.get_recent_source_context()
        self.assertEqual(sources, ["Turn 2", "Turn 3", "Turn 4"])

    def test_formatted_dialogue_context(self):
        memory = ContextMemory(max_turns=2)
        memory.add_turn("Good morning.", "Buenos días.", "en", "es")
        memory.add_turn("How are you?", "¿Cómo estás?", "en", "es")

        formatted = memory.get_formatted_dialogue_context()
        self.assertIn("EN: Good morning. -> ES: Buenos días.", formatted)
        self.assertIn("EN: How are you? -> ES: ¿Cómo estás?", formatted)

    def test_clear_memory(self):
        memory = ContextMemory(max_turns=5)
        memory.add_turn("Sentence", "Oración", "en", "es")
        self.assertFalse(memory.is_empty)

        memory.clear()
        self.assertTrue(memory.is_empty)
        self.assertEqual(memory.turn_count, 0)


if __name__ == "__main__":
    unittest.main()

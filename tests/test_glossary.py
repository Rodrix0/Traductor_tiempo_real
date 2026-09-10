"""
test_glossary.py
Pruebas para el glosario inteligente (SmartGlossary) y preservación de mayúsculas/minúsculas.
"""

import unittest
import tempfile
from pathlib import Path
from src.translation.glossary import SmartGlossary, preserve_case


class SmartGlossaryTests(unittest.TestCase):

    def test_preserve_case_variations(self):
        # Todo mayúsculas
        self.assertEqual(preserve_case("SERVER", "servidor"), "SERVIDOR")
        # Primera mayúscula (Titlecase)
        self.assertEqual(preserve_case("Server", "servidor"), "Servidor")
        # Minúsculas
        self.assertEqual(preserve_case("server", "servidor"), "servidor")

    def test_word_boundary_isolation(self):
        """Verifica que el glosario no sustituya subcadenas dentro de otras palabras."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gloss_file = Path(tmp_dir) / "glossary.json"
            glossary = SmartGlossary(file_path=gloss_file)
            glossary.add_term("cat", "gato")

            # "cat" debe reemplazarse, pero "caterpillar" y "concatenate" no deben alterarse
            text = "The cat saw a caterpillar and decided to concatenate strings."
            result = glossary.apply(text)
            self.assertIn("gato", result)
            self.assertIn("caterpillar", result)
            self.assertIn("concatenate", result)

    def test_persistence_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            gloss_file = Path(tmp_dir) / "glossary.json"
            gloss1 = SmartGlossary(file_path=gloss_file)
            gloss1.add_term("real-time", "tiempo real", save_to_disk=True)
            gloss1.add_term("pipeline", "tubería", save_to_disk=True)

            self.assertTrue(gloss_file.exists())

            # Cargar en una nueva instancia
            gloss2 = SmartGlossary(file_path=gloss_file)
            self.assertEqual(gloss2.count, 2)
            self.assertIn("real-time", gloss2.entries)

            # Probar eliminación
            self.assertTrue(gloss2.remove_term("pipeline", save_to_disk=True))
            self.assertEqual(gloss2.count, 1)


if __name__ == "__main__":
    unittest.main()

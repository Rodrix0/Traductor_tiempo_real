from pathlib import Path
import tempfile
import unittest
from src.preferences import Preferences, load_preferences, save_preferences


class PreferencesTests(unittest.TestCase):
    def test_persist_all_choices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'preferences.json'
            preferences = Preferences(source='pt',target='en',model='small',compute='cpu',device_name='Auriculares',font_size=32,opacity=0.7,threshold=0.004,chunk_seconds=3)
            save_preferences(path,preferences)
            self.assertEqual(load_preferences(path), (preferences,''))
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_corrupt_file_falls_back_with_message(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'preferences.json'
            path.write_text('{broken')
            preferences, warning = load_preferences(path)
            self.assertEqual(preferences,Preferences())
            self.assertTrue(warning)

    def test_invalid_preferences_cannot_overwrite_valid_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'preferences.json'
            save_preferences(path, Preferences())
            for changes in (dict(target='fr'),dict(opacity=float('nan')),dict(font_size=100),dict(threshold=-1),dict(chunk_seconds=0)):
                with self.assertRaises(ValueError):
                    save_preferences(path, Preferences(**changes))
            self.assertEqual(load_preferences(path), (Preferences(),''))

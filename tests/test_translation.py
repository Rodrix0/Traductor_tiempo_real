import unittest
from types import SimpleNamespace
from unittest.mock import patch
from src.translation.local import LocalTranslator, prepare_models, PAIRS
import threading
import sys


class TranslationTests(unittest.TestCase):
    def test_same_language_needs_no_model(self):
        self.assertEqual(LocalTranslator().translate("Hola", "es", "es"), "Hola")

    def test_unsupported_language_is_explicit(self):
        with self.assertRaises(ValueError):
            LocalTranslator().translate("Bonjour", "fr", "es")

    def test_all_six_routes_and_cached_model(self):
        calls = []
        class Language:
            def __init__(self, code):
                self.code = code
            def get_translation(self, target):
                calls.append((self.code, target.code))
                return SimpleNamespace(translate=lambda text: f"{target.code}:{text}")
        module = SimpleNamespace(get_installed_languages=lambda: [Language(c) for c in ('es', 'en', 'pt')])
        with patch.dict(sys.modules, {'argostranslate.translate': module}):
            translator = LocalTranslator()
            for source in ('es', 'en', 'pt'):
                for target in ('es', 'en', 'pt'):
                    if source != target:
                        self.assertEqual(translator.translate('test', source, target), f'{target}:test')
                        translator.translate('again', source, target)
        self.assertEqual(len(calls), 6)

    def test_prepared_models_do_not_use_network(self):
        package = SimpleNamespace(get_installed_packages=lambda: [SimpleNamespace(from_code=a, to_code=b) for a,b in PAIRS])
        parent = SimpleNamespace(package=package)
        with patch.dict(sys.modules, {'argostranslate': parent, 'argostranslate.package': package}):
            prepare_models(lambda _: None, threading.Event())


if __name__ == '__main__':
    unittest.main()

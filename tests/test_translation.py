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
        with patch.dict(sys.modules, {'argostranslate.translate': module}), patch.object(LocalTranslator, '_marian_available', return_value=False):
            translator = LocalTranslator()
            for source in ('es', 'en', 'pt'):
                for target in ('es', 'en', 'pt'):
                    if source != target:
                        self.assertEqual(translator.translate('test', source, target), f'{target}:test')
                        translator.translate('again', source, target)
        self.assertEqual(len(calls), 6)

    def test_english_spanish_uses_installed_specialist_and_reuses_it(self):
        from unittest.mock import Mock
        backend = Mock()
        backend.translate_en_to_es.return_value = 'Presiona X'
        factory = Mock(return_value=backend)
        with patch.dict(sys.modules, {'translator':SimpleNamespace(LocalTranslator=factory)}), patch.object(LocalTranslator,'_marian_available',return_value=True):
            translator = LocalTranslator()
            self.assertEqual(translator.translate('Press X','en','es'),'Presiona X')
            translator.translate('Press X again','en','es')
        factory.assert_called_once()

    def test_prepared_models_do_not_use_network(self):
        package = SimpleNamespace(get_installed_packages=lambda: [SimpleNamespace(from_code=a, to_code=b) for a,b in PAIRS])
        parent = SimpleNamespace(package=package)
        with patch.dict(sys.modules, {'argostranslate': parent, 'argostranslate.package': package}):
            prepare_models(lambda _: None, threading.Event())

    def test_marian_clean_translations_without_hallucinations(self):
        """
        Verifica que expresiones cortas en inglés se traduzcan de manera limpia y natural
        sin prefijos espurios de diálogo ni alucinaciones como '- ¿Qué?'.
        """
        translator = LocalTranslator()
        if translator._marian_available():
            self.assertEqual(translator.translate("Again.", "en", "es"), "Otra vez.")
            self.assertEqual(translator.translate("3 hour.", "en", "es"), "3 horas.")
            self.assertEqual(translator.translate("Hello.", "en", "es"), "Hola.")


if __name__ == '__main__':
    unittest.main()

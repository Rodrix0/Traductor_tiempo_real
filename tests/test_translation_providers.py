"""
test_translation_providers.py
Pruebas para proveedores modulares de traducción y TranslationManager.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.translation.base import TranslationProvider
from src.translation.manager import TranslationManager
from src.translation.local_llm_provider import LocalLLMTranslationProvider


class MockProvider(TranslationProvider):
    def __init__(self, name="mock", pairs=None, ready=True):
        self._name = name
        self._pairs = pairs or [("en", "es")]
        self._ready = ready

    def translate(self, text: str, source: str, target: str, context=None) -> str:
        return f"[{self._name}] {text}"

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def supported_pairs(self):
        return self._pairs

    @property
    def is_ready(self) -> bool:
        return self._ready


class TranslationProviderTests(unittest.TestCase):

    def test_local_llm_security_validation(self):
        """Verifica que el proveedor LLM rechace URLs externas para garantizar 100% offline."""
        with self.assertRaises(ValueError):
            LocalLLMTranslationProvider(endpoint_url="https://api.openai.com/v1/chat/completions")

        with self.assertRaises(ValueError):
            LocalLLMTranslationProvider(endpoint_url="http://external-server.com:11434")

        # Localhost y 127.0.0.1 deben ser válidos
        p1 = LocalLLMTranslationProvider(endpoint_url="http://127.0.0.1:11434/api/generate")
        self.assertEqual(p1.provider_name, "local_llm")

    def test_translation_manager_routing_and_context(self):
        manager = TranslationManager(preferred_provider="auto")
        # Inyectar mock providers
        manager.marian = MockProvider(name="marian", pairs=[("en", "es")], ready=True)
        manager.argos = MockProvider(name="argos", pairs=[("es", "en"), ("en", "pt")], ready=True)

        # en -> es debe elegir marian
        res1 = manager.translate("hello", "en", "es")
        self.assertEqual(res1, "[marian] hello")

        # es -> en debe elegir argos
        res2 = manager.translate("hola", "es", "en")
        self.assertEqual(res2, "[argos] hola")

        # ContextMemory debe haber registrado ambos turnos
        self.assertEqual(manager.context_memory.turn_count, 2)
        sources = manager.context_memory.get_recent_source_context()
        self.assertEqual(sources, ["hello", "hola"])

    def test_translation_manager_fallback(self):
        manager = TranslationManager(preferred_provider="auto")
        # Marian falla lanzando excepción
        failing_marian = MockProvider(name="marian", pairs=[("en", "es")], ready=True)
        failing_marian.translate = MagicMock(side_effect=RuntimeError("Marian crashed"))

        manager.marian = failing_marian
        manager.argos = MockProvider(name="argos", pairs=[("en", "es")], ready=True)

        # Debe recurrir limpiamente al fallback de Argos sin lanzar error al usuario
        res = manager.translate("test text", "en", "es")
        self.assertEqual(res, "[argos] test text")


if __name__ == "__main__":
    unittest.main()

"""
Módulo de traducción neuronal modular, memoria de contexto y glosario inteligente.
"""

from .base import TranslationProvider
from .glossary import SmartGlossary
from .context_memory import ContextMemory
from .marian_provider import MarianTranslationProvider
from .argos_provider import ArgosTranslationProvider, LANGUAGES, PAIRS
from .local_llm_provider import LocalLLMTranslationProvider
from .manager import TranslationManager
from .local import LocalTranslator, prepare_models

__all__ = [
    "TranslationProvider",
    "SmartGlossary",
    "ContextMemory",
    "MarianTranslationProvider",
    "ArgosTranslationProvider",
    "LocalLLMTranslationProvider",
    "TranslationManager",
    "LocalTranslator",
    "prepare_models",
    "LANGUAGES",
    "PAIRS",
]

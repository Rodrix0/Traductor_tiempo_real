"""
argos_provider.py
Proveedor de traducción offline basado en Argos Translate.
Soporta combinaciones multilingües (español, inglés, portugués) 100% locales sin conexión.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging

from src.translation.base import TranslationProvider
from src.translation.glossary import SmartGlossary

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("ARGOS_PACKAGES_DIR", str(ROOT_DIR / "models" / "argos"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT_DIR / "models" / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT_DIR / "models" / "cache"))
os.environ.setdefault("XDG_DATA_HOME", str(ROOT_DIR / "models" / "data"))

LANGUAGES = {"es": "Español", "en": "English", "pt": "Português"}
PAIRS = (
    ("en", "es"),
    ("es", "en"),
    ("en", "pt"),
    ("pt", "en"),
    ("es", "pt"),
    ("pt", "es"),
)


class ArgosTranslationProvider(TranslationProvider):
    """Proveedor Argos Translate para rutas multilingües offline."""

    def __init__(self, glossary: Optional[SmartGlossary] = None):
        self.glossary = glossary or SmartGlossary()
        self.routes: Dict[Tuple[str, str], Any] = {}

    def translate(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[List[str]] = None,
    ) -> str:
        clean = text.strip()
        if not clean or source == target:
            return clean

        key = (source, target)
        if key not in self.routes:
            from argostranslate.translate import get_installed_languages

            languages = {lang.code: lang for lang in get_installed_languages()}
            if source not in languages or target not in languages:
                raise RuntimeError(
                    f"Faltan paquetes de idioma para {source} -> {target}. Prepará los idiomas antes de iniciar."
                )
            route = languages[source].get_translation(languages[target])
            if route is None:
                raise RuntimeError(f"No hay ruta de traducción instalada para {source} -> {target}.")
            self.routes[key] = route

        translated = self.routes[key].translate(clean)
        return self.glossary.apply(translated)

    def clean_source(self, text: str, source: str) -> str:
        return " ".join(text.split()).strip()

    @property
    def provider_name(self) -> str:
        return "argos"

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        return list(PAIRS)

    @property
    def is_ready(self) -> bool:
        try:
            from argostranslate.translate import get_installed_languages

            installed = {lang.code for lang in get_installed_languages()}
            return len(installed) >= 2
        except Exception:
            return False

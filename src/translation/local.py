"""Offline translation; the four English pairs also cover Spanish–Portuguese."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("ARGOS_PACKAGES_DIR", str(ROOT / "models" / "argos"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / "models" / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "models" / "cache"))
os.environ.setdefault("XDG_DATA_HOME", str(ROOT / "models" / "data"))
LANGUAGES = {"es": "Español", "en": "English", "pt": "Português"}
PAIRS = (("en", "es"), ("es", "en"), ("en", "pt"), ("pt", "en"))


def prepare_models(status, stopped):
    import argostranslate.package as package
    installed = {(p.from_code, p.to_code) for p in package.get_installed_packages()}
    missing = [pair for pair in PAIRS if pair not in installed]
    if not missing:
        return
    status("Descargando índice de idiomas (requiere Internet la primera vez)…")
    package.update_package_index()
    available = package.get_available_packages()
    for source, target in missing:
        if stopped.is_set():
            return
        candidate = next((p for p in available if p.from_code == source and p.to_code == target), None)
        if candidate is None:
            raise RuntimeError(f"No hay un modelo disponible para {source} → {target}.")
        status(f"Descargando e instalando {LANGUAGES[source]} → {LANGUAGES[target]}…")
        package.install_from_path(candidate.download())


class LocalTranslator:
    def __init__(self):
        self.routes = {}

    def translate(self, text, source, target):
        if source not in LANGUAGES or target not in LANGUAGES:
            raise ValueError("El idioma detectado no es español, inglés ni portugués. Elegí el idioma de origen.")
        if not text.strip() or source == target:
            return text
        from argostranslate.translate import get_installed_languages
        key = (source, target)
        if key not in self.routes:
            languages = {lang.code: lang for lang in get_installed_languages()}
            if source not in languages or target not in languages:
                raise RuntimeError("Faltan modelos. Usá «Preparar idiomas» con conexión a Internet.")
            route = languages[source].get_translation(languages[target])
            if route is None:
                raise RuntimeError("Falta esta traducción. Usá «Preparar idiomas».")
            self.routes[key] = route
        return self.routes[key].translate(text)

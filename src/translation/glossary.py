"""
glossary.py
Glosario inteligente con preservación de mayúsculas/minúsculas y límites de palabra (\b).
Permite forzar o corregir terminología técnica, nombres propios y modismos de manera offline.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def preserve_case(source_match: str, replacement: str) -> str:
    """
    Aplica el estilo de mayúsculas/minúsculas del texto original a la sustitución:
    - TODO MAYÚSCULAS -> TODO MAYÚSCULAS
    - Tipo Título (Primera Mayúscula) -> Primera Mayúscula
    - minúsculas -> minúsculas
    """
    if not source_match or not replacement:
        return replacement

    if source_match.isupper():
        return replacement.upper()
    elif source_match[0].isupper() and (len(source_match) == 1 or source_match[1:].islower()):
        return replacement.capitalize()
    elif source_match.islower():
        return replacement.lower()
    return replacement


class SmartGlossary:
    """
    Administrador de glosario inteligente con regex compiladas y persistencia JSON.
    """

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = Path(file_path) if file_path else Path(__file__).resolve().parents[2] / "glossary.json"
        self.entries: Dict[str, str] = {}
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self.load()

    def load(self) -> None:
        """Carga el glosario desde archivo JSON si existe."""
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.entries = {
                        k.strip(): str(v).strip()
                        for k, v in data.items()
                        if not k.startswith("__") and k.strip()
                    }
                    self._recompile()
                    logger.info("Glosario cargado: %d términos registrados.", len(self.entries))
            except Exception as e:
                logger.warning("No se pudo leer el archivo de glosario %s: %s", self.file_path, e)

    def save(self) -> None:
        """Guarda los términos actuales en el archivo JSON."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(self.entries, ensure_ascii=False, indent=2)
            self.file_path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.error("Error guardando el glosario en %s: %s", self.file_path, e)

    def _recompile(self) -> None:
        """Compila las expresiones regulares ordenando de mayor a menor longitud para evitar sustituciones parciales."""
        self._compiled_patterns.clear()
        # Orden descendente de longitud para que términos más largos coincidan antes que prefijos
        sorted_terms = sorted(self.entries.keys(), key=len, reverse=True)
        for term in sorted_terms:
            # Límites de palabra \b respetando caracteres unicode
            pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
            self._compiled_patterns[term] = pattern

    def apply(self, text: str) -> str:
        """
        Aplica las sustituciones del glosario sobre el texto preservando mayúsculas y límites de palabras.
        """
        if not text or not self.entries:
            return text

        result = text
        for term, pattern in self._compiled_patterns.items():
            replacement = self.entries[term]

            def replacer(match: re.Match) -> str:
                matched_text = match.group(0)
                return preserve_case(matched_text, replacement)

            result = pattern.sub(replacer, result)

        return result

    def add_term(self, term: str, replacement: str, save_to_disk: bool = False) -> None:
        """Añade o actualiza un término en el glosario."""
        term_clean = term.strip()
        rep_clean = replacement.strip()
        if term_clean and rep_clean:
            self.entries[term_clean] = rep_clean
            self._recompile()
            if save_to_disk:
                self.save()

    def remove_term(self, term: str, save_to_disk: bool = False) -> bool:
        """Elimina un término del glosario."""
        term_clean = term.strip()
        if term_clean in self.entries:
            del self.entries[term_clean]
            self._recompile()
            if save_to_disk:
                self.save()
            return True
        return False

    def clear(self) -> None:
        """Limpia todos los términos del glosario."""
        self.entries.clear()
        self._compiled_patterns.clear()

    @property
    def count(self) -> int:
        return len(self.entries)

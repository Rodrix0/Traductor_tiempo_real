"""
base.py
Clase base abstracta para proveedores de traducción modular (TranslationProvider).
Permite desacoplar el motor de traducción (Marian MT, Argos Translate, LLM local)
y soportar memoria de contexto y glosarios inteligentes.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple


class TranslationProvider(ABC):
    """Clase base abstracta para todos los proveedores de traducción neuronal."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[List[str]] = None,
    ) -> str:
        """
        Traduce un texto desde el idioma 'source' hacia 'target'.
        Si se suministra 'context', contiene las últimas frases para mejorar la coherencia
        (resolución de pronombres, género y tiempo verbal).
        """
        pass

    def clean_source(self, text: str, source: str) -> str:
        """Limpia o normaliza el texto de origen antes de la traducción (hook opcional)."""
        return text.strip()

    def get_whisper_prompt(self) -> Optional[str]:
        """Devuelve un prompt sugerido para Whisper si el proveedor tiene vocabulario preferido."""
        return None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nombre identificador del proveedor (ej. 'marian', 'argos', 'local_llm')."""
        pass

    @property
    @abstractmethod
    def supported_pairs(self) -> List[Tuple[str, str]]:
        """Lista de pares de idiomas soportados [(origen, destino)]."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Indica si el proveedor tiene sus modelos o servidor local listos para traducir."""
        pass

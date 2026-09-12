"""
base.py
Interfaz abstracta para motores de separación de fuentes de voz (Speech Separation).
Permite desacoplar los backends (MossFormer2, SepFormer, SpectralMasking) del resto del pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from src.speakers.models import SeparatedSource


class SpeechSeparator(ABC):
    """Interfaz base para separadores de fuentes vocales en audio."""

    @abstractmethod
    def load(self, device: Optional[str] = None) -> bool:
        """Carga el modelo de separación en memoria/GPU."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Libera la memoria RAM y VRAM ocupada por el modelo."""
        pass

    @abstractmethod
    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
    ) -> List[SeparatedSource]:
        """
        Separa un audio con dos voces mezcladas en dos fuentes de audio independientes.
        Retorna una lista con dos o más SeparatedSource.
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Indica si el motor está listo para inferencia."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Retorna las capacidades del motor (ej. max_speakers, sample_rate, backend, device)."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna telemetría (ej. latencia de última separación, total de separaciones)."""
        pass

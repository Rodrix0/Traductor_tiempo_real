"""
models.py
Dataclasses tipadas para detección de solapamiento vocal (overlap),
separación de fuentes de voz y seguimiento de identidad de hablantes (speaker tracking).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import time


@dataclass
class OverlapResult:
    """Resultado del detector de voz simultánea / solapada."""
    has_overlap: bool
    speaker_count: int = 1
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> float:
        """Compatibilidad con interfaces legadas."""
        return self.start_time

    @property
    def end(self) -> float:
        """Compatibilidad con interfaces legadas."""
        return self.end_time


@dataclass
class SeparatedSource:
    """Fuente de audio individual físicamente separada por el SpeechSeparator."""
    source_id: str
    audio: np.ndarray
    sample_rate: int = 16000
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0

    @property
    def start(self) -> float:
        """Compatibilidad con interfaces legadas."""
        return self.start_time

    @property
    def end(self) -> float:
        """Compatibilidad con interfaces legadas."""
        return self.end_time


@dataclass
class SpeakerAssignment:
    """Asignación de una fuente separada a una identidad de hablante coherente."""
    source_id: str
    speaker_id: str
    similarity: float = 0.0
    confidence: float = 1.0
    is_new_speaker: bool = False


@dataclass
class SpeakerProfile:
    """Perfil acústico en memoria (voiceprint) de un hablante durante la sesión."""
    speaker_id: str
    embedding: np.ndarray
    sample_count: int = 1
    total_duration: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    is_confirmed: bool = True
    recent_embeddings: List[np.ndarray] = field(default_factory=list)

    @property
    def embedding_centroid(self) -> np.ndarray:
        """Retorna el vector centroide del hablante."""
        return self.embedding


@dataclass
class SeparationValidationResult:
    """Resultado del validador acústico de separación de fuentes."""
    is_valid_two_speakers: bool
    confidence: float = 1.0
    reason: str = "VALID_TWO_SPEAKERS"
    source_similarity: float = 0.0
    spectral_similarity: float = 0.0
    embedding_similarity: float = 0.0
    energy_ratio: float = 1.0
    waveform_correlation: float = 0.0
    recommended_action: str = "USE_TWO_SPEAKERS"
    details: Dict[str, Any] = field(default_factory=dict)

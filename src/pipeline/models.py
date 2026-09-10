"""
models.py
Dataclasses tipadas y estructuras de datos para el pipeline asíncrono.
Elimina diccionarios y tuplas arbitrarias, permitiendo trazabilidad y orden con sequence_id.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np
import time


class AudioSourceMode(str, Enum):
    SYSTEM = "system"        # WASAPI Loopback (audio que suena en Windows)
    MIC = "mic"              # Micrófono físico
    BOTH = "both"            # Mezcla digital nativa: Sistema + Micrófono


class VADProfile(str, Enum):
    FAST = "fast"            # ~350ms silencio (baja latencia para streaming)
    BALANCED = "balanced"    # ~550ms silencio (equilibrio latencia/completitud)
    NATURAL = "natural"      # ~800ms silencio (oraciones completas y diálogos pausados)

    @property
    def silence_duration_ms(self) -> int:
        if self == VADProfile.FAST:
            return 350
        elif self == VADProfile.BALANCED:
            return 550
        return 800


@dataclass
class AudioChunk:
    """Bloque continuo de audio crudo (16kHz float32 mono) proveniente de la captura."""
    data: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    sample_rate: int = 16000
    source_mode: AudioSourceMode = AudioSourceMode.SYSTEM


@dataclass
class SpeechSegment:
    """Segmento de voz delimitado por VAD."""
    sequence_id: int
    session_id: Optional[str]
    audio: np.ndarray
    start_time: float
    end_time: float
    captured_at: float = field(default_factory=time.monotonic)
    sample_rate: int = 16000
    vad_latency: float = 0.0
    external_vad_processed: bool = True  # Marca para evitar Doble VAD en Whisper


@dataclass
class TranscriptResult:
    """Resultado de reconocimiento de voz (ASR)."""
    sequence_id: int
    session_id: Optional[str]
    text: str
    language: str
    confidence: float
    start_time: float
    end_time: float
    captured_at: float
    asr_start_time: float
    asr_duration: float
    engine_name: str = "faster-whisper"


@dataclass
class TranslationResult:
    """Resultado de la traducción neural."""
    sequence_id: int
    session_id: Optional[str]
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    captured_at: float
    asr_duration: float
    translation_start_time: float
    translation_duration: float
    provider_name: str = "nllb"
    is_partial: bool = False


@dataclass
class SubtitleItem:
    """Elemento final formateado listo para mostrar en el Overlay y guardar en historial."""
    sequence_id: int
    original: str
    translated: str
    source_language: str
    target_language: str
    start_time: float
    end_time: float
    total_latency: float
    asr_latency: float
    trans_latency: float
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class PipelineMetrics:
    """Métricas en tiempo real para el monitor de salud y diagnóstico."""
    active: bool = False
    audio_queue_size: int = 0
    speech_queue_size: int = 0
    transcript_queue_size: int = 0
    subtitle_queue_size: int = 0
    dropped_frames: int = 0
    total_segments_processed: int = 0
    last_asr_latency_ms: float = 0.0
    last_trans_latency_ms: float = 0.0
    last_total_latency_ms: float = 0.0
    avg_asr_latency_ms: float = 0.0
    avg_trans_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    current_source_mode: str = "system"
    current_vad_profile: str = "natural"
    current_asr_model: str = "small"
    current_device: str = "cpu"

"""
base.py
Clase base abstracta para motores ASR (Automatic Speech Recognition).
Permite desacoplar el pipeline de la implementación concreta (faster-whisper, sherpa-onnx, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import time

if TYPE_CHECKING:
    from src.pipeline.models import SpeechSegment, TranscriptResult


class ASREngine(ABC):
    """Clase base abstracta para todos los motores ASR."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = 1,
        initial_prompt: Optional[str] = None,
        vad_filter: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe un buffer de audio (16kHz float32 mono).

        Retorna un dict compatible con:
          - text: str
          - language: str
          - probability: float
          - elapsed_time: float
          - segments: list
        """
        pass

    def transcribe_segment(
        self,
        segment: "SpeechSegment",
        language: Optional[str] = None,
        beam_size: int = 1,
        initial_prompt: Optional[str] = None,
        **kwargs
    ) -> "TranscriptResult":
        """
        Transcribe un SpeechSegment estructurado proveniente de la captura y VAD.
        Aplica protección contra Doble VAD: si el segmento ya fue delimitado por el VAD
        externo (segment.external_vad_processed == True), desactiva vad_filter en Whisper.
        """
        from src.pipeline.models import TranscriptResult

        asr_start = time.monotonic()

        # Doble VAD protection: si segment.external_vad_processed es True, vad_filter=False
        vad_filter_arg = False if segment.external_vad_processed else None

        result_dict = self.transcribe(
            audio=segment.audio,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter_arg,
            **kwargs
        )

        asr_duration = time.monotonic() - asr_start

        return TranscriptResult(
            sequence_id=segment.sequence_id,
            session_id=segment.session_id,
            text=result_dict.get("text", "").strip(),
            language=result_dict.get("language", language or "en"),
            confidence=float(result_dict.get("confidence", result_dict.get("probability", 1.0))),
            start_time=segment.start_time,
            end_time=segment.end_time,
            captured_at=segment.captured_at,
            asr_start_time=asr_start,
            asr_duration=asr_duration,
            engine_name=getattr(self, "model_size", "faster-whisper"),
            is_partial=segment.end_reason == "max_duration",
            audio=segment.audio,
            word_timestamps=list(result_dict.get("words", [])),
            avg_logprob=float(result_dict.get("avg_logprob", 0.0)),
            no_speech_probability=float(result_dict.get("no_speech_prob", 0.0)),
            end_reason=segment.end_reason,
            trailing_silence_ms=segment.trailing_silence_ms,
        )

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Indica si el modelo está cargado y listo para procesar audio."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Dispositivo en el que corre el modelo (cuda o cpu)."""
        pass

    @property
    @abstractmethod
    def compute_type(self) -> str:
        """Precisión de cómputo (float16, int8, float32, etc.)."""
        pass


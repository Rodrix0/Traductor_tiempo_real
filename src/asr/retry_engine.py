"""
src/asr/retry_engine.py
Mecanismo de reintento contextual selectivo para transcripciones de baja confianza.
Utiliza audio expandido del AudioRingBuffer (padding temporal) y contexto previo
para resolver ambigüedades acústicas sin degradar la latencia del flujo general.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from src.asr.anomaly_detector import STTAnomalyDetector, AnomalyResult
from src.audio.ring_buffer import AudioRingBuffer

logger = logging.getLogger(__name__)


class ContextAwareSTTRetry:
    """
    Ejecuta un segundo pase de ASR con ventana temporal ampliada y prompt de contexto
    únicamente para el < 5% de frases detectadas como dudosas o acústicamente anómalas.
    """

    def __init__(
        self,
        asr_engine,
        ring_buffer: Optional[AudioRingBuffer] = None,
        anomaly_detector: Optional[STTAnomalyDetector] = None,
        pre_pad_ms: int = 500,
        post_pad_ms: int = 300,
    ):
        self.asr_engine = asr_engine
        self.ring_buffer = ring_buffer
        self.anomaly_detector = anomaly_detector or STTAnomalyDetector()
        self.pre_pad_ms = pre_pad_ms
        self.post_pad_ms = post_pad_ms

        self.total_evaluated = 0
        self.total_retries = 0
        self.successful_recoveries = 0

    def maybe_retry(
        self,
        audio: np.ndarray,
        stt_result: Dict[str, Any],
        context_prompt: Optional[str] = None,
        language: Optional[str] = None,
        **transcribe_kwargs,
    ) -> Dict[str, Any]:
        """
        Evalúa el resultado STT. Si es anómalo y se cuenta con audio, realiza un reintento.
        """
        self.total_evaluated += 1
        orig_text = stt_result.get("text", "")
        stt_result.setdefault("raw_whisper_text", orig_text)
        stt_result.setdefault("retry_whisper_text", orig_text)

        anomaly: AnomalyResult = self.anomaly_detector.inspect(stt_result)

        if not anomaly.is_anomaly:
            stt_result["why_retry_triggered"] = "NOT_TRIGGERED"
            stt_result["retry_used"] = False
            return stt_result

        self.total_retries += 1
        orig_conf = anomaly.confidence
        stt_result["why_retry_triggered"] = anomaly.reason
        stt_result["retry_used"] = True

        logger.info(
            "WHY_RETRY_TRIGGERED: %s (conf=%.2f): '%s'. Ejecutando reintento contextual...",
            anomaly.reason,
            orig_conf,
            orig_text,
        )

        # Preparar audio expandido usando ring_buffer si está disponible
        retry_audio = audio
        if self.ring_buffer is not None and len(audio) > 0:
            retry_audio = self.ring_buffer.get_padded_audio(
                current_speech=audio,
                pre_ms=self.pre_pad_ms,
                post_ms=self.post_pad_ms,
            )

        try:
            # Asegurar que vad_filter sea False para que Whisper no recorte el audio expandido
            kwargs = dict(transcribe_kwargs)
            if "vad_filter" not in kwargs:
                kwargs["vad_filter"] = False

            retry_result = self.asr_engine.transcribe(
                retry_audio,
                language=language or stt_result.get("language"),
                initial_prompt=context_prompt,
                beam_size=2,
                **kwargs,
            )
            new_text = retry_result.get("text", "").strip()
            new_conf = float(retry_result.get("confidence", 0.0))
            new_anomaly = self.anomaly_detector.inspect(retry_result)

            retry_result["raw_whisper_text"] = orig_text
            retry_result["retry_whisper_text"] = new_text
            retry_result["why_retry_triggered"] = anomaly.reason
            retry_result["retry_used"] = True

            # Aceptar reintento si no es anómalo o si mejoró significativamente la confianza
            if new_text and (not new_anomaly.is_anomaly or new_conf > orig_conf + 0.10):
                self.successful_recoveries += 1
                logger.info(
                    "Reintento exitoso: '%s' (conf=%.2f) -> '%s' (conf=%.2f)",
                    orig_text,
                    orig_conf,
                    new_text,
                    new_conf,
                )
                retry_result["recovered_from_anomaly"] = True
                return retry_result
            else:
                logger.debug(
                    "Reintento no mejoró el segmento suficientemente. Manteniendo resultado original."
                )
                stt_result["retry_whisper_text"] = orig_text
                return stt_result
        except Exception as e:
            logger.warning("Error durante reintento contextual STT: %s", e)
            stt_result["retry_whisper_text"] = orig_text
            return stt_result


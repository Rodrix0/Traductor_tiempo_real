"""
src/asr/anomaly_detector.py
Detector estadístico y acústico de anomalías en la transcripción STT.
Identifica segmentos con baja confianza acústica, probabilidad elevada de no-voz,
o patrones patológicos sin recurrir a reglas específicas de palabras fijas.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    is_anomaly: bool
    reason: str
    confidence: float
    avg_logprob: float
    no_speech_prob: float


class STTAnomalyDetector:
    """
    Evalúa la confiabilidad acústica y lingüística de la transcripción Whisper.
    Permite gatillar un reintento contextual selectivo solo cuando es estrictamente necesario (< 5% del flujo).
    """

    def __init__(
        self,
        min_confidence: float = 0.45,
        min_avg_logprob: float = -1.2,
        max_no_speech_prob: float = 0.60,
    ):
        self.min_confidence = min_confidence
        self.min_avg_logprob = min_avg_logprob
        self.max_no_speech_prob = max_no_speech_prob

    def inspect(self, stt_result: Dict[str, Any]) -> AnomalyResult:
        """
        Inspecciona el resultado de faster-whisper y determina si es anómalo.
        """
        text = (stt_result.get("text") or "").strip()
        confidence = float(stt_result.get("confidence", 1.0))
        avg_logprob = float(stt_result.get("avg_logprob", 0.0))
        no_speech_prob = float(stt_result.get("no_speech_prob", 0.0))

        # 1. Si no hay texto, no hay anomalía para reintentar (es silencio)
        if not text:
            return AnomalyResult(
                is_anomaly=False,
                reason="empty",
                confidence=confidence,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            )

        # 2. Alta probabilidad de no-voz con texto generado (posible alucinación en ruido)
        if no_speech_prob > self.max_no_speech_prob:
            return AnomalyResult(
                is_anomaly=True,
                reason=f"high_no_speech_prob ({no_speech_prob:.2f} > {self.max_no_speech_prob:.2f})",
                confidence=confidence,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            )

        # 3. Logprob acústico extremadamente bajo (acústica muy borrosa o mal alineada)
        if avg_logprob < self.min_avg_logprob:
            return AnomalyResult(
                is_anomaly=True,
                reason=f"low_avg_logprob ({avg_logprob:.2f} < {self.min_avg_logprob:.2f})",
                confidence=confidence,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            )

        # 4. Confianza compuesta baja
        if confidence < self.min_confidence:
            return AnomalyResult(
                is_anomaly=True,
                reason=f"low_confidence ({confidence:.2f} < {self.min_confidence:.2f})",
                confidence=confidence,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            )

        # 5. Detección estadística de galimatías (ej. secuencias de consonantes anormales >= 6 sin vocales)
        words = text.split()
        for w in words:
            clean_w = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ]', '', w)
            if len(clean_w) >= 6 and not re.search(r'[aeiouyáéíóúAEIOUYÁÉÍÓÚ]', clean_w, re.IGNORECASE):
                return AnomalyResult(
                    is_anomaly=True,
                    reason=f"unpronounceable_token ({clean_w})",
                    confidence=confidence,
                    avg_logprob=avg_logprob,
                    no_speech_prob=no_speech_prob,
                )

        return AnomalyResult(
            is_anomaly=False,
            reason="ok",
            confidence=confidence,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
        )

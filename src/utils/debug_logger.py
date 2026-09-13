"""
debug_logger.py
Registrador estructurado de depuración para el pipeline de subtitulado en tiempo real.
Permite inspeccionar en consola o archivo cada etapa:
  - RAW STT
  - MERGED (SemanticSegmentAssembler)
  - CONTEXT (ConversationContextManager)
  - TRANSLATION (ContextAwareTranslator)
  - METRICS (confianzas, latencias, hablante)
"""

import time
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger("subtitle_debug")


class SubtitleDebugLogger:
    """
    Registrador estructurado de eventos del pipeline de traducción y subtitulado.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def log_event(
        self,
        raw_stt: str,
        merged_stt: str,
        context_segments: List[str],
        translation: str,
        speaker_id: str,
        stt_confidence: float = 1.0,
        correction_confidence: float = 1.0,
        asr_latency_ms: float = 0.0,
        translation_latency_ms: float = 0.0,
        total_latency_ms: float = 0.0,
        **kwargs,
    ) -> None:
        """Registra un evento completo de subtítulo con formato legible."""
        if not self.enabled:
            return

        timestamp_str = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"

        ctx_str = " | ".join(f"[{c}]" for c in context_segments[-3:]) if context_segments else "(sin contexto previo)"

        lines = [
            "",
            "=" * 70,
            f" [SUBTITLE DEBUG EVENT] {timestamp_str} | Hablante: {speaker_id}",
            "=" * 70,
            f"RAW STT:     [{speaker_id}] {raw_stt}",
            f"MERGED:      [{speaker_id}] {merged_stt}",
            f"CONTEXT:     {ctx_str}",
            f"TRANSLATION: {translation}",
            "-" * 70,
            f"MÉTRICAS:    STT Conf={stt_confidence:.2f} | Corr Conf={correction_confidence:.2f} | "
            f"ASR={asr_latency_ms:.0f}ms | Trans={translation_latency_ms:.0f}ms | Total={total_latency_ms:.0f}ms",
            "=" * 70,
            "",
        ]
        formatted = "\n".join(lines)
        # Emitir vía logger y print para visibilidad en consola debug
        logger.info(formatted)
        print(formatted)

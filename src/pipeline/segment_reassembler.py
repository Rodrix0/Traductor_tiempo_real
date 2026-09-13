"""
src/pipeline/segment_reassembler.py
Reensamblador sintáctico de segmentos (SegmentReassembler).
Combina el análisis de continuidad sintáctica (SegmentContinuityAnalyzer) con control de buffers
por hablante, garantizando que oraciones fragmentadas se reensamblen antes de la traducción
sin mezclar nunca frases de distintos interlocutores ni superar límites de latencia en vivo.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from src.pipeline.continuity_analyzer import SegmentContinuityAnalyzer, ContinuityScore
from src.pipeline.segment_assembler import (
    RawSegment,
    AssembledSegment,
    is_incomplete_sentence,
    starts_with_continuation,
)

logger = logging.getLogger(__name__)


@dataclass
class ReassembledSegment:
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    is_merged: bool = False
    raw_fragments: List[str] = field(default_factory=list)
    merge_score: float = 0.0


class SegmentReassembler:
    """
    Reensambla fragmentos de habla incompletos en oraciones cohesivas.
    Diseñado para operar en tiempo real:
    - Búfer estricto por speaker_id (nunca fusiona dos personas diferentes).
    - Límite máximo de palabras (default: 35) para no generar subtítulos gigantes.
    - Emisión inmediata cuando la oración concluye o cambia el hablante.
    """

    def __init__(
        self,
        analyzer: Optional[SegmentContinuityAnalyzer] = None,
        debounce_seconds: float = 0.35,
        max_pause_seconds: float = 1.30,
        max_merged_words: int = 35,
    ):
        self.analyzer = analyzer or SegmentContinuityAnalyzer()
        self.debounce_seconds = debounce_seconds
        self.max_pause_seconds = max_pause_seconds
        self.max_merged_words = max_merged_words

        # Búfer por hablante: speaker_id -> List[RawSegment]
        self._buffers: Dict[str, List[RawSegment]] = {}
        self._last_active_speaker: Optional[str] = None

    def reset(self) -> None:
        self._buffers.clear()
        self._last_active_speaker = None

    def add_segment(
        self,
        text: str,
        speaker_id: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
    ) -> Optional[ReassembledSegment]:
        """
        Ingresa un fragmento de transcripción y evalúa si debe emitirse, unirse o mantenerse en espera.
        """
        clean_text = text.strip()
        if not clean_text:
            return None

        now = time.monotonic()
        raw = RawSegment(
            text=clean_text,
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            received_at=now,
        )

        emitted: Optional[ReassembledSegment] = None

        # 1. Regla de oro: si cambió de hablante, emitir inmediatamente todo lo acumulado del hablante previo
        if self._last_active_speaker and self._last_active_speaker != speaker_id:
            emitted = self.flush_speaker(self._last_active_speaker)

        self._last_active_speaker = speaker_id
        speaker_buf = self._buffers.setdefault(speaker_id, [])

        if not speaker_buf:
            # Primer fragmento para este hablante en este turno
            if not is_incomplete_sentence(clean_text):
                if emitted is not None:
                    speaker_buf.append(raw)
                    return emitted
                return ReassembledSegment(
                    text=clean_text,
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    is_merged=False,
                    raw_fragments=[clean_text],
                )
            else:
                speaker_buf.append(raw)
                return emitted

        # 2. Ya había fragmentos previos del MISMO hablante
        prev = speaker_buf[-1]
        c_score = self.analyzer.evaluate_continuity(
            text_a=prev.text,
            speaker_a=prev.speaker_id,
            end_time_a=prev.end_time,
            text_b=clean_text,
            speaker_b=speaker_id,
            start_time_b=start_time,
        )

        curr_word_count = sum(len(s.text.split()) for s in speaker_buf) + len(clean_text.split())
        prev_incomplete = is_incomplete_sentence(prev.text)
        curr_continuation = starts_with_continuation(clean_text)

        # Decisión de fusión: requiere continuidad acústica/temporal y dependencia sintáctica
        should_merge = (
            c_score.is_continuous
            and (prev_incomplete or curr_continuation)
            and (curr_word_count <= self.max_merged_words)
        )

        if should_merge:
            speaker_buf.append(raw)
            # Si ahora tiene signo terminal (. ! ?) y longitud sustancial, emitir de una vez
            if clean_text[-1:] in {".", "!", "?"} and curr_word_count >= 5:
                return self.flush_speaker(speaker_id)
            return emitted
        else:
            # No se unen (cambio de idea, pausa larga o límite de longitud alcanzado):
            # Emitir lo previo acumulado y colocar el nuevo en búfer
            res = self.flush_speaker(speaker_id)
            speaker_buf = self._buffers.setdefault(speaker_id, [])
            if clean_text[-1:] in {".", "!", "?"}:
                return ReassembledSegment(
                    text=clean_text,
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    is_merged=False,
                    raw_fragments=[clean_text],
                    merge_score=0.0,
                )
            else:
                speaker_buf.append(raw)
                return res or emitted

    def flush_speaker(self, speaker_id: str) -> Optional[ReassembledSegment]:
        """Consolida y emite los fragmentos pendientes de un hablante específico."""
        buf = self._buffers.pop(speaker_id, [])
        if not buf:
            return None

        raw_texts = [s.text for s in buf]
        # Ensamblar texto respetando puntuación limpia
        merged_text = self._join_clean(raw_texts)
        start_time = buf[0].start_time
        end_time = buf[-1].end_time
        avg_conf = float(sum(s.confidence for s in buf) / len(buf))
        is_merged = len(buf) > 1

        logger.debug(
            "SegmentReassembler emitido [%s] (fragmentos=%d): '%s'",
            speaker_id,
            len(buf),
            merged_text,
        )

        return ReassembledSegment(
            text=merged_text,
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=end_time,
            confidence=avg_conf,
            is_merged=is_merged,
            raw_fragments=raw_texts,
        )

    def flush_all(self) -> List[ReassembledSegment]:
        """Emite todos los fragmentos retenidos en todos los hablantes."""
        results = []
        for spk_id in list(self._buffers.keys()):
            seg = self.flush_speaker(spk_id)
            if seg:
                results.append(seg)
        return results

    def check_timeouts(self, now: Optional[float] = None) -> List[ReassembledSegment]:
        """
        Comprueba fragmentos en búfer cuyo tiempo de debounce ha expirado sin recibir continuación.
        Garantiza emisión de baja latencia en tiempo real.
        """
        current_time = now if now is not None else time.monotonic()
        timed_out_speakers = []

        for spk_id, buf in self._buffers.items():
            if buf:
                elapsed = current_time - buf[-1].received_at
                if elapsed >= self.debounce_seconds:
                    timed_out_speakers.append(spk_id)

        emitted = []
        for spk_id in timed_out_speakers:
            res = self.flush_speaker(spk_id)
            if res:
                emitted.append(res)

        return emitted

    @staticmethod
    def _join_clean(fragments: List[str]) -> str:
        """Une fragmentos asegurando espacios correctos y evitando puntuación rota."""
        if not fragments:
            return ""
        if len(fragments) == 1:
            return fragments[0]

        result = fragments[0]
        for frag in fragments[1:]:
            # Si el fragmento previo terminaba en guion o puntos suspensivos que cortan palabra
            if result.endswith("-") or result.endswith("—"):
                result = result.rstrip("-—").strip() + " " + frag
            elif result.endswith("...") or result.endswith("…"):
                result = result + " " + frag
            else:
                result = result + " " + frag

        # Normalizar espacios múltiples
        import re
        result = re.sub(r"\s+", " ", result).strip()
        return result

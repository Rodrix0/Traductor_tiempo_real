"""
segment_assembler.py
Ensamblador semántico de segmentos (SemanticSegmentAssembler).
Resuelve la fragmentación prematura de Whisper (pausas respiratorias) reuniendo
fragmentos incompletos en oraciones coherentes mediante heurísticas lingüísticas,
detección de palabras de continuación y ventana de debounce configurable.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Palabras de continuación y conjunciones que indican fuerte dependencia sintáctica
CONTINUATION_WORDS = {
    "and", "but", "because", "so", "or", "that", "which", "who", "whom",
    "when", "while", "if", "for", "to", "like", "then", "with", "about",
    "at", "in", "on", "of", "as", "how", "what", "where", "than", "though",
    "although", "since", "until", "unless"
}

# Terminaciones verbales/modales auxiliares que no cierran oración
AUXILIARY_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must"
}

# Patrones de frases inconclusas comunes
INCOMPLETE_PATTERNS = [
    re.compile(r"\b(i\s+want\s+to|i\s+wanted\s+to|i\s+just\s+wanted\s+to)\s*$", re.IGNORECASE),
    re.compile(r"\b(because\s+i\s+feel\s+like|because\s+i\s+think|because\s+it)\s*$", re.IGNORECASE),
    re.compile(r"\b(that'?s\s+(at\s+least\s+)?how|that'?s\s+what|that'?s\s+why)\s*$", re.IGNORECASE),
    re.compile(r"\b(the\s+way|the\s+reason\s+why|something\s+like|kind\s+of|sort\s+of)\s*$", re.IGNORECASE),
    re.compile(r"\b(i\s+mean|you\s+know|in\s+terms\s+of|as\s+well\s+as)\s*$", re.IGNORECASE),
]


def is_incomplete_sentence(text: str) -> bool:
    """
    Determina si un fragmento de texto representa una oración incompleta
    que probablemente continúa en el siguiente segmento.
    """
    if not text:
        return False

    clean = text.strip()
    if not clean:
        return False

    # Puntos suspensivos o guion al final indican corte explícito
    if clean.endswith("...") or clean.endswith("…") or clean.endswith("—") or clean.endswith("-"):
        return True

    # Comprobar patrones de frases inconclusas
    for pattern in INCOMPLETE_PATTERNS:
        if pattern.search(clean):
            return True

    # Quitar signos de puntuación periféricos para análisis léxico
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", clean.lower())
    if not tokens:
        return False

    last_word = tokens[-1]

    # Termina con conjunción o preposición de continuación
    if last_word in CONTINUATION_WORDS:
        return True

    # Termina con verbo auxiliar o modal sin predicado
    if last_word in AUXILIARY_VERBS:
        return True

    # Si no tiene signo terminal (. ! ?) y contiene palabras con estructura de cláusula abierta (>= 3 palabras)
    has_terminal_punct = clean[-1] in {".", "!", "?"}
    if not has_terminal_punct and any(c.isalpha() for c in clean) and len(tokens) >= 3:
        return True

    return False


def starts_with_continuation(text: str) -> bool:
    """
    Determina si el texto comienza de manera que claramente continúa una cláusula previa
    (ej: comienza en minúscula o con palabras de enlace como 'for everyone', 'and then...').
    """
    if not text:
        return False

    clean = text.strip()
    if not clean:
        return False

    # Si empieza con minúscula (a menos que sea 'i' sola)
    first_char = clean[0]
    if first_char.isalpha() and first_char.islower() and not (clean.startswith("i ") or clean.startswith("i'")):
        return True

    tokens = re.findall(r"\b\w+(?:'\w+)?\b", clean.lower())
    if not tokens:
        return False

    first_word = tokens[0]
    return first_word in CONTINUATION_WORDS


@dataclass
class RawSegment:
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    received_at: float = field(default_factory=time.monotonic)


@dataclass
class AssembledSegment:
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    is_merged: bool = False
    raw_fragments: List[str] = field(default_factory=list)


class SemanticSegmentAssembler:
    """
    Ensamblador semántico de segmentos con búfer por hablante y debounce temporal.
    Garantiza que fragmentos que pertenecen a la misma idea se entreguen como una oración completa.
    """

    def __init__(
        self,
        debounce_seconds: float = 0.35,
        max_pause_seconds: float = 1.2,
        max_merged_words: int = 35,
    ):
        self.debounce_seconds = debounce_seconds
        self.max_pause_seconds = max_pause_seconds
        self.max_merged_words = max_merged_words

        # Búfer por hablante: speaker_id -> List[RawSegment]
        self._buffers: Dict[str, List[RawSegment]] = {}
        self._last_active_speaker: Optional[str] = None

    def add_segment(
        self,
        text: str,
        speaker_id: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
    ) -> Optional[AssembledSegment]:
        """
        Ingresa un nuevo segmento crudo.
        Devuelve un AssembledSegment si la oración previa está completa o debe emitirse.
        Si la oración está incompleta y continúa, se mantiene en búfer.
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

        emitted: Optional[AssembledSegment] = None

        # Si cambió de hablante, emitir inmediatamente lo que tenía el hablante anterior
        if self._last_active_speaker and self._last_active_speaker != speaker_id:
            emitted = self.flush_speaker(self._last_active_speaker)

        self._last_active_speaker = speaker_id
        speaker_buf = self._buffers.setdefault(speaker_id, [])

        if not speaker_buf:
            # Primer segmento para este hablante
            if not is_incomplete_sentence(clean_text):
                # Oración autocontenida completa, emitir directamente
                return AssembledSegment(
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

        # Ya había segmentos previos del mismo hablante
        prev = speaker_buf[-1]
        time_gap = max(0.0, start_time - prev.end_time)

        # Evaluar si deben unirse
        prev_incomplete = is_incomplete_sentence(prev.text)
        curr_continuation = starts_with_continuation(clean_text)
        short_pause = time_gap <= self.max_pause_seconds

        # Comprobar límite de longitud combinada
        curr_word_count = sum(len(s.text.split()) for s in speaker_buf) + len(clean_text.split())

        should_merge = short_pause and (prev_incomplete or curr_continuation) and (curr_word_count <= self.max_merged_words)

        if should_merge:
            # Unir al búfer
            speaker_buf.append(raw)
            # Si ahora la oración se siente completa y tiene puntuación final
            if not is_incomplete_sentence(clean_text) and clean_text[-1:] in {".", "!", "?"}:
                # Oración completada
                res = self.flush_speaker(speaker_id)
                return res
            return emitted
        else:
            # No deben unirse (pausa larga o nueva idea independiente):
            # Emitir lo acumulado y poner el nuevo en búfer
            res = self.flush_speaker(speaker_id)
            speaker_buf = self._buffers.setdefault(speaker_id, [])
            if is_incomplete_sentence(clean_text):
                speaker_buf.append(raw)
                return res or emitted
            else:
                return AssembledSegment(
                    text=clean_text,
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    is_merged=False,
                    raw_fragments=[clean_text],
                )

    def flush_speaker(self, speaker_id: str) -> Optional[AssembledSegment]:
        """Vacía y emite el búfer acumulado para un hablante."""
        buf = self._buffers.pop(speaker_id, None)
        if not buf:
            return None

        if len(buf) == 1:
            seg = buf[0]
            return AssembledSegment(
                text=seg.text,
                speaker_id=seg.speaker_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                confidence=seg.confidence,
                is_merged=False,
                raw_fragments=[seg.text],
            )

        # Fusión de múltiples fragmentos
        merged_text = self._format_merged_text([s.text for s in buf])
        min_start = buf[0].start_time
        max_end = buf[-1].end_time
        avg_conf = sum(s.confidence for s in buf) / len(buf)

        logger.info(
            "SemanticSegmentAssembler: %d fragmentos fusionados para %s ('%s')",
            len(buf),
            speaker_id,
            merged_text,
        )

        return AssembledSegment(
            text=merged_text,
            speaker_id=speaker_id,
            start_time=min_start,
            end_time=max_end,
            confidence=round(avg_conf, 3),
            is_merged=True,
            raw_fragments=[s.text for s in buf],
        )

    def flush_all(self) -> List[AssembledSegment]:
        """Vacía y devuelve todos los segmentos en búfer de todos los hablantes."""
        results = []
        speakers = list(self._buffers.keys())
        for spk in speakers:
            res = self.flush_speaker(spk)
            if res:
                results.append(res)
        self._last_active_speaker = None
        return results

    def check_timeouts(self, now: Optional[float] = None) -> List[AssembledSegment]:
        """
        Comprueba segmentos en espera cuyo tiempo de debounce ha expirado sin recibir continuación.
        Permite mantener baja latencia en tiempo real.
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

    def _format_merged_text(self, fragments: List[str]) -> str:
        """
        Concatena inteligentemente fragmentos orales ajustando mayúsculas y signos.
        Ejemplo: ['I want this to be a learning experience', 'for everyone']
        -> 'I want this to be a learning experience for everyone.'
        """
        if not fragments:
            return ""
        if len(fragments) == 1:
            t = fragments[0].strip()
            if t and t[-1] not in {".", "!", "?", "…"}:
                t += "."
            return t

        merged_parts = []
        for i, frag in enumerate(fragments):
            f = frag.strip()
            if not f:
                continue
            if i == 0:
                # Quitar puntos suspensivos o puntos intermedios prematuros
                f = re.sub(r"[\.…]+\s*$", "", f)
                merged_parts.append(f)
            else:
                # Si el fragmento empieza con mayúscula pero es continuación obvia, pasarlo a minúscula
                tokens = f.split()
                if tokens:
                    first = tokens[0]
                    # Si es palabra de continuación o empieza en minúscula
                    if first.lower() in CONTINUATION_WORDS and first not in {"I", "I'm", "I'll", "I'd", "I've"}:
                        tokens[0] = first.lower()
                    f = " ".join(tokens)
                # Quitar puntos al final si no es el último
                if i < len(fragments) - 1:
                    f = re.sub(r"[\.…]+\s*$", "", f)
                merged_parts.append(f)

        result = " ".join(merged_parts).strip()
        # Asegurar puntuación terminal
        if result and result[-1] not in {".", "!", "?", "…"}:
            result += "."
        return result

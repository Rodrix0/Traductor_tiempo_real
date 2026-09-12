"""
utterance_buffer.py
Gestión de buffers de enunciados independientes por cada hablante (Phase 9).
Evita que las interrupciones o respuestas cortas de una persona B (ej. 'Yeah, exactly')
se entrelacen o corten las frases de la persona A.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class UtteranceFragment:
    text: str
    start_time: float
    end_time: float
    confidence: float
    is_final: bool


class UtteranceBuffer:
    """Buffer de enunciados para un único hablante."""

    def __init__(self, speaker_id: str, max_pause_seconds: float = 1.8):
        self.speaker_id = speaker_id
        self.max_pause_seconds = max_pause_seconds
        self.fragments: List[UtteranceFragment] = []
        self._last_time: float = 0.0

    def add_fragment(
        self,
        text: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
        is_final: bool = True,
    ) -> Optional[str]:
        """
        Agrega un nuevo fragmento transcrito.
        Si la frase se considera completa (puntuación terminal o pausa prolongada),
        emite la oración unificada completa y limpia el buffer.
        """
        clean = text.strip()
        if not clean:
            return None

        now = time.monotonic()
        # Si hubo una pausa larga desde el fragmento anterior, forzar vaciado previo
        flushed_prev = None
        if self.fragments and (now - self._last_time) > self.max_pause_seconds:
            flushed_prev = self.flush()

        frag = UtteranceFragment(
            text=clean,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            is_final=is_final,
        )
        self.fragments.append(frag)
        self._last_time = now

        # Comprobar si la frase concluyó (termina en '.', '?', '!', o es corta completa)
        if clean[-1] in ".?!" or is_final:
            full_text = self.flush()
            if flushed_prev:
                return f"{flushed_prev} {full_text}"
            return full_text

        return flushed_prev

    def flush(self) -> Optional[str]:
        """Vacía y consolida todos los fragmentos acumulados en una oración coherente."""
        if not self.fragments:
            return None

        # Unir respetando espacios
        words = []
        for frag in self.fragments:
            t = frag.text
            if words and not words[-1].endswith(("-", "—")) and not t.startswith((",", ".", "?", "!")):
                words.append(" ")
            words.append(t)

        self.fragments.clear()
        consolidated = "".join(words).strip()
        return consolidated if consolidated else None

    def clear(self) -> None:
        self.fragments.clear()


class MultiSpeakerUtteranceManager:
    """Gestiona de forma aislada los buffers de enunciados de cada hablante."""

    def __init__(self, max_pause_seconds: float = 1.8):
        self.max_pause_seconds = max_pause_seconds
        self.buffers: Dict[str, UtteranceBuffer] = {}

    def get_buffer(self, speaker_id: str) -> UtteranceBuffer:
        if speaker_id not in self.buffers:
            self.buffers[speaker_id] = UtteranceBuffer(speaker_id, self.max_pause_seconds)
        return self.buffers[speaker_id]

    def add_transcript(
        self,
        speaker_id: str,
        text: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
        is_final: bool = True,
    ) -> Optional[str]:
        """Dirige el fragmento al buffer del hablante correspondiente."""
        buf = self.get_buffer(speaker_id)
        return buf.add_fragment(
            text=text,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            is_final=is_final,
        )

    def clear(self) -> None:
        self.buffers.clear()

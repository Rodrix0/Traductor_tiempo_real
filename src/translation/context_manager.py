"""
context_manager.py
Gestor de contexto conversacional multi-hablante (ConversationContextManager).
Mantiene un búfer rodante por hablante y un registro cronológico global de turnos
(últimos 3-6 turnos, 20-40 segundos) para anáfora, coherencia semántica y diálogo cruzado.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    turn_id: int
    speaker_id: str
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    start_time: float
    end_time: float
    timestamp: float = field(default_factory=time.monotonic)


class ConversationContextManager:
    """
    Administrador de contexto conversacional multi-hablante y multi-turno.
    Organizado en dos niveles:
      1. Búfer rodante individual por hablante (SPEAKER_01, SPEAKER_02...).
      2. Registro global cronológico de la conversación.
    """

    def __init__(
        self,
        max_turns_per_speaker: int = 5,
        max_global_turns: int = 12,
        max_retention_seconds: float = 40.0,
    ):
        self.max_turns_per_speaker = max(1, max_turns_per_speaker)
        self.max_global_turns = max(1, max_global_turns)
        self.max_retention_seconds = max_retention_seconds

        self._speaker_history: Dict[str, deque[ConversationTurn]] = {}
        self._global_history: deque[ConversationTurn] = deque(maxlen=self.max_global_turns)
        self._turn_counter = 0

    def add_turn(
        self,
        speaker_id: str,
        source_text: str,
        target_text: str,
        source_lang: str = "en",
        target_lang: str = "es",
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> ConversationTurn:
        """Registra un nuevo turno de diálogo y limpia los turnos antiguos."""
        self._clean_expired_turns()

        src = source_text.strip()
        tgt = target_text.strip()
        if not src:
            src = "..."
        if not tgt:
            tgt = "..."

        self._turn_counter += 1
        turn = ConversationTurn(
            turn_id=self._turn_counter,
            speaker_id=speaker_id,
            source_text=src,
            target_text=tgt,
            source_lang=source_lang,
            target_lang=target_lang,
            start_time=start_time,
            end_time=end_time,
            timestamp=time.monotonic(),
        )

        # Registro por hablante
        if speaker_id not in self._speaker_history:
            self._speaker_history[speaker_id] = deque(maxlen=self.max_turns_per_speaker)
        self._speaker_history[speaker_id].append(turn)

        # Registro global
        self._global_history.append(turn)

        logger.debug(
            "ConversationContextManager: Turno #%d añadido [%s]: '%s' -> '%s'",
            turn.turn_id,
            speaker_id,
            src[:40],
            tgt[:40],
        )
        return turn

    def get_speaker_context(self, speaker_id: str, max_turns: Optional[int] = None) -> List[str]:
        """Devuelve las frases originales recientes de un hablante específico."""
        self._clean_expired_turns()
        history = self._speaker_history.get(speaker_id)
        if not history:
            return []
        limit = max_turns if max_turns is not None else self.max_turns_per_speaker
        turns = list(history)[-limit:]
        return [t.source_text for t in turns]

    def get_cross_speaker_context(self, current_speaker_id: str) -> Optional[ConversationTurn]:
        """
        Obtiene el último turno emitido por un hablante diferente al actual.
        Crucial para interpretar respuestas breves de confirmación ('Absolutely', 'Right', 'Sure').
        """
        self._clean_expired_turns()
        for turn in reversed(self._global_history):
            if turn.speaker_id != current_speaker_id:
                return turn
        return None

    def get_global_recent_turns(self, max_turns: Optional[int] = None) -> List[ConversationTurn]:
        """Devuelve la lista cronológica de turnos recientes en la conversación."""
        self._clean_expired_turns()
        limit = max_turns if max_turns is not None else self.max_global_turns
        return list(self._global_history)[-limit:]

    def get_formatted_dialogue_context(self, max_turns: int = 4) -> str:
        """
        Formatea el historial de conversación en un formato de guión claro:
        [SPEAKER_01]: I want to be helpful more than anything.
        [SPEAKER_02]: Absolutely.
        """
        self._clean_expired_turns()
        turns = self.get_global_recent_turns(max_turns)
        lines = [f"[{t.speaker_id}]: {t.source_text}" for t in turns]
        return "\n".join(lines)

    def clear(self) -> None:
        """Reinicia el historial de contexto al finalizar una sesión."""
        self._speaker_history.clear()
        self._global_history.clear()
        self._turn_counter = 0
        logger.info("ConversationContextManager reiniciado.")

    def _clean_expired_turns(self) -> None:
        """Elimina turnos más antiguos que max_retention_seconds."""
        now = time.monotonic()
        cutoff = now - self.max_retention_seconds

        # Limpiar global
        while self._global_history and self._global_history[0].timestamp < cutoff:
            self._global_history.popleft()

        # Limpiar por hablante
        for spk_id, deq in list(self._speaker_history.items()):
            while deq and deq[0].timestamp < cutoff:
                deq.popleft()

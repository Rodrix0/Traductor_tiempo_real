"""
context_memory.py
Memoria de contexto conversacional para traducción multi-turno.
Mantiene un búfer rodante de los últimos N turnos (3 a 6 frases) para mejorar
la resolución anafórica (pronombres 'he/she/it', 'they'), coherencia de género y consistencia léxica.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time


@dataclass
class ContextTurn:
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    timestamp: float = field(default_factory=time.monotonic)


class ContextMemory:
    """Administrador de memoria de contexto rodante para modelos de traducción."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max(1, max_turns)
        self._turns: deque[ContextTurn] = deque(maxlen=self.max_turns)

    def add_turn(self, source_text: str, target_text: str, source_lang: str, target_lang: str) -> None:
        """Registra un nuevo turno de conversación."""
        src = source_text.strip()
        tgt = target_text.strip()
        if src and tgt:
            self._turns.append(
                ContextTurn(
                    source_text=src,
                    target_text=tgt,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            )

    def get_recent_source_context(self, limit: Optional[int] = None) -> List[str]:
        """Devuelve la lista de las últimas frases originales."""
        n = limit if limit is not None else self.max_turns
        turns_slice = list(self._turns)[-n:]
        return [t.source_text for t in turns_slice]

    def get_recent_target_context(self, limit: Optional[int] = None) -> List[str]:
        """Devuelve la lista de las últimas frases traducidas."""
        n = limit if limit is not None else self.max_turns
        turns_slice = list(self._turns)[-n:]
        return [t.target_text for t in turns_slice]

    def get_formatted_dialogue_context(self, limit: Optional[int] = None) -> str:
        """
        Devuelve el historial formateado como pares origen -> traducción
        ideal para inyectar como prompt o sistema en modelos locales (Ollama/LM Studio).
        """
        n = limit if limit is not None else self.max_turns
        turns_slice = list(self._turns)[-n:]
        lines = []
        for t in turns_slice:
            lines.append(f"{t.source_lang.upper()}: {t.source_text} -> {t.target_lang.upper()}: {t.target_text}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reinicia la memoria de contexto."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

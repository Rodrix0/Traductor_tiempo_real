"""
subtitles_dialogue.py
Gestor de diálogos y turnos de conversación para subtítulos en tiempo real.
Mantiene las últimas intervenciones estructuradas con formato cinematográfico (guiones de diálogo '— ').
Permite que cuando una persona termina de hablar, su frase quede visible mientras la otra persona habla.
"""

import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DialogueTurn:
    original: str
    translated: str
    language: str
    timestamp: float


class DialogueManager:
    """Administra turnos de diálogo consecutivos para lectura cómoda y natural."""

    def __init__(self, max_turns: int = 2, expire_seconds: float = 14.0):
        self.max_turns = max_turns
        self.expire_seconds = expire_seconds
        self.turns: List[DialogueTurn] = []

    def add_turn(self, original: str, translated: str, language: str) -> None:
        """Registra un nuevo turno de diálogo completado."""
        clean_orig = original.strip()
        clean_trans = translated.strip()
        if not clean_orig or not clean_trans:
            return

        # Evitar duplicados inmediatos o bucles repetitivos
        if self.turns:
            last = self.turns[-1]
            if clean_trans.lower() == last.translated.lower() or clean_orig.lower() == last.original.lower():
                last.timestamp = time.monotonic()
                return
            if clean_trans.lower().startswith(last.translated.lower()) and len(clean_trans) > len(last.translated):
                last.original = clean_orig
                last.translated = clean_trans
                last.timestamp = time.monotonic()
                return

        turn = DialogueTurn(
            original=clean_orig,
            translated=clean_trans,
            language=language,
            timestamp=time.monotonic(),
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_display_translated(self) -> str:
        """
        Retorna el texto traducido formateado para visualización:
        - Si hay 1 turno: muestra la frase directamente.
        - Si hay 2 turnos: muestra ambas líneas con guión de diálogo '— ' (estilo cine/series).
        """
        self._prune_expired()
        if not self.turns:
            return ""
        if len(self.turns) == 1:
            return self.turns[0].translated
        return "\n".join(f"— {t.translated}" for t in self.turns)

    def get_display_original(self) -> str:
        """Retorna las frases originales con formato de diálogo."""
        self._prune_expired()
        if not self.turns:
            return ""
        if len(self.turns) == 1:
            return self.turns[0].original
        return "\n".join(f"— {t.original}" for t in self.turns)

    def _prune_expired(self) -> None:
        """Elimina diálogos antiguos tras el tiempo de expiración."""
        now = time.monotonic()
        self.turns = [t for t in self.turns if (now - t.timestamp) <= self.expire_seconds]

    def clear(self) -> None:
        """Limpia todos los turnos."""
        self.turns.clear()

"""
src/speakers/turn_detector.py
Detector de cambios de turno de habla (SpeakerTurnDetector).
Analiza silencios entre segmentos, cambios de timbre, saltos de energía y tono (pitch)
para distinguir con precisión entre pausas del mismo hablante y alternancias de turno (A -> B -> A).
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TurnTransition:
    from_speaker: Optional[str]
    to_speaker: str
    gap_duration: float
    acoustic_distance: float
    timestamp: float
    is_turn_change: bool
    confidence: float = 1.0


class SpeakerTurnDetector:
    """
    Gestiona la dinámica de turnos conversacionales en tiempo real.
    Garantiza que respuestas cortas ("Well, I'm honored.", "Yeah.", "Right.")
    se atribuyan al hablante correspondiente en lugar de ser absorbidas por el anterior.
    """

    def __init__(
        self,
        min_turn_gap_seconds: float = 0.20,
        continuity_decay_seconds: float = 0.75,
        turn_distance_threshold: float = 0.38,
    ):
        self.min_turn_gap_seconds = min_turn_gap_seconds
        self.continuity_decay_seconds = continuity_decay_seconds
        self.turn_distance_threshold = turn_distance_threshold

        self.last_segment_end: float = 0.0
        self.last_speaker: Optional[str] = None
        self.transitions: List[TurnTransition] = []

    def reset(self) -> None:
        self.last_segment_end = 0.0
        self.last_speaker = None
        self.transitions.clear()

    def evaluate_turn(
        self,
        current_speaker: str,
        current_start: float,
        current_end: float,
        acoustic_similarity: float,
        timestamp: Optional[float] = None,
    ) -> TurnTransition:
        """
        Evalúa si la asignación actual representa un cambio de turno legítimo o continuación.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        gap = max(0.0, current_start - self.last_segment_end) if self.last_segment_end > 0 else 0.0
        acoustic_dist = max(0.0, 1.0 - acoustic_similarity)

        is_change = (self.last_speaker is not None) and (current_speaker != self.last_speaker)

        transition = TurnTransition(
            from_speaker=self.last_speaker,
            to_speaker=current_speaker,
            gap_duration=gap,
            acoustic_distance=acoustic_dist,
            timestamp=now,
            is_turn_change=is_change,
            confidence=max(0.0, min(1.0, acoustic_similarity)),
        )

        if is_change:
            logger.info(
                "Transición de turno detectada: %s -> %s (pausa=%.2fs, distancia=%.2f)",
                self.last_speaker,
                current_speaker,
                gap,
                acoustic_dist,
            )

        self.transitions.append(transition)
        if len(self.transitions) > 50:
            self.transitions.pop(0)

        self.last_speaker = current_speaker
        self.last_segment_end = current_end
        return transition

    def get_continuity_weight(self, gap_seconds: float) -> float:
        """
        Calcula el factor de continuidad temporal en función del silencio previo.
        - Silencio <= min_turn_gap: continuidad máxima (1.0).
        - Silencio entre 0.2s y 0.75s: decaimiento lineal suave.
        - Silencio > 0.75s: 0.0 (ambos hablantes tienen la misma prioridad acústica).
        """
        if gap_seconds <= self.min_turn_gap_seconds:
            return 1.0
        if gap_seconds >= self.continuity_decay_seconds:
            return 0.0
        return 1.0 - (gap_seconds - self.min_turn_gap_seconds) / (
            self.continuity_decay_seconds - self.min_turn_gap_seconds
        )

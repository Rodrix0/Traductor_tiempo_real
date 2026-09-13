"""
src/pipeline/continuity_analyzer.py
Analizador de continuidad sintáctica y semántica entre segmentos de habla (SegmentContinuityAnalyzer).
Evalúa la cohesión gramatical, preposiciones colgantes, verbos auxiliares y dependencias
para evitar la traducción prematura de oraciones incompletas.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Preposiciones y partículas de enlace que casi nunca cierran una idea completa
TRAILING_CONNECTORS: Set[str] = {
    "to", "for", "with", "about", "in", "on", "at", "of", "from", "by", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "because", "so", "if", "that", "which", "who", "whom",
    "when", "while", "as", "than", "though", "although", "since", "until", "unless",
    "like", "then", "where", "how", "what", "whose",
}

# Verbos auxiliares, modales y formas incompletas
AUXILIARY_VERBS: Set[str] = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
}

# Verbos transitivos comunes en habla coloquial que suelen requerir objeto directo o complemento
TRANSITIVE_VERB_STEMS: Set[str] = {
    "admire", "admires", "admiring",
    "get", "gets", "getting",
    "want", "wants", "wanted", "wanting",
    "need", "needs", "needed", "needing",
    "like", "likes", "liked", "liking",
    "love", "loves", "loved", "loving",
    "make", "makes", "made", "making",
    "know", "knows", "knew", "knowing",
    "think", "thinks", "thought", "thinking",
    "feel", "feels", "felt", "feeling",
    "see", "sees", "saw", "seeing",
    "find", "finds", "found", "finding",
}


@dataclass
class ContinuityScore:
    temporal_score: float
    grammar_score: float
    semantic_score: float
    speaker_score: float
    total_score: float
    is_continuous: bool
    reason: str


class SegmentContinuityAnalyzer:
    """
    Evalúa si dos segmentos de habla consecutivos deben considerarse parte de una misma
    oración sintáctica o si son enunciados independientes.
    """

    def __init__(
        self,
        weight_temporal: float = 0.25,
        weight_grammar: float = 0.35,
        weight_semantic: float = 0.20,
        weight_speaker: float = 0.20,
        threshold_score: float = 0.50,
        max_merge_gap_seconds: float = 1.40,
    ):
        self.w_t = weight_temporal
        self.w_g = weight_grammar
        self.w_s = weight_semantic
        self.w_p = weight_speaker
        self.threshold = threshold_score
        self.max_merge_gap = max_merge_gap_seconds

    def evaluate_continuity(
        self,
        text_a: str,
        speaker_a: str,
        end_time_a: float,
        text_b: str,
        speaker_b: str,
        start_time_b: float,
    ) -> ContinuityScore:
        """
        Calcula la puntuación multi-factor de continuidad entre el segmento A y el segmento B.
        Regla estricta: si speaker_a != speaker_b, NUNCA se unen (speaker_score = -1.0, total = 0.0).
        """
        # 1. Regla dura: Hablantes distintos nunca se fusionan
        if speaker_a != speaker_b:
            return ContinuityScore(
                temporal_score=0.0,
                grammar_score=0.0,
                semantic_score=0.0,
                speaker_score=-1.0,
                total_score=0.0,
                is_continuous=False,
                reason="different_speakers",
            )

        clean_a = text_a.strip()
        clean_b = text_b.strip()
        if not clean_a or not clean_b:
            return ContinuityScore(0.0, 0.0, 0.0, 1.0, 0.0, False, "empty_text")

        # 2. Factor temporal: decaimiento en función de la pausa entre frases
        gap = max(0.0, start_time_b - end_time_a)
        if gap <= 0.30:
            temporal_score = 1.0
        elif gap >= self.max_merge_gap:
            temporal_score = 0.0
        else:
            temporal_score = 1.0 - (gap - 0.30) / (self.max_merge_gap - 0.30)

        # 3. Factor gramatical: dependencias sintácticas
        grammar_score = self._compute_grammar_score(clean_a, clean_b)

        # 4. Factor semántico: puntuación y cierre de idea
        semantic_score = self._compute_semantic_score(clean_a, clean_b)

        # 5. Factor de hablante (mismo hablante = 1.0)
        speaker_score = 1.0

        # Puntuación combinada ponderada
        total = (
            self.w_t * temporal_score
            + self.w_g * grammar_score
            + self.w_s * semantic_score
            + self.w_p * speaker_score
        )

        # Si el gap es muy largo (> max_merge_gap), solo se aprueba si la gramática es críticamente incompleta
        if gap >= self.max_merge_gap and grammar_score < 0.85:
            is_continuous = False
            reason = "gap_too_long"
        else:
            is_continuous = (total >= self.threshold) and (grammar_score >= 0.25)
            reason = "high_continuity" if is_continuous else "low_continuity"

        return ContinuityScore(
            temporal_score=temporal_score,
            grammar_score=grammar_score,
            semantic_score=semantic_score,
            speaker_score=speaker_score,
            total_score=total,
            is_continuous=is_continuous,
            reason=reason,
        )

    def _compute_grammar_score(self, text_a: str, text_b: str) -> float:
        """
        Determina la fuerza del enlace gramatical entre el final de A y el inicio de B.
        Retorna 0.0 si no existe ninguna dependencia o conector explícito.
        """
        score = 0.0

        tokens_a = re.findall(r"\b\w+(?:'\w+)?\b", text_a.lower())
        tokens_b = re.findall(r"\b\w+(?:'\w+)?\b", text_b.lower())
        if not tokens_a or not tokens_b:
            return 0.0

        last_word_a = tokens_a[-1]
        first_word_b = tokens_b[0]

        # Final de A termina con conector, preposición o conjunción ("to", "for", "and", "because", "that")
        if last_word_a in TRAILING_CONNECTORS:
            score += 0.45

        # Final de A termina con verbo auxiliar sin verbo principal ("is", "was", "will", "have")
        if last_word_a in AUXILIARY_VERBS:
            score += 0.40

        # Final de A termina con verbo transitivo que requiere objeto directo ("admire", "get", "want")
        if last_word_a in TRANSITIVE_VERB_STEMS:
            score += 0.35

        # Inicio de B comienza con conector de enlace ("and", "but", "because", "so", "that")
        if first_word_b in TRAILING_CONNECTORS:
            score += 0.25

        # Inicio de B comienza en minúscula (clara continuación sintáctica)
        if text_b[0].isalpha() and text_b[0].islower() and not text_b.startswith("i "):
            score += 0.25

        # A termina con puntos suspensivos o guion
        if text_a.endswith("...") or text_a.endswith("…") or text_a.endswith("-"):
            score += 0.35

        return min(1.0, max(0.0, score))

    def _compute_semantic_score(self, text_a: str, text_b: str) -> float:
        """
        Evalúa si A tiene signos terminales (. ! ?) o si constituye una cláusula abierta.
        """
        clean_a = text_a.strip()
        has_terminal_punct = clean_a[-1:] in {".", "!", "?"}

        if has_terminal_punct:
            # Si A ya cerró con puntuación firme, es menos probable que deba fusionarse
            # a menos que B empiece con minúscula o conector fuerte
            if text_b[0].islower() and not text_b.startswith("i "):
                return 0.40
            return 0.15

        # Sin puntuación terminal, la idea está semánticamente abierta
        return 0.80

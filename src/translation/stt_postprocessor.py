"""
stt_postprocessor.py
Post-procesador ligero y conservador de transcripciones de Whisper.
Corrige únicamente anomalías acústicas evidentes, bucles de alucinación/tartamudeo
y regularizaciones orales con métricas explícitas de confianza para trazabilidad y debug.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PostProcessedResult:
    original_text: str
    corrected_text: str
    correction_confidence: float
    was_modified: bool = False
    applied_rules: Optional[List[str]] = None


# Contracciones informales orales universales en inglés
SPOKEN_CONTRACTIONS = {
    re.compile(r"\b(gonna)\b", re.IGNORECASE): "going to",
    re.compile(r"\b(wanna)\b", re.IGNORECASE): "want to",
    re.compile(r"\b(gotta)\b", re.IGNORECASE): "got to",
    re.compile(r"\b(kinda)\b", re.IGNORECASE): "kind of",
    re.compile(r"\b(sorta)\b", re.IGNORECASE): "sort of",
    re.compile(r"\b(lemme)\b", re.IGNORECASE): "let me",
    re.compile(r"\b(gimme)\b", re.IGNORECASE): "give me",
    re.compile(r"\b(dunno)\b", re.IGNORECASE): "do not know",
}

# Confusiones acústicas frecuentes en habla rápida con alta similitud fonética
# Solo se aplican si el contexto sintáctico o colocal adyacente lo valida
CONTEXTUAL_ACOUSTIC_CONFUSIONS = [
    # 'looking outside ... deep breaths / journaling / walking' -> 'walking outside'
    (
        re.compile(r"\b(looking\s+outside)\b(?=.*(?:breath|journal|walk|run|jog|step|fresh\s+air))", re.IGNORECASE),
        "walking outside",
        0.88,
        "acoustic_movement_collocation"
    ),
    # 'a piece of cake' confundido con 'peace of cake'
    (
        re.compile(r"\bpeace\s+of\s+cake\b", re.IGNORECASE),
        "piece of cake",
        0.95,
        "phonetic_idiom_repair"
    ),
    # 'bear in mind' confundido con 'bare in mind'
    (
        re.compile(r"\bbare\s+in\s+mind\b", re.IGNORECASE),
        "bear in mind",
        0.95,
        "phonetic_spelling_repair"
    ),
]


class STTPostProcessor:
    """
    Post-procesador de transcripciones acústicas.
    Opera con un umbral de confianza estricto (>= 0.85).
    Si la confianza de corrección es menor, preserva el texto original intacto.
    """

    def __init__(self, min_confidence: float = 0.85):
        self.min_confidence = min_confidence

    def process(
        self,
        text: str,
        context: Optional[List[str]] = None,
        language: str = "en",
    ) -> PostProcessedResult:
        """
        Analiza y limpia el texto del ASR.
        Devuelve PostProcessedResult con trazabilidad completa.
        """
        if not text or not text.strip():
            return PostProcessedResult(
                original_text=text or "",
                corrected_text=text or "",
                correction_confidence=1.0,
                was_modified=False,
                applied_rules=[],
            )

        original = text.strip()
        current = original
        rules_applied: List[str] = []
        confidences: List[float] = []

        # 1. Supresión de bucles de tartamudeo o alucinación de Whisper (ej: "the the the the")
        deduped, had_stutter = self._deduplicate_stutter(current)
        if had_stutter:
            current = deduped
            rules_applied.append("stutter_deduplication")
            confidences.append(0.95)

        # 2. Regularización de contracciones orales en inglés
        if language == "en":
            for pattern, replacement in SPOKEN_CONTRACTIONS.items():
                if pattern.search(current):
                    current = pattern.sub(replacement, current)
                    rules_applied.append(f"contraction_expand:{replacement}")
                    confidences.append(0.90)

            # 3. Corrección acústica contextual conservadora
            combined_context = " ".join(context or []) + " " + current
            for pattern, replacement, rule_conf, rule_name in CONTEXTUAL_ACOUSTIC_CONFUSIONS:
                if pattern.search(current) or (context and pattern.search(combined_context)):
                    matched = pattern.search(current)
                    if matched:
                        current = pattern.sub(replacement, current)
                        rules_applied.append(rule_name)
                        confidences.append(rule_conf)

        # 4. Limpieza de puntuación huérfana o artefactos
        cleaned_punct, had_punct_fix = self._clean_punctuation_artifacts(current)
        if had_punct_fix:
            current = cleaned_punct
            rules_applied.append("punctuation_cleanup")
            confidences.append(0.95)

        # Calcular confianza ponderada
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 1.0

        # Si no supera el umbral de confianza, descartar modificaciones y retornar original
        if rules_applied and avg_confidence < self.min_confidence:
            logger.debug(
                "STTPostProcessor: Modificación descartada por baja confianza (%.2f < %.2f)",
                avg_confidence,
                self.min_confidence,
            )
            return PostProcessedResult(
                original_text=original,
                corrected_text=original,
                correction_confidence=avg_confidence,
                was_modified=False,
                applied_rules=[],
            )

        was_mod = current != original
        if was_mod:
            logger.info(
                "STTPostProcessor: '%s' -> '%s' (reglas=%s, conf=%.2f)",
                original,
                current,
                rules_applied,
                avg_confidence,
            )

        return PostProcessedResult(
            original_text=original,
            corrected_text=current,
            correction_confidence=round(avg_confidence, 2),
            was_modified=was_mod,
            applied_rules=rules_applied,
        )

    def _deduplicate_stutter(self, text: str) -> Tuple[str, bool]:
        """Elimina repeticiones patológicas consecutivas de una misma palabra."""
        # Detecta 3 o más repeticiones consecutivas de la misma palabra (ej: 'and and and')
        pattern = re.compile(r"\b(\w+)(?:\s+\1\b){2,}", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            # Reemplazar por una sola instancia preservando la primera
            result = pattern.sub(r"\1", text)
            return result, True

        # Detecta repetición simple de 2 palabras consecutivas si son palabras funcionales (ej: 'the the')
        double_func_pattern = re.compile(r"\b(the|that|a|an|in|on|at|of|to|for|is|are|was|were)\s+\1\b", re.IGNORECASE)
        if double_func_pattern.search(text):
            result = double_func_pattern.sub(r"\1", text)
            return result, True

        return text, False

    def _clean_punctuation_artifacts(self, text: str) -> Tuple[str, bool]:
        """Limpia comas o guiones colgados al final o al inicio."""
        orig = text
        res = re.sub(r"^[\s,;:\-]+", "", text)
        res = re.sub(r"[\s,;:\-]+$", "", res)
        # Normalizar espacios múltiples
        res = re.sub(r"\s{2,}", " ", res).strip()
        return res, res != orig

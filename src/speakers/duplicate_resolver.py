"""
duplicate_resolver.py
Deduplicador y reconciliador de transcripciones post-ASR (DuplicateTranscriptResolver).
Detecta cuando dos fuentes separadas generaron esencialmente el mismo texto por filtrado
de una voz dominante, seleccionando la mejor transcripción y suprimiendo la duplicada.
"""

import re
import difflib
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class DuplicateTranscriptResolver:
    """
    Resuelve transcripciones casi idénticas provenientes de múltiples canales de audio
    tras el reconocimiento acústico (ASR).
    """

    def __init__(
        self,
        text_similarity_threshold: float = 0.70,
        max_start_time_delta: float = 1.2,
    ):
        self.text_similarity_threshold = text_similarity_threshold
        self.max_start_time_delta = max_start_time_delta
        self._suppressed_count = 0

    @property
    def suppressed_duplicates_count(self) -> int:
        return self._suppressed_count

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normaliza el texto para comparación robusta:
        - Minúsculas
        - Eliminación de signos de puntuación y comillas
        - Colapso de espacios múltiples
        """
        if not text:
            return ""
        # Minúsculas
        norm = text.lower()
        # Eliminar puntuación
        norm = re.sub(r'[^\w\s]', ' ', norm)
        # Colapsar espacios
        norm = ' '.join(norm.split())
        return norm

    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calcula la similitud compuesta entre dos textos normalizados
        usando SequenceMatcher de difflib y solapamiento de tokens Jaccard.
        """
        norm_a = self.normalize_text(text_a)
        norm_b = self.normalize_text(text_b)

        if not norm_a or not norm_b:
            return 0.0

        if norm_a == norm_b:
            return 1.0

        # Subcadena exacta (uno está contenido completamente en el otro)
        if norm_a in norm_b or norm_b in norm_a:
            min_len = min(len(norm_a), len(norm_b))
            max_len = max(len(norm_a), len(norm_b))
            if min_len / max_len >= 0.50:
                return 0.90

        # Similitud de secuencia (Levenshtein-like con difflib)
        seq_ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

        # Similitud de tokens (Jaccard)
        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())
        union = tokens_a | tokens_b
        jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0

        # Ponderación combinada
        combined = 0.65 * seq_ratio + 0.35 * jaccard
        return max(0.0, min(1.0, combined))

    def are_near_duplicates(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
    ) -> Tuple[bool, float]:
        """
        Determina si dos resultados de transcripción representan la misma locución.
        """
        text_a = item_a.get("text", "")
        text_b = item_b.get("text", "")

        # Comprobar proximidad temporal
        start_a = item_a.get("start_time", item_a.get("start", 0.0))
        start_b = item_b.get("start_time", item_b.get("start", 0.0))
        end_a = item_a.get("end_time", item_a.get("end", 0.0))
        end_b = item_b.get("end_time", item_b.get("end", 0.0))

        time_delta = abs(start_a - start_b)
        # O bien empiezan casi al mismo tiempo, o sus intervalos se solapan
        has_time_overlap = (time_delta <= self.max_start_time_delta) or (start_a < end_b and start_b < end_a)

        if not has_time_overlap:
            return False, 0.0

        # Contención de tokens y coincidencia de prefijo (fuga acústica con truncamiento o garbles)
        norm_a = self.normalize_text(text_a)
        norm_b = self.normalize_text(text_b)
        words_norm_a = norm_a.split()
        words_norm_b = norm_b.split()
        
        # 1. Coincidencia de prefijo común (ambos canales arrancan transcribiendo lo mismo)
        if words_norm_a and words_norm_b:
            common_prefix_len = 0
            for w1, w2 in zip(words_norm_a, words_norm_b):
                if w1 == w2:
                    common_prefix_len += 1
                else:
                    break
            min_len_words = min(len(words_norm_a), len(words_norm_b))
            if common_prefix_len >= 3 and (common_prefix_len / min_len_words) >= 0.35:
                return True, round(common_prefix_len / min_len_words, 3)

        # 2. Contención de conjunto de tokens
        tokens_a = set(words_norm_a)
        tokens_b = set(words_norm_b)
        min_tokens = min(len(tokens_a), len(tokens_b))
        if min_tokens >= 3:
            containment = len(tokens_a & tokens_b) / min_tokens
            if containment >= 0.65:
                return True, round(containment, 3)

        sim = self.calculate_similarity(text_a, text_b)
        is_dup = sim >= self.text_similarity_threshold
        return is_dup, sim

    def pick_best_transcript(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Selecciona la mejor transcripción entre dos candidatas duplicadas.
        Prioriza:
        1. Menos repeticiones/bucles de alucinación.
        2. Completitud de la frase (más palabras informativas sin bucles ni cortes finales).
        3. Confianza ASR más alta.
        """
        text_a = item_a.get("text", "").strip()
        text_b = item_b.get("text", "").strip()

        words_a = text_a.split()
        words_b = text_b.split()

        # Penalizar repeticiones consecutivas (alucinaciones)
        rep_a = sum(1 for i in range(1, len(words_a)) if words_a[i].lower() == words_a[i - 1].lower())
        rep_b = sum(1 for i in range(1, len(words_b)) if words_b[i].lower() == words_b[i - 1].lower())

        score_a = len(words_a) - (rep_a * 3) + item_a.get("confidence", 0.8)
        score_b = len(words_b) - (rep_b * 3) + item_b.get("confidence", 0.8)

        # Penalizar palabras truncadas o letras sueltas al final (ej: "physicalism m")
        if words_a and len(words_a[-1]) == 1 and words_a[-1].lower() not in ('a', 'i', 'y', 'o', 'e', 'u'):
            score_a -= 2.5
        if words_b and len(words_b[-1]) == 1 and words_b[-1].lower() not in ('a', 'i', 'y', 'o', 'e', 'u'):
            score_b -= 2.5

        if score_b > score_a:
            winner, loser = item_b, item_a
        else:
            winner, loser = item_a, item_b

        logger.info(
            "Duplicado resuelto: conservado '%s' (SPK=%s), suprimido '%s' (SPK=%s)",
            winner.get("text", "")[:40],
            winner.get("speaker_id", ""),
            loser.get("text", "")[:40],
            loser.get("speaker_id", ""),
        )
        return winner

    def resolve(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Procesa una lista de transcripciones contemporáneas y elimina duplicados.
        Si encuentra dos transcripciones casi idénticas, suprime la redundante.
        """
        if not transcripts or len(transcripts) <= 1:
            return transcripts

        # Para el caso habitual de 2 fuentes simultáneas:
        if len(transcripts) == 2:
            is_dup, sim = self.are_near_duplicates(transcripts[0], transcripts[1])
            if is_dup:
                self._suppressed_count += 1
                winner = self.pick_best_transcript(transcripts[0], transcripts[1])
                return [winner]
            return transcripts

        # Para N transcripciones:
        keep = []
        suppressed_indices = set()

        for i in range(len(transcripts)):
            if i in suppressed_indices:
                continue
            current = transcripts[i]
            for j in range(i + 1, len(transcripts)):
                if j in suppressed_indices:
                    continue
                other = transcripts[j]
                is_dup, _ = self.are_near_duplicates(current, other)
                if is_dup:
                    self._suppressed_count += 1
                    current = self.pick_best_transcript(current, other)
                    suppressed_indices.add(j)
            keep.append(current)

        return keep

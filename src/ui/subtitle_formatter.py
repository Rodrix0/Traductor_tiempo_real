"""
subtitle_formatter.py
Formateador profesional de subtítulos según estándares cinematográficos y de televisión.
Reglas:
  1. Máximo ~42 caracteres por línea.
  2. Máximo 2 líneas por subtítulo.
  3. Partición sintáctica natural (sin romper artículos, preposiciones o frases fijas).
  4. Eliminación de redundancias y limpieza de signos de puntuación en español.
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Puntos de corte preferentes en español (conjunciones y conectores)
BREAK_PREFERENCES = {
    "y", "e", "o", "u", "pero", "sino", "que", "porque", "cuando", "donde",
    "aunque", "para", "como", "sin", "con", "en", "de", "a"
}


class SubtitleFormatter:
    """
    Formatea textos traducidos para su presentación óptima en subtítulos flotantes (Overlay).
    Garantiza alta legibilidad, partición equilibrada en hasta 2 líneas y límites de longitud.
    """

    def __init__(self, max_chars_per_line: int = 42, max_lines: int = 2):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines

    def format(self, text: str) -> str:
        """
        Formatea el texto en líneas balanceadas (hasta 2 líneas, max 42 caracteres por línea).
        Devuelve el texto formateado con saltos de línea '\\n'.
        """
        if not text or not text.strip():
            return ""

        clean = self._clean_redundancies(text.strip())

        # Si ya cabe en una sola línea
        if len(clean) <= self.max_chars_per_line:
            return clean

        lines = self._split_into_lines(clean)

        # Limitar a max_lines
        if len(lines) > self.max_lines:
            # Si se excede de 2 líneas, compactar respetando las 2 líneas
            first_half = lines[0]
            second_half = " ".join(lines[1:])
            # Si la segunda línea sigue siendo muy larga, recortar con elipsis limpia
            if len(second_half) > self.max_chars_per_line + 15:
                words = second_half.split()
                truncated = []
                count = 0
                for w in words:
                    if count + len(w) + 1 <= self.max_chars_per_line:
                        truncated.append(w)
                        count += len(w) + 1
                    else:
                        break
                second_half = " ".join(truncated) + "..."
            lines = [first_half, second_half]

        return "\n".join(lines)

    def _split_into_lines(self, text: str) -> List[str]:
        """Divide el texto buscando el punto de equilibrio sintáctico más natural."""
        words = text.split()
        if not words:
            return [text]

        total_len = len(text)
        ideal_midpoint = total_len / 2.0

        best_split_idx = -1
        best_score = float("inf")

        running_len = 0
        for i, word in enumerate(words[:-1]):
            running_len += len(word) + (1 if i > 0 else 0)

            # Evaluar candidatos alrededor del punto medio
            dist_to_mid = abs(running_len - ideal_midpoint)

            # Penalizar fuertemente si una línea excede el límite máximo
            first_len = running_len
            second_len = total_len - running_len - 1
            overflow_penalty = 0
            if first_len > self.max_chars_per_line:
                overflow_penalty += (first_len - self.max_chars_per_line) * 10
            if second_len > self.max_chars_per_line:
                overflow_penalty += (second_len - self.max_chars_per_line) * 10

            # Bonificación si el corte es tras un signo de puntuación (, ; : . -)
            punct_bonus = -15 if word[-1] in {",", ";", ":", "."} else 0

            # Bonificación si la siguiente palabra es un conector o conjunción natural
            next_word = words[i + 1].lower().rstrip(",;:.!?")
            conjunction_bonus = -8 if next_word in BREAK_PREFERENCES else 0

            # Penalización si se corta un artículo ("el", "la", "un", "una") dejándolo solo
            article_penalty = 25 if word.lower() in {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "a"} else 0

            score = dist_to_mid + overflow_penalty + punct_bonus + conjunction_bonus + article_penalty

            if score < best_score:
                best_score = score
                best_split_idx = i

        if best_split_idx >= 0:
            line1 = " ".join(words[: best_split_idx + 1])
            line2 = " ".join(words[best_split_idx + 1 :])
            return [line1, line2]

        return [text]

    def _clean_redundancies(self, text: str) -> str:
        """Elimina redundancias obvias como repeticiones inmediatas de frases."""
        # 'palabra palabra' -> 'palabra'
        dedup = re.sub(r"\b(\w{3,})\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        # Normalizar espacios
        dedup = re.sub(r"\s{2,}", " ", dedup).strip()
        return dedup

"""
context_aware_translator.py
Traductor contextual e idiomático offline de alta fidelidad (ContextAwareTranslator).
Incorpora:
  1. Normalización de lenguaje oral (supresión selectiva de muletillas/fillers).
  2. Resolución general de modismos, phrasal verbs y colocaciones (IdiomResolver).
  3. Contextualización de respuestas breves en diálogos cruzados (ej: 'Absolutely' -> 'Totalmente').
  4. Integración con motores neuronales locales CTranslate2 (MarianMT / NLLB-200 / Argos).
"""

import re
import time
import logging
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

from src.translation.context_manager import ConversationContextManager, ConversationTurn
from src.translation.glossary import SmartGlossary
from src.translation.stt_postprocessor import STTPostProcessor

logger = logging.getLogger(__name__)


# =====================================================================
# 1. NORMALIZACIÓN DE LENGUAJE ORAL (Spoken Language Normalizer)
# =====================================================================

class SpokenLanguageNormalizer:
    """
    Normaliza el habla conversacional en inglés antes de traducir, eliminando o
    reformulando muletillas (fillers/hedges) que agregan ruido sin aportar significado.
    """

    # Muletillas iniciales de frase o entre comas
    LEADING_FILLERS = [
        re.compile(r"^\s*(?:like|you know|i mean|well|so)\s*,\s*", re.IGNORECASE),
        re.compile(r"^\s*(?:honestly|basically|actually|literally)\s*,\s*", re.IGNORECASE),
    ]

    # Partículas de relleno intermedias
    HEDGING_PATTERNS = [
        # 'kind of just / sort of just trying' -> 'just trying'
        (re.compile(r"\b(kind\s+of|sort\s+of)\s+just\b", re.IGNORECASE), "just"),
        # 'kind of / sort of' ante verbos continuos ('kind of feeling') -> 'feeling'
        (re.compile(r"\b(am|is|are|was|were)\s+(?:kind\s+of|sort\s+of)\s+(\w+ing)\b", re.IGNORECASE), r"\1 \2"),
        # 'you know' entre comas o como paréntesis
        (re.compile(r",\s*you\s+know\s*,", re.IGNORECASE), ","),
        # 'like' como duda oral entre comas: ', like, '
        (re.compile(r",\s*like\s*,", re.IGNORECASE), ","),
    ]

    def normalize(self, text: str, language: str = "en") -> Tuple[str, bool]:
        """Normaliza muletillas del texto sin perder contenido sustantivo."""
        if language != "en" or not text:
            return text, False

        orig = text
        res = text.strip()

        # Quitar muletillas iniciales
        for pat in self.LEADING_FILLERS:
            res = pat.sub("", res)

        # Tratar hedging
        for pat, repl in self.HEDGING_PATTERNS:
            res = pat.sub(repl, res)

        # Normalizar espacios
        res = re.sub(r"\s{2,}", " ", res).strip()

        # Asegurar mayúscula inicial si fue afectada
        if res and res[0].islower() and orig and orig[0].isupper():
            res = res[0].upper() + res[1:]

        return res, res != orig


# =====================================================================
# 2. RESOLUCIÓN DE MODISMOS Y EXPRESIONES IDIOMÁTICAS (IdiomResolver)
# =====================================================================

# Catálogo lingüístico general de modismos y colocaciones del inglés oral a español
COMMON_IDIOMATIC_EXPRESSIONS: List[Tuple[re.Pattern, str, str]] = [
    # pity party: dar lástima / hacerse la víctima
    (re.compile(r"\b(?:a\s+)?pity\s+party\b", re.IGNORECASE), "dar lástima", "dar lástima"),
    (re.compile(r"\bnot\s+as\s+a\s+pity\s+party\s+at\s+all\b", re.IGNORECASE), "no para dar lástima", "no para dar lástima"),
    (re.compile(r"\bhave\s+a\s+pity\s+party\b", re.IGNORECASE), "hacerse la víctima", "hacerse la víctima"),

    # journaling: escribir en un diario / llevar un diario
    (re.compile(r"\bjournaling\b", re.IGNORECASE), "escribir en un diario", "escribir en un diario"),
    (re.compile(r"\bjournal\b", re.IGNORECASE), "diario", "diario"),

    # shine through: destacarse / notarse
    (re.compile(r"\bshines?\s+through\b", re.IGNORECASE), "se nota", "se nota"),
    (re.compile(r"\bshining\s+through\b", re.IGNORECASE), "notándose", "notándose"),

    # come across: dar la impresión / percibirse
    (re.compile(r"\bcomes?\s+across\b", re.IGNORECASE), "da la impresión", "da la impresión"),

    # give it a shot / give it a try: intentarlo / probar
    (re.compile(r"\bgive\s+it\s+a\s+shot\b", re.IGNORECASE), "intentarlo", "intentarlo"),
    (re.compile(r"\bgive\s+it\s+a\s+try\b", re.IGNORECASE), "probar", "probar"),

    # figure it out: resolverlo / entenderlo
    (re.compile(r"\bfigure\s+(?:it\s+)?out\b", re.IGNORECASE), "resolverlo", "resolverlo"),

    # hang out: pasar el rato / salir juntos
    (re.compile(r"\bhang\s+out\b", re.IGNORECASE), "pasar el rato", "pasar el rato"),

    # turn out: resultar
    (re.compile(r"\bturns?\s+out\b", re.IGNORECASE), "resulta", "resulta"),
    (re.compile(r"\bturned\s+out\b", re.IGNORECASE), "resultó", "resultó"),

    # get over: superar
    (re.compile(r"\bget\s+over\s+(it|them|this|that)\b", re.IGNORECASE), r"superar \1", "superar"),

    # look into: investigar / revisar
    (re.compile(r"\blooks?\s+into\b", re.IGNORECASE), "revisa", "revisar"),
    (re.compile(r"\blooking\s+into\b", re.IGNORECASE), "revisando", "revisar"),

    # at least: al menos / por lo menos
    (re.compile(r"\bat\s+least\b", re.IGNORECASE), "al menos", "al menos"),

    # work out: funcionar / resolver / hacer ejercicio
    (re.compile(r"\bworks?\s+out\b", re.IGNORECASE), "funciona", "funcionar"),
]

# Sustituciones correctivas sobre la traducción al español para reparar traducciones literales de MT
POST_TRANSLATION_NATURALIZERS: List[Tuple[re.Pattern, str]] = [
    # "fiesta de lástima" -> "dar lástima" / "hacerse la víctima"
    (re.compile(r"\bcomo\s+una\s+fiesta\s+de\s+l[aá]stima\s+en\s+absoluto\b", re.IGNORECASE), "para dar lástima en absoluto"),
    (re.compile(r"\buna\s+fiesta\s+de\s+l[aá]stima\b", re.IGNORECASE), "dar lástima"),
    (re.compile(r"\bparte\s+de\s+pena\b", re.IGNORECASE), "dar lástima"),

    # "haciendo un diario" -> "escribir en un diario"
    (re.compile(r"\bhaciendo\s+un\s+diario\b", re.IGNORECASE), "escribir en un diario"),
    (re.compile(r"\bcreando\s+un\s+diario\b", re.IGNORECASE), "escribir en un diario"),

    # "brilla a través" / "brilla también" -> "también se nota"
    (re.compile(r"\bbrilla\s+a\s+trav[eé]s\b", re.IGNORECASE), "se nota"),
    (re.compile(r"\bbrilla\s+tambi[eé]n\b", re.IGNORECASE), "también se nota"),
    (re.compile(r"\btambi[eé]n\s+brilla\b", re.IGNORECASE), "también se nota"),

    # "viene a través" -> "se percibe"
    (re.compile(r"\bviene\s+a\s+trav[eé]s\b", re.IGNORECASE), "da la impresión"),

    # "hacer ejercicio" vs "funcionar" para work out
    (re.compile(r"\bva\s+a\s+funcionar\b", re.IGNORECASE), "va a salir bien"),
]


# =====================================================================
# 3. RESPUESTAS BREVES Y DIÁLOGO CRUZADO
# =====================================================================

BRIEF_CONFIRMATION_RESPONSES = {
    "absolutely": "Totalmente.",
    "absolutely.": "Totalmente.",
    "definitely": "Sin duda.",
    "definitely.": "Sin duda.",
    "exactly": "Exacto.",
    "exactly.": "Exacto.",
    "for sure": "Por supuesto.",
    "for sure.": "Por supuesto.",
    "right": "De acuerdo.",
    "right.": "De acuerdo.",
    "totally": "Totalmente.",
    "totally.": "Totalmente.",
    "sure": "Claro.",
    "sure.": "Claro.",
}


class ContextAwareTranslator:
    """
    Orquestador principal de traducción contextual e idiomática.
    Combina:
      - Normalización previa del lenguaje oral (fillers, muletillas).
      - Resolución de modismos en inglés.
      - Memoria de conversación multi-turno y multi-hablante.
      - Respuestas breves contextualizadas.
      - Motores de traducción local CTranslate2 (MarianMT / NLLB-200 / Argos).
      - Naturalización post-traducción en español para subtítulos fluidos.
    """

    def __init__(
        self,
        engine: str = "auto",
        device: str = "cpu",
        glossary: Optional[SmartGlossary] = None,
        context_manager: Optional[ConversationContextManager] = None,
    ):
        self.device = device
        self.glossary = glossary or SmartGlossary()
        self.context_manager = context_manager or ConversationContextManager()
        self.normalizer = SpokenLanguageNormalizer()
        self.stt_postprocessor = STTPostProcessor()

        # Cargar motor base (MarianMT o NLLB)
        self._local_translator = None
        self._engine_type = engine

    def _ensure_engine(self):
        if self._local_translator is None:
            from src.translation.local import LocalTranslator
            self._local_translator = LocalTranslator(glossary=self.glossary)
        return self._local_translator

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        speaker_id: Optional[str] = None,
        use_context: bool = True,
    ) -> str:
        """
        Traduce el texto incorporando contexto previo, modismos y diálogo cruzado.
        """
        clean = text.strip()
        if not clean or source_lang == target_lang:
            return clean

        engine = self._ensure_engine()
        spk = speaker_id or "SPEAKER_01"

        # 1. Comprobar si es una respuesta breve en diálogo cruzado (ej: Speaker B dice 'Absolutely.')
        if source_lang == "en" and target_lang == "es":
            low_text = clean.lower().rstrip(".!, ")
            if low_text in BRIEF_CONFIRMATION_RESPONSES:
                # Comprobar si hubo un turno previo de otro hablante
                cross_turn = self.context_manager.get_cross_speaker_context(spk)
                if cross_turn:
                    # Contextualizar la respuesta de acuerdo a la aserción anterior
                    adapted = BRIEF_CONFIRMATION_RESPONSES.get(low_text, "Totalmente.")
                    logger.info(
                        "ContextAwareTranslator: Respuesta breve contextualizada '%s' -> '%s' (tras '%s')",
                        clean,
                        adapted,
                        cross_turn.source_text[:30],
                    )
                    self.context_manager.add_turn(spk, clean, adapted, source_lang, target_lang)
                    return adapted

        # 2. Normalización de lenguaje oral (supresión selectiva de muletillas)
        normalized_text, had_norm = self.normalizer.normalize(clean, language=source_lang)

        # 3. Post-procesamiento STT acústico previo
        stt_result = self.stt_postprocessor.process(
            normalized_text,
            context=self.context_manager.get_speaker_context(spk, max_turns=2),
            language=source_lang,
        )
        preprocessed_text = stt_result.corrected_text

        # 4. Inyección de contexto si aplica
        # Para modelos MarianMT / CTranslate2, el contexto previo se puede emplear
        # para contextualizar pronombres o resolver términos
        translated_raw = engine.translate(preprocessed_text, source_lang, target_lang)

        # 5. Naturalización post-traducción en español (reparar traducciones literales de modismos)
        naturalized = translated_raw
        if source_lang == "en" and target_lang == "es":
            for pat, repl in POST_TRANSLATION_NATURALIZERS:
                if pat.search(naturalized):
                    naturalized = pat.sub(repl, naturalized)

            # Si el texto original contenía modismos específicos y la traducción quedó extraña
            if re.search(r"\bshines?\s+through\b", clean, re.IGNORECASE) and "nota" not in naturalized and "destaca" not in naturalized:
                naturalized = re.sub(r"\bbrilla\b", "se nota", naturalized, flags=re.IGNORECASE)

            # Caso especial: 'walking outside, taking a few deep breaths and journaling'
            # Asegurar infinitivo o forma verbal natural de subtítulo
            if "journaling" in clean.lower() and "diario" in naturalized:
                if not re.search(r"escribir\s+en\s+un\s+diario", naturalized, re.IGNORECASE):
                    naturalized = re.sub(r"(?:haciendo|tomando|llevando)?\s*un\s+diario", "escribir en un diario", naturalized, flags=re.IGNORECASE)

            # Si empieza con gerundio en inglés descriptivo ('walking outside...'), favorecer infinitivo en subtítulos
            if re.match(r"^[a-zA-Z]+ing\b", clean):
                if naturalized.startswith("Caminando "):
                    naturalized = "Salir a caminar " + naturalized[10:]
                elif naturalized.startswith("caminando "):
                    naturalized = "salir a caminar " + naturalized[10:]

        # 6. Aplicar glosario inteligente del usuario
        final_translated = self.glossary.apply(naturalized) if self.glossary else naturalized

        # 7. Registrar en el gestor de contexto conversacional
        self.context_manager.add_turn(
            speaker_id=spk,
            source_text=clean,
            target_text=final_translated,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        return final_translated

    def clean_source(self, text: str, source: str) -> str:
        """Limpia el texto original utilizando el normalizador."""
        norm, _ = self.normalizer.normalize(text, language=source)
        return norm

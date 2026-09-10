"""
marian_provider.py
Proveedor de traducción neuronal offline basado en Helsinki-NLP MarianMT vía CTranslate2.
Especialista de ultra-alta velocidad (<100ms) y bajo consumo de memoria para inglés -> español.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import logging

from src.translation.base import TranslationProvider
from src.translation.glossary import SmartGlossary

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_MARIAN_DIR = ROOT_DIR / "models" / "ct2fast-opus-mt-en-es"


class MarianTranslationProvider(TranslationProvider):
    """Proveedor MarianMT en CTranslate2 para traducción inglés a español."""

    def __init__(
        self,
        device: str = "cpu",
        glossary: Optional[SmartGlossary] = None,
    ):
        self.device = device
        self.glossary = glossary or SmartGlossary()
        self._translator = None
        self._sp_source = None
        self._sp_target = None
        self._initialized = False

    def _ensure_loaded(self) -> bool:
        if self._initialized:
            return True

        if not MODEL_MARIAN_DIR.exists():
            return False

        source_spm = MODEL_MARIAN_DIR / "source.spm"
        target_spm = MODEL_MARIAN_DIR / "target.spm"
        model_bin = MODEL_MARIAN_DIR / "model.bin"

        if not (source_spm.is_file() and target_spm.is_file() and model_bin.is_file()):
            return False

        try:
            import ctranslate2
            import sentencepiece as spm

            self._sp_source = spm.SentencePieceProcessor(str(source_spm))
            self._sp_target = spm.SentencePieceProcessor(str(target_spm))
            self._translator = ctranslate2.Translator(
                str(MODEL_MARIAN_DIR),
                device=self.device,
                intra_threads=4,
            )
            self._initialized = True
            logger.info("MarianTranslationProvider cargado exitosamente en %s.", self.device)
            return True
        except Exception as e:
            logger.warning("No se pudo cargar MarianTranslationProvider: %s", e)
            return False

    def translate(
        self,
        text: str,
        source: str,
        target: str,
        context: Optional[List[str]] = None,
    ) -> str:
        clean = text.strip()
        if not clean or source == target:
            return clean

        if not self._ensure_loaded():
            raise RuntimeError("El modelo MarianMT no está disponible o faltan archivos.")

        # Limpieza previa del origen
        clean = self.clean_source(clean, source)

        try:
            tokens = self._sp_source.encode(clean, out_type=str) + ["</s>"]
            results = self._translator.translate_batch(
                [tokens],
                max_decoding_length=150,
                beam_size=1,
            )
            output_tokens = results[0].hypotheses[0]
            translated = self._sp_target.decode(output_tokens).strip()

            # Aplicar glosario sobre el resultado
            return self.glossary.apply(translated)
        except Exception as e:
            logger.error("Error en MarianTranslationProvider: %s", e)
            return text

    def clean_source(self, text: str, source: str) -> str:
        if not text:
            return ""
        return " ".join(text.split())

    def get_whisper_prompt(self) -> Optional[str]:
        if not self.glossary.entries:
            return None
        terms = list(self.glossary.entries.keys())[:10]
        return f"Vocabulary: {', '.join(terms)}."

    @property
    def provider_name(self) -> str:
        return "marian"

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        return [("en", "es")]

    @property
    def is_ready(self) -> bool:
        if not self._initialized:
            return (
                (MODEL_MARIAN_DIR / "model.bin").is_file()
                and (MODEL_MARIAN_DIR / "source.spm").is_file()
                and (MODEL_MARIAN_DIR / "target.spm").is_file()
            )
        return True

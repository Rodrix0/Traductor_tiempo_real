"""
translator.py
Módulo de traducción local y offline de alta fidelidad.
Soporta dos motores en CTranslate2:
  1. 'nllb'   : Meta NLLB-200-distilled-600M (Mayor precisión, modismos, lenguaje moderno).
  2. 'marian' : Helsinki-NLP MarianMT (Ultra-liviano y veloz, ~150ms).
Incluye sistema de glosario personalizable (glossary.json) para calibración exacta de términos.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict
import ctranslate2
import sentencepiece as spm

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_MARIAN_DIR = MODELS_DIR / "ct2fast-opus-mt-en-es"
MODEL_NLLB_DIR = MODELS_DIR / "nllb-200-ct2-int8"
GLOSSARY_FILE = BASE_DIR / "glossary.json"


class LocalTranslator:
    """Traductor local multi-motor con soporte de glosario y alta precisión."""

    def __init__(
        self,
        engine: str = "nllb",
        device: str = "cpu",
        glossary_path: Optional[Path] = None,
    ):
        """
        :param engine: 'nllb' (Meta NLLB-200, recomendado) o 'marian' (MarianMT, liviano).
        :param device: 'cpu' o 'cuda'.
        """
        self.engine = engine.lower()
        self.device = device
        self.glossary_path = Path(glossary_path or GLOSSARY_FILE)
        self.glossary: Dict[str, str] = {}

        self.translator: Optional[ctranslate2.Translator] = None
        self.sp_source: Optional[spm.SentencePieceProcessor] = None
        self.sp_target: Optional[spm.SentencePieceProcessor] = None
        self.sp_nllb: Optional[spm.SentencePieceProcessor] = None

        self._load_glossary()
        self._load_engine()

    def _load_glossary(self) -> None:
        """Carga el archivo de glosario personalizado si existe."""
        if self.glossary_path.exists():
            try:
                with open(self.glossary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.glossary = {
                        k.strip(): v.strip()
                        for k, v in data.items()
                        if not k.startswith("__") and k.strip()
                    }
                logger.info("Glosario cargado: %d términos registrados.", len(self.glossary))
            except Exception as e:
                logger.warning("No se pudo leer glossary.json: %s", e)

    def _load_engine(self) -> None:
        """Carga el motor seleccionado (NLLB o MarianMT)."""
        if self.engine == "nllb":
            self._load_nllb()
        else:
            self._load_marian()

    def _load_nllb(self) -> None:
        """Inicializa Meta NLLB-200 en CTranslate2 (int8)."""
        if not MODEL_NLLB_DIR.exists():
            logger.info("Descargando modelo Meta NLLB-200 INT8 (~600MB)...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                "JustFrederik/nllb-200-distilled-600M-ct2-int8",
                local_dir=str(MODEL_NLLB_DIR),
            )

        spm_model = MODEL_NLLB_DIR / "sentencepiece.bpe.model"
        if not spm_model.exists():
            raise FileNotFoundError(f"Falta sentencepiece.bpe.model en {MODEL_NLLB_DIR}")

        self.sp_nllb = spm.SentencePieceProcessor(str(spm_model))
        self.translator = ctranslate2.Translator(
            str(MODEL_NLLB_DIR),
            device=self.device,
            intra_threads=6,
        )
        logger.info("Motor de traducción NLLB-200 cargado exitosamente en %s.", self.device)

    def _load_marian(self) -> None:
        """Inicializa MarianMT en CTranslate2."""
        if not MODEL_MARIAN_DIR.exists():
            logger.info("Descargando modelo MarianMT (~150MB)...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                "michaelfeil/ct2fast-opus-mt-en-es",
                local_dir=str(MODEL_MARIAN_DIR),
            )

        source_spm = MODEL_MARIAN_DIR / "source.spm"
        target_spm = MODEL_MARIAN_DIR / "target.spm"

        self.sp_source = spm.SentencePieceProcessor(str(source_spm))
        self.sp_target = spm.SentencePieceProcessor(str(target_spm))
        self.translator = ctranslate2.Translator(
            str(MODEL_MARIAN_DIR),
            device=self.device,
            intra_threads=4,
        )
        logger.info("Motor de traducción MarianMT cargado exitosamente en %s.", self.device)

    def clean_english_source(self, text: str) -> str:
        """Normaliza el texto en inglés de manera 100% neutral y universal."""
        if not text:
            return ""

        res = re.sub(r"\s+", " ", text).strip()

        # Solo si el usuario configuró términos personalizados en glossary.json
        for term, replacement in self.glossary.items():
            if not any(ord(c) > 127 for c in replacement):
                pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
                res = pattern.sub(replacement, res)
        return res

    def get_whisper_prompt(self) -> Optional[str]:
        """Prompt 100% neutral para Whisper. No fuerza ningún tema predefinido."""
        if not self.glossary:
            return None
        extra = [k for k in self.glossary.keys() if not k.startswith("__")]
        if not extra:
            return None
        return f"Vocabulary: {', '.join(extra[:15])}."

    def apply_glossary(self, text: str) -> str:
        """Aplica sustituciones del glosario respetando límites de palabras completas."""
        if not self.glossary:
            return text

        result = text
        for term, replacement in self.glossary.items():
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            result = pattern.sub(replacement, result)
        return result

    def translate_en_to_es(self, text: str) -> str:
        """Traduce inglés a español con el motor seleccionado y aplica el glosario."""
        text = self.clean_english_source(text.strip())
        if not text:
            return ""

        if self.translator is None:
            raise RuntimeError("El traductor no está inicializado.")

        try:
            if self.engine == "nllb" and self.sp_nllb is not None:
                # 1. Tokenizar para NLLB con prefijo de idioma inglés
                tokens = self.sp_nllb.encode(text, out_type=str)
                source_tokens = ["eng_Latn"] + tokens + ["</s>"]
                target_prefix = [["spa_Latn"]]

                results = self.translator.translate_batch(
                    [source_tokens],
                    target_prefix=target_prefix,
                    max_decoding_length=150,
                    beam_size=1,
                )

                output_tokens = results[0].hypotheses[0]
                clean_tokens = [t for t in output_tokens if t not in ["spa_Latn", "</s>"]]
                translated = self.sp_nllb.decode(clean_tokens).strip()

            else:
                # MarianMT
                tokens = self.sp_source.encode(text, out_type=str) + ["</s>"]
                results = self.translator.translate_batch(
                    [tokens],
                    max_decoding_length=150,
                    beam_size=1,
                )
                output_tokens = results[0].hypotheses[0]
                translated = self.sp_target.decode(output_tokens).strip()

            # Aplicar glosario de términos personalizados sobre el texto final
            translated_calibrated = self.apply_glossary(translated)
            return translated_calibrated

        except Exception as e:
            logger.error("Error en traducción: %s", e)
            return text

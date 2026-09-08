"""
translator.py
Módulo de traducción local y offline de alta velocidad usando CTranslate2 y MarianMT.
Traduce de inglés a español localmente; el tiempo depende del texto y del hardware.
"""

import logging
from pathlib import Path
from typing import Optional
import ctranslate2
import sentencepiece as spm

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_EN_ES_DIR = MODELS_DIR / "ct2fast-opus-mt-en-es"


class LocalTranslator:
    """Traductor local ultrarrápido basado en CTranslate2 (Helsinki-NLP opus-mt-en-es)."""

    def __init__(self, model_dir: Optional[Path] = None, device: str = "cpu"):
        self.model_dir = Path(model_dir or MODEL_EN_ES_DIR)
        self.device = device

        self.sp_source: Optional[spm.SentencePieceProcessor] = None
        self.sp_target: Optional[spm.SentencePieceProcessor] = None
        self.translator: Optional[ctranslate2.Translator] = None

        self._load_model()

    def _load_model(self) -> None:
        """Carga el modelo y los tokenizadores SentencePiece."""
        if not self.model_dir.exists():
            logger.info("Descargando modelo de traducción local 'michaelfeil/ct2fast-opus-mt-en-es'...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                "michaelfeil/ct2fast-opus-mt-en-es",
                local_dir=str(self.model_dir),
            )

        source_spm = self.model_dir / "source.spm"
        target_spm = self.model_dir / "target.spm"

        if not source_spm.exists() or not target_spm.exists():
            raise FileNotFoundError(f"No se encontraron los tokenizadores .spm en {self.model_dir}")

        self.sp_source = spm.SentencePieceProcessor(str(source_spm))
        self.sp_target = spm.SentencePieceProcessor(str(target_spm))
        self.translator = ctranslate2.Translator(
            str(self.model_dir),
            device=self.device,
            compute_type='int8' if self.device == 'cpu' else 'float16',
            intra_threads=4,
        )
        logger.info("Traductor local EN->ES cargado exitosamente en %s.", self.device)

    def translate_en_to_es(self, text: str) -> str:
        """
        Traduce un texto en inglés a español en milisegundos.
        """
        text = text.strip()
        if not text:
            return ""

        if self.translator is None or self.sp_source is None or self.sp_target is None:
            raise RuntimeError("El traductor local no está inicializado.")

        try:
            # Tokenizar agregando el token de fin de secuencia '</s>' para evitar repeticiones
            tokens = self.sp_source.encode(text, out_type=str) + ["</s>"]

            # Beam search preserves more candidate meanings than greedy decoding.
            results = self.translator.translate_batch(
                [tokens],
                max_decoding_length=150,
                beam_size=4,
            )

            output_tokens = results[0].hypotheses[0]
            # Decodificar tokens a texto en español
            translated = self.sp_target.decode(output_tokens)
            return translated.strip()

        except Exception as e:
            logger.error("Error en la traducción: %s", e)
            raise RuntimeError(f'No se pudo traducir al español: {e}') from e

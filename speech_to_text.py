"""
speech_to_text.py
Motor de transcripción local de voz a texto con faster-whisper.
Optimizado para tiempo real (beam_size=1, hilos de CPU acelerados y temperatura=0.0).
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Silenciar advertencias de symlinks en Windows para HuggingFace Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class SpeechToText:
    """Motor de transcripción local optimizado para mínima latencia."""

    def __init__(
        self,
        model_size: str = "small",
        default_language: Optional[str] = "en",
        beam_size: int = 1,
        download_root: Optional[str] = None,
    ):
        self.model_size = model_size
        self.default_language = default_language
        self.beam_size = beam_size
        self.download_root = download_root or str(MODELS_DIR)
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"

        self._load_model()

    def _load_model(self) -> None:
        """Carga el modelo faster-whisper una sola vez con soporte multi-hilo en CPU."""
        from faster_whisper import WhisperModel

        cuda_supported = False
        try:
            import ctranslate2
            cuda_supported = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda_supported = False

        if cuda_supported:
            try:
                logger.info("Intentando cargar Whisper '%s' en GPU (CUDA)...", self.model_size)
                model_cand = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16",
                    download_root=self.download_root,
                )
                dummy = np.zeros(1600, dtype=np.float32)
                list(model_cand.transcribe(dummy, beam_size=1)[0])

                self.model = model_cand
                self.device = "cuda"
                self.compute_type = "float16"
                logger.info("Whisper '%s' cargado exitosamente en GPU (CUDA).", self.model_size)
                return
            except Exception as e:
                logger.info(
                    "CUDA no disponible o faltan librerías nativas (%s). Recurriendo a CPU...",
                    str(e),
                )

        logger.info("Cargando Whisper '%s' en CPU (int8, multi-hilo)...", self.model_size)
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=6,  # Acelera la inferencia usando varios núcleos de CPU
                download_root=self.download_root,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            logger.info("Whisper '%s' cargado exitosamente en CPU (int8).", self.model_size)
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo faster-whisper '{self.model_size}': {e}") from e

    def transcribe(
        self,
        audio_16k: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe un segmento de audio (16kHz float32 mono).
        Retorna diccionario con:
          - text: texto transcripto
          - elapsed_time: tiempo que demoró Whisper
          - language: idioma detectado o usado
        """
        if self.model is None:
            raise RuntimeError("El modelo faster-whisper no está cargado.")

        if len(audio_16k) == 0:
            return {"text": "", "elapsed_time": 0.0, "language": ""}

        rms = float(np.sqrt(np.mean(audio_16k**2)))
        if rms < 0.003:
            return {"text": "", "elapsed_time": 0.0, "language": ""}

        lang = language if language is not None else self.default_language
        start_t = time.perf_counter()

        try:
            segments_gen, info = self.model.transcribe(
                audio_16k,
                language=lang,
                beam_size=self.beam_size,
                temperature=0.0,
                initial_prompt=initial_prompt,
                vad_filter=False,
                condition_on_previous_text=False,
            )

            parts = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
            elapsed = time.perf_counter() - start_t
            detected_lang = info.language if info else lang

            return {
                "text": " ".join(parts).strip(),
                "elapsed_time": elapsed,
                "language": detected_lang,
            }

        except Exception as e:
            logger.error("Error durante la transcripción: %s", e)
            return {"text": "", "elapsed_time": 0.0, "language": ""}

"""
speech_to_text.py
Motor de transcripción de voz a texto local con faster-whisper.
Configurado para alta precisión en español con beam_size=5 y modelo 'small'.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import numpy as np

# Silenciar advertencias de symlinks en Windows para HuggingFace Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class SpeechToText:
    """Motor de transcripción local con faster-whisper cargado una sola vez."""

    def __init__(
        self,
        model_size: str = "small",
        language: str = "es",
        beam_size: int = 5,
        download_root: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.download_root = download_root or str(MODELS_DIR)
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"

        self._load_model()

    def _load_model(self) -> None:
        """Carga el modelo faster-whisper una sola vez."""
        from faster_whisper import WhisperModel

        # Comprobar disponibilidad de CUDA
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
                # Validar con un fragmento dummy que no falten dlls (cublas/cudnn)
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

        # Carga en CPU con int8
        logger.info("Cargando Whisper '%s' en CPU (int8)...", self.model_size)
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.download_root,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            logger.info("Whisper '%s' cargado exitosamente en CPU (int8).", self.model_size)
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo faster-whisper '{self.model_size}': {e}") from e

    def transcribe(self, audio_16k: np.ndarray) -> str:
        """
        Transcribe un segmento de audio (16kHz float32 mono).
        Retorna la transcripción limpia en español.
        """
        if self.model is None:
            raise RuntimeError("El modelo faster-whisper no está cargado.")

        if len(audio_16k) == 0:
            return ""

        # Si el audio es silencio puro o energía despreciable, evitar alucinaciones
        rms = float(np.sqrt(np.mean(audio_16k**2)))
        if rms < 0.003:
            return ""

        try:
            # Desactivamos vad_filter interno de Whisper porque nuestro VAD externo ya
            # recortó la frase exacta. Esto evita que Whisper elimine palabras suaves.
            segments_gen, _ = self.model.transcribe(
                audio_16k,
                language=self.language,
                beam_size=self.beam_size,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
            )

            parts = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
            return " ".join(parts).strip()

        except Exception as e:
            logger.error("Error durante la transcripción: %s", e)
            return ""

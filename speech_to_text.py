"""
speech_to_text.py
Módulo de reconocimiento de voz usando faster-whisper.
Funciona 100% de forma local, intenta usar GPU NVIDIA automáticamente y recurre a CPU si no está disponible.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import numpy as np

# Silenciar advertencias de symlinks en Windows para HuggingFace Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

# Carpeta local para almacenar los modelos de forma permanente
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class SpeechToText:
    """Motor local de transcripción de voz a texto con faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = "es",
        download_root: Optional[str] = None,
    ):
        """
        :param model_size: Tamaño del modelo ("tiny", "base", "small", "medium").
        :param language: Código del idioma ("es", "en", etc.) o None para autodetección.
        :param download_root: Ruta donde guardar el modelo localmente.
        """
        self.model_size = model_size
        self.language = language
        self.download_root = download_root or str(MODELS_DIR)
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"

        self._load_model()

    def _load_model(self) -> None:
        """Carga el modelo faster-whisper intentando GPU (CUDA) primero y luego CPU."""
        from faster_whisper import WhisperModel

        # 1. Verificar si CTranslate2 detecta GPU CUDA
        cuda_supported = False
        try:
            import ctranslate2
            cuda_supported = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda_supported = False

        # 2. Intentar cargar en GPU si está disponible
        if cuda_supported:
            try:
                logger.info("Intentando cargar modelo '%s' en GPU (CUDA float16)...", self.model_size)
                model_candidate = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16",
                    download_root=self.download_root,
                )
                # Prueba rápida de inferencia para validar que las librerías cuBLAS / cuDNN estén presentes
                dummy = np.zeros(1600, dtype=np.float32)
                list(model_candidate.transcribe(dummy, beam_size=1)[0])

                self.model = model_candidate
                self.device = "cuda"
                self.compute_type = "float16"
                logger.info("Modelo cargado exitosamente en GPU (CUDA).")
                return
            except Exception as e:
                logger.warning(
                    "CUDA no pudo inicializarse (%s). Recurriendo automáticamente a CPU...",
                    str(e),
                )

        # 3. Carga en CPU con cuantización int8 (rápida y compatible con cualquier máquina)
        try:
            logger.info("Cargando modelo '%s' en CPU (int8)...", self.model_size)
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.download_root,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            logger.info("Modelo cargado exitosamente en CPU.")
        except Exception as e:
            raise RuntimeError(f"Error crítico al cargar el modelo Whisper: {e}") from e

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe un segmento de audio (16kHz float32 mono).
        Retorna el texto transcrito como una cadena limpia.
        """
        if self.model is None:
            raise RuntimeError("El modelo de transcripción no está inicializado.")

        if len(audio_data) == 0:
            return ""

        try:
            # Transcripción optimizada para tiempo real y baja latencia
            segments_gen, _ = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=1,  # beam_size 1 es el más rápido
                vad_filter=True,  # Filtro secundario para descartar ruidos de fondo
                condition_on_previous_text=False,
            )

            parts = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
            return " ".join(parts).strip()

        except Exception as e:
            logger.error("Error durante la transcripción: %s", e)
            return ""

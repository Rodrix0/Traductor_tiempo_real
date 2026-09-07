"""
Motor de reconocimiento de voz usando faster-whisper.
Optimizado para baja latencia con CTranslate2, selección automática de GPU (CUDA) o CPU,
y guardado persistente de modelos para funcionamiento 100% offline.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import logging
from typing import Optional, Dict, Any, List
import numpy as np

from config.settings import (
    MODELS_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    BEAM_SIZE,
)

logger = logging.getLogger(__name__)


class WhisperEngine:
    """Envoltorio optimizado sobre faster-whisper para transcripción en tiempo real."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        download_root: Optional[str] = None,
    ):
        self.model_size = model_size
        self.download_root = download_root or str(MODELS_DIR)
        self.model = None
        self.actual_device = device
        self.actual_compute_type = compute_type

        self._load_model(device, compute_type)

    def _load_model(self, requested_device: str, requested_compute_type: str) -> None:
        """Carga el modelo faster-whisper con soporte de fallback automático a CPU."""
        from faster_whisper import WhisperModel

        target_device = requested_device
        target_compute = requested_compute_type
        if requested_device != 'auto' and requested_compute_type == 'auto':
            target_compute = 'int8' if requested_device == 'cpu' else 'float16'

        # Si se especificó 'auto', intentar GPU primero
        if requested_device == "auto":
            try:
                import ctranslate2
                cuda_available = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                cuda_available = False

            if cuda_available:
                target_device = "cuda"
                target_compute = "float16" if requested_compute_type == "auto" else requested_compute_type
            else:
                target_device = "cpu"
                target_compute = "int8" if requested_compute_type == "auto" else requested_compute_type

        logger.info(
            "Cargando modelo faster-whisper '%s' en [%s - %s] (Directorio: %s)...",
            self.model_size,
            target_device,
            target_compute,
            self.download_root,
        )

        try:
            self.model = WhisperModel(
                self.model_size,
                device=target_device,
                compute_type=target_compute,
                download_root=self.download_root,
            )
            # Prueba de inferencia rápida para validar disponibilidad de librerías CUDA
            dummy_test = np.zeros(1600, dtype=np.float32)
            list(self.model.transcribe(dummy_test, beam_size=1)[0])

            self.actual_device = target_device
            self.actual_compute_type = target_compute
            logger.info("Modelo cargado exitosamente en %s.", target_device)

        except Exception as e:
            if target_device == "cuda":
                logger.warning(
                    "CUDA no disponible o faltan librerías nativas (%s). Ejecutando en CPU con int8...",
                    str(e),
                )
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self.download_root,
                )
                self.actual_device = "cpu"
                self.actual_compute_type = "int8"
                logger.info("Modelo cargado exitosamente en CPU (int8).")
            else:
                raise e

        # Calentamiento inicial rápido (warm-up) para evitar latencia en la primera frase
        self._warmup()

    def _warmup(self) -> None:
        """Realiza una transcripción en blanco de prueba para compilar caches de ejecución."""
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            list(self.model.transcribe(dummy_audio, beam_size=1)[0])
        except Exception as e:
            logger.debug("Warmup finalizado: %s", e)

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = BEAM_SIZE,
    ) -> Dict[str, Any]:
        """
        Transcribe un array de audio (16kHz float32 mono).

        Retorna un diccionario con:
          - text: texto transcripto consolidado
          - language: idioma detectado o utilizado
          - probability: probabilidad del idioma detectado
          - elapsed_time: tiempo en segundos que tomó la transcripción
          - segments: lista de segmentos detallados con marcas de tiempo
        """
        if self.model is None:
            raise RuntimeError("El modelo faster-whisper no está cargado.")

        start_time = time.perf_counter()

        # Configuración de transcripción optimizada para tiempo real
        segments_gen, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=True,  # Filtro secundario para descartar ruidos residuales
            vad_parameters=dict(min_silence_duration_ms=400),
            condition_on_previous_text=False,  # Evita repetir frases anteriores
        )

        segments_list = []
        full_text_parts: List[str] = []

        for seg in segments_gen:
            clean_text = seg.text.strip()
            if clean_text:
                full_text_parts.append(clean_text)
                segments_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": clean_text,
                })

        elapsed = time.perf_counter() - start_time
        full_text = " ".join(full_text_parts).strip()

        return {
            "text": full_text,
            "language": info.language if info else language,
            "probability": info.language_probability if info else 1.0,
            "elapsed_time": elapsed,
            "segments": segments_list,
        }

"""
manager.py
Gestor centralizado de motores ASR (Automatic Speech Recognition).
Detecta automáticamente capacidades de hardware (CUDA / CPU),
recomienda configuraciones óptimas y administra el ciclo de vida de los motores.
"""

import logging
from typing import Optional, Dict, Any
from src.asr.base import ASREngine
from src.asr.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """
    Detecta si CUDA está disponible en CTranslate2 y devuelve la configuración óptima recomendada.
    """
    cuda_available = False
    device_count = 0
    try:
        import ctranslate2
        device_count = ctranslate2.get_cuda_device_count()
        cuda_available = device_count > 0
    except Exception as e:
        logger.debug("No se pudo consultar CTranslate2 para CUDA: %s", e)

    if cuda_available:
        return {
            "cuda_available": True,
            "device_count": device_count,
            "recommended_device": "cuda",
            "recommended_compute": "float16",
        }
    else:
        return {
            "cuda_available": False,
            "device_count": 0,
            "recommended_device": "cpu",
            "recommended_compute": "int8",
        }


def sanitize_model_for_language(model_size: str, language: Optional[str]) -> str:
    """
    Evita usar modelos '.en' monolingües cuando el idioma es multilingüe o automático.
    Si el usuario eligió 'auto' o 'es' pero el modelo configurado es 'small.en',
    lo mapea a 'small' para evitar transcripciones erróneas.
    """
    if model_size.endswith(".en"):
        if language is None or language.strip().lower() in ("", "auto", "automático", "es", "de", "fr", "it", "pt", "ja", "zh"):
            base_model = model_size[:-3]
            logger.warning(
                "Modelo monolingüe '%s' incompatible con idioma '%s'. Usando modelo multilingüe '%s'.",
                model_size, language, base_model
            )
            return base_model
    return model_size


class ASRManager:
    """Administrador de motores ASR y asignación de recursos."""

    def __init__(self):
        self._current_engine: Optional[ASREngine] = None
        self._current_model_size: Optional[str] = None
        self._current_device: Optional[str] = None
        self._current_compute: Optional[str] = None

    def get_engine(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None,
    ) -> ASREngine:
        """
        Obtiene o reutiliza una instancia de ASREngine.
        Si ya existe una instancia con la misma configuración, se reutiliza.
        Si la configuración cambia, libera la anterior para conservar memoria.
        """
        if (
            self._current_engine is not None
            and self._current_model_size == model_size
            and (device == "auto" or self._current_device == device)
        ):
            return self._current_engine

        # Si había un motor previo y cambia el modelo, liberamos referencia
        if self._current_engine is not None:
            logger.info("Cambiando motor ASR de %s a %s", self._current_model_size, model_size)
            self._current_engine = None

        engine = WhisperEngine(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )

        self._current_engine = engine
        self._current_model_size = model_size
        self._current_device = engine.device
        self._current_compute = engine.compute_type

        return engine

    @property
    def current_engine(self) -> Optional[ASREngine]:
        return self._current_engine

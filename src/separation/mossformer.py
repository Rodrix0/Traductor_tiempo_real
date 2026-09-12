"""
mossformer.py
Backend para separación de voces utilizando arquitectura MossFormer2 / ClearVoice.
Diseñado para separación de 2 hablantes en habla simultánea.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.separation.base import SpeechSeparator
from src.speakers.models import SeparatedSource

logger = logging.getLogger(__name__)


class MossFormerSeparator(SpeechSeparator):
    """
    Adaptador para MossFormer2 (ClearVoice / ModelScope).
    Ofrece carga bajo demanda, detección de hardware CUDA/CPU y liberación de memoria.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model = None
        self._device = "cpu"
        self._last_latency_ms = 0.0
        self._total_separations = 0

    def load(self, device: Optional[str] = None) -> bool:
        """Carga el modelo MossFormer2 si las dependencias y modelos locales están disponibles."""
        import os
        if not (os.path.isdir("models/mossformer") or os.path.isdir("models/ClearVoice")):
            logger.info("Modelos locales MossFormer2 no encontrados en 'models/mossformer'. Modo offline en espera.")
            return False

        try:
            import torch
            target_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = target_dev

            # Intentar cargar ClearVoice si está instalado
            try:
                from clearvoice import ClearVoice
                self.model = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
                logger.info("MossFormer2 cargado exitosamente en %s.", self._device)
                return True
            except ImportError:
                logger.info("ClearVoice no está instalado. MossFormerSeparator en espera de backend.")
                return False

        except Exception as e:
            logger.warning("No se pudo inicializar MossFormer2: %s", e)
            self.model = None
            return False

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("MossFormer2 descargado de memoria.")

    def is_ready(self) -> bool:
        return self.model is not None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "backend": "mossformer2",
            "max_speakers": 2,
            "sample_rate": self.sample_rate,
            "device": self._device,
            "is_loaded": self.is_ready(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "last_latency_ms": self._last_latency_ms,
            "total_separations": self._total_separations,
        }

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
    ) -> List[SeparatedSource]:
        """Ejecuta la separación mediante el modelo MossFormer2."""
        t0 = time.perf_counter()
        if not self.is_ready():
            raise RuntimeError("MossFormerSeparator no está cargado o no se encuentra disponible.")

        # Inferencia con ClearVoice
        output = self.model(audio)
        # Adaptar fuentes devueltas
        s0 = output[0] if len(output) > 0 else audio
        s1 = output[1] if len(output) > 1 else np.zeros_like(audio)

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._total_separations += 1

        duration = len(audio) / sample_rate
        end_time = start_time + duration

        return [
            SeparatedSource(
                source_id="source_0",
                audio=np.asarray(s0, dtype=np.float32),
                sample_rate=sample_rate,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                confidence=0.92,
            ),
            SeparatedSource(
                source_id="source_1",
                audio=np.asarray(s1, dtype=np.float32),
                sample_rate=sample_rate,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                confidence=0.92,
            ),
        ]

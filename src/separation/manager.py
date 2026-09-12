"""
manager.py
Gestor central de separación de fuentes vocales (SpeechSeparatorManager).
Orquesta la selección automática de backend, detección de hardware CUDA/CPU,
gestión de ciclo de vida de memoria y fallback seguro a audio original ante fallos.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.separation.base import SpeechSeparator
from src.separation.spectral_separator import SpectralSpeechSeparator
from src.separation.mossformer import MossFormerSeparator
from src.separation.sepformer import SepFormerSeparator
from src.speakers.models import SeparatedSource

logger = logging.getLogger(__name__)


class SpeechSeparatorManager:
    """
    Administrador unificado de motores de separación de fuentes de voz.
    Garantiza que la separación sea transparente, resiliente a errores y 100% offline.
    """

    def __init__(
        self,
        mode: str = "auto",  # 'auto', 'mossformer', 'sepformer', 'spectral', 'disabled'
        preferred_device: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        self.mode = mode
        self.preferred_device = preferred_device
        self.sample_rate = sample_rate

        self._active_separator: Optional[SpeechSeparator] = None
        self._backend_name: str = "none"
        self._failures_count: int = 0
        self._total_calls: int = 0
        self._last_latency_ms: float = 0.0

    def load(self) -> bool:
        """Inicializa el backend de separación según el modo seleccionado."""
        if self.mode == "disabled":
            self.unload()
            self._backend_name = "disabled"
            return True

        # Modo SPECTRAL: carga directa instantánea
        if self.mode == "spectral":
            sep = SpectralSpeechSeparator(sample_rate=self.sample_rate)
            sep.load(device="cpu")
            self._active_separator = sep
            self._backend_name = "spectral_wiener"
            return True

        import os
        has_moss = os.path.isdir("models/mossformer") or os.path.isdir("models/ClearVoice")
        has_sep = os.path.isdir("models/sepformer")

        # Si estamos en auto y no hay modelos neuronales descargados, usar Spectral inmediatamente (0 ms)
        if self.mode == "auto" and not (has_moss or has_sep):
            spectral = SpectralSpeechSeparator(sample_rate=self.sample_rate)
            spectral.load(device="cpu")
            self._active_separator = spectral
            self._backend_name = "spectral_wiener"
            logger.info("SpeechSeparatorManager inicializado con backend de alta velocidad '%s'.", self._backend_name)
            return True

        # Determinar hardware disponible solo si es necesario para modelos neuronales
        device = self.preferred_device
        if not device:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        if self.mode == "mossformer" or (self.mode == "auto" and has_moss):
            sep = MossFormerSeparator(sample_rate=self.sample_rate)
            if sep.load(device=device):
                self._active_separator = sep
                self._backend_name = "mossformer2"
                return True
            logger.warning("Fallo al cargar MossFormer. Intentando fallback...")

        if self.mode == "sepformer" or (self.mode == "auto" and has_sep):
            sep = SepFormerSeparator(sample_rate=self.sample_rate)
            if sep.load(device=device):
                self._active_separator = sep
                self._backend_name = "sepformer"
                return True
            logger.warning("Fallo al cargar SepFormer. Intentando fallback...")

        # Fallback siempre disponible, instantáneo y 100% offline
        spectral = SpectralSpeechSeparator(sample_rate=self.sample_rate)
        spectral.load(device="cpu")
        self._active_separator = spectral
        self._backend_name = "spectral_wiener"
        logger.info("SpeechSeparatorManager inicializado con backend de alta velocidad '%s'.", self._backend_name)
        return True

    def unload(self) -> None:
        """Libera los recursos del separador activo."""
        if self._active_separator:
            try:
                self._active_separator.unload()
            except Exception as e:
                logger.warning("Error descargando separador: %s", e)
            self._active_separator = None
        self._backend_name = "none"

    def is_ready(self) -> bool:
        return self._active_separator is not None and self._active_separator.is_ready()

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
    ) -> Tuple[List[SeparatedSource], bool]:
        """
        Separa el audio en dos fuentes independientes.
        Retorna (lista_fuentes, separation_failed).
        En caso de error o modo desactivado, retorna el audio original intacto sin perder nada.
        """
        self._total_calls += 1
        t0 = time.perf_counter()

        if self.mode == "disabled":
            end_t = start_time + (len(audio) / sample_rate if audio is not None else 0.0)
            orig = SeparatedSource(source_id="original", audio=audio, sample_rate=sample_rate, start_time=start_time, end_time=end_t)
            return [orig], False

        if not self.is_ready():
            self.load()

        if not self.is_ready():
            # Si no hay separador disponible, fallback inmediato al original
            end_t = start_time + (len(audio) / sample_rate if audio is not None else 0.0)
            orig = SeparatedSource(source_id="original", audio=audio, sample_rate=sample_rate, start_time=start_time, end_time=end_t)
            return [orig], False

        try:
            sources = self._active_separator.separate(audio, sample_rate=sample_rate, start_time=start_time)
            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return sources, False

        except Exception as exc:
            self._failures_count += 1
            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("Error durante separación con '%s': %s. Aplicando fallback a audio original.", self._backend_name, exc, exc_info=True)
            end_t = start_time + (len(audio) / sample_rate if audio is not None else 0.0)
            fallback_source = SeparatedSource(
                source_id="original",
                audio=audio,
                sample_rate=sample_rate,
                start_time=round(start_time, 3),
                end_time=round(end_t, 3),
                confidence=0.0,
            )
            return [fallback_source], True

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "backend": self._backend_name,
            "mode": self.mode,
            "is_ready": self.is_ready(),
            "total_calls": self._total_calls,
            "failures": self._failures_count,
            "last_latency_ms": round(self._last_latency_ms, 2),
        }

"""
sepformer.py
Backend para separación de voces utilizando SpeechBrain SepFormer.
Optimizado para separación de 2 hablantes (WSJ0-2mix / Libri2Mix).
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.separation.base import SpeechSeparator
from src.speakers.models import SeparatedSource

logger = logging.getLogger(__name__)


class SepFormerSeparator(SpeechSeparator):
    """
    Adaptador para SepFormer de SpeechBrain.
    Permite cargar el modelo preentrenado localmente y realizar inferencia por bloques.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model = None
        self._device = "cpu"
        self._last_latency_ms = 0.0
        self._total_separations = 0

    def load(self, device: Optional[str] = None) -> bool:
        """Carga el modelo SepFormer si speechbrain y los pesos locales están disponibles."""
        import os
        if not os.path.isdir("models/sepformer"):
            logger.info("Directorio local 'models/sepformer' no encontrado. SepFormer offline en espera.")
            return False

        try:
            import torch
            target_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = target_dev

            try:
                from speechbrain.inference.separation import SepformerSeparation
                self.model = SepformerSeparation.from_hparams(
                    source="models/sepformer",
                    savedir="models/sepformer",
                    run_opts={"device": self._device},
                )
                logger.info("SepFormer cargado exitosamente en %s.", self._device)
                return True
            except (ImportError, Exception) as e:
                logger.info("SpeechBrain SepFormer no disponible en entorno actual: %s", e)
                return False

        except Exception as e:
            logger.warning("Error intentando cargar SepFormer: %s", e)
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
            logger.info("SepFormer descargado de memoria.")

    def is_ready(self) -> bool:
        return self.model is not None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "backend": "speechbrain_sepformer",
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
        """Separa la señal de audio en 2 fuentes mediante SepFormer."""
        t0 = time.perf_counter()
        if not self.is_ready():
            raise RuntimeError("SepFormerSeparator no está cargado o no se encuentra disponible.")

        import torch
        tensor = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        with torch.no_grad():
            est_sources = self.model.separate_batch(tensor)
            # est_sources tiene dimensiones [batch, time, num_spks]

        s0 = est_sources[0, :, 0].detach().cpu().numpy()
        s1 = est_sources[0, :, 1].detach().cpu().numpy()

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
                confidence=0.94,
            ),
            SeparatedSource(
                source_id="source_1",
                audio=np.asarray(s1, dtype=np.float32),
                sample_rate=sample_rate,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                confidence=0.94,
            ),
        ]

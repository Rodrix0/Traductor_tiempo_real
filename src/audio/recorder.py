"""
Módulo de captura de audio desde micrófono en tiempo real usando sounddevice.
Captura buffers pequeños continuos de audio mono en formato float32 sin bloquear el hilo principal.
"""

import queue
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import sounddevice as sd

from config.settings import SAMPLE_RATE, CHANNELS, DTYPE, FRAME_SIZE

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Capturador continuo de audio en streaming con cola thread-safe."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
        blocksize: int = FRAME_SIZE,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """Devuelve la lista de dispositivos de entrada de audio disponibles."""
        devices = sd.query_devices()
        input_devices = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devices.append({
                    "id": idx,
                    "name": dev.get("name"),
                    "hostapi": dev.get("hostapi"),
                    "channels": dev.get("max_input_channels"),
                    "default_samplerate": dev.get("default_samplerate"),
                })
        return input_devices

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback invocado por sounddevice en cada bloque de audio."""
        if status:
            logger.warning("Estado de audio callback: %s", status)

        # Clonar datos y aplanar a 1D (mono)
        data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.audio_queue.put(data)

    def start(self) -> None:
        """Inicia el stream de captura del micrófono."""
        if self._is_recording:
            return

        logger.info(
            "Iniciando captura de audio (Samplerate: %d Hz, Frame size: %d samples)...",
            self.sample_rate,
            self.blocksize,
        )
        self._is_recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Detiene y cierra el stream de captura."""
        if not self._is_recording:
            return

        logger.info("Deteniendo captura de audio...")
        self._is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Obtiene el siguiente frame de audio de la cola. Retorna None si expira el timeout."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Vacía cualquier audio residual acumulado en la cola."""
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

    @property
    def is_recording(self) -> bool:
        return self._is_recording

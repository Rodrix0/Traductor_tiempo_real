"""
audio_capture.py
Módulo encargado de capturar audio continuo del micrófono y detectar
fragmentos de voz (VAD) descartando silencios largos.
"""

import queue
import logging
from collections import deque
from typing import Optional, List, Dict, Any
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captura audio del micrófono y entrega fragmentos donde se detecta voz."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        energy_threshold: float = 0.015,
        silence_duration_ms: int = 700,
        min_speech_duration_ms: int = 400,
        max_speech_duration_ms: int = 10000,
    ):
        """
        :param device_index: ID del micrófono (None para usar el predeterminado del sistema).
        :param sample_rate: Frecuencia de muestreo (16000 Hz estándar de Whisper).
        :param energy_threshold: Umbral de energía RMS para distinguir voz de silencio.
        :param silence_duration_ms: Milisegundos de silencio para dar por terminada la frase.
        :param min_speech_duration_ms: Duración mínima en ms para descartar ruidos breves (chasquidos, teclas).
        :param max_speech_duration_ms: Duración máxima en ms antes de forzar el corte de un fragmento.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms

        # Cada bloque de audio dura 30 ms (480 muestras a 16kHz)
        self.frame_size = int(self.sample_rate * 0.030)

        # Cola de bloques de voz listos para transcribir
        self.speech_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Búfer circular de pre-voz (240 ms) para no recortar la primera sílaba
        pre_frames_count = max(1, int(0.240 / 0.030))
        self.pre_buffer: deque = deque(maxlen=pre_frames_count)

        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False

        # Estado del detector de voz
        self._is_speaking = False
        self._current_frames: List[np.ndarray] = []
        self._silence_frames = 0
        self._total_speech_samples = 0

    @staticmethod
    def list_microphones() -> List[Dict[str, Any]]:
        """Devuelve la lista de dispositivos de entrada de audio disponibles en el sistema."""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            mic_list = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    mic_list.append({
                        "id": idx,
                        "name": dev.get("name"),
                        "channels": dev.get("max_input_channels"),
                        "is_default": (idx == default_input),
                    })
            return mic_list
        except Exception as e:
            logger.error("Error al consultar dispositivos de audio: %s", e)
            return []

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback llamado automáticamente por sounddevice para cada frame de 30ms."""
        if status:
            logger.warning("Aviso de audio: %s", status)

        # Convertir a mono 1D
        data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        # Calcular energía RMS del frame
        rms = float(np.sqrt(np.mean(data**2)))
        is_speech = rms >= self.energy_threshold

        if not self._is_speaking:
            if is_speech:
                # Comienza la frase: incorporamos el búfer previo para no cortar palabras
                self._is_speaking = True
                self._current_frames = list(self.pre_buffer)
                self._current_frames.append(data)
                self._total_speech_samples = sum(len(f) for f in self._current_frames)
                self._silence_frames = 0
                self.pre_buffer.clear()
            else:
                self.pre_buffer.append(data)
        else:
            # Seguimos acumulando voz
            self._current_frames.append(data)
            self._total_speech_samples += len(data)

            if is_speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            silence_ms = (self._silence_frames * self.frame_size / self.sample_rate) * 1000.0
            speech_ms = (self._total_speech_samples / self.sample_rate) * 1000.0

            # Caso 1: Silencio prolongado tras hablar -> Frase terminada
            # Caso 2: El usuario habló continuamente más del límite máximo -> Cortar para no retrasar
            if silence_ms >= self.silence_duration_ms or speech_ms >= self.max_speech_duration_ms:
                if speech_ms >= self.min_speech_duration_ms and self._current_frames:
                    segment = np.concatenate(self._current_frames)
                    self.speech_queue.put(segment)

                # Reiniciar estado
                self._is_speaking = False
                self._current_frames = []
                self._silence_frames = 0
                self._total_speech_samples = 0
                self.pre_buffer.clear()

    def start(self) -> None:
        """Inicia la captura del micrófono."""
        if self._is_recording:
            return

        try:
            self._is_recording = True
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_size,
                device=self.device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            self._is_recording = False
            raise RuntimeError(
                f"No se pudo acceder al micrófono (ID {self.device_index}). "
                f"Verifica que el dispositivo esté conectado y habilitado en Windows. Error: {e}"
            ) from e

    def stop(self) -> None:
        """Detiene y cierra el stream de captura."""
        self._is_recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def get_speech_segment(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Retorna el siguiente fragmento de voz detectado.
        Retorna None si no hay fragmentos disponibles dentro del tiempo límite.
        """
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

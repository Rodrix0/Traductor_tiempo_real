"""
ring_buffer.py
Búfer circular de audio en memoria RAM (AudioRingBuffer).
Conserva continuamente los últimos N segundos de audio PCM (16 kHz mono float32)
para permitir:
  1. Pre-roll buffer (250-350 ms) para no recortar fonemas o sílabas iniciales ("I've done", "Well").
  2. Post-roll buffer y hangover para no cortar finales de frase prematuramente.
  3. Extracción de ventanas ampliadas (+/- 500-1000 ms) para reintentos locales (STT retry).
"""

import threading
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioRingBuffer:
    """
    Búfer circular continuo de audio seguro para subprocesos (thread-safe).
    Mantiene un historial rodante de muestras a 16 kHz.
    """

    def __init__(self, capacity_seconds: float = 5.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.capacity_samples = int(capacity_seconds * sample_rate)
        self.buffer = np.zeros(self.capacity_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_samples_written = 0
        self.lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        """Escribe nuevas muestras de audio en el búfer circular."""
        if samples is None or len(samples) == 0:
            return

        data = samples.astype(np.float32)
        n = len(data)

        with self.lock:
            if n >= self.capacity_samples:
                # Si las muestras entrantes superan toda la capacidad, conservar solo las últimas
                self.buffer[:] = data[-self.capacity_samples:]
                self.write_pos = 0
                self.total_samples_written += n
                return

            end_pos = self.write_pos + n
            if end_pos <= self.capacity_samples:
                self.buffer[self.write_pos:end_pos] = data
            else:
                first_part = self.capacity_samples - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                second_part = n - first_part
                self.buffer[:second_part] = data[first_part:]

            self.write_pos = (self.write_pos + n) % self.capacity_samples
            self.total_samples_written += n

    def get_recent(self, seconds: float) -> np.ndarray:
        """Obtiene los últimos N segundos de audio continuo escritos en el búfer."""
        samples_needed = int(seconds * self.sample_rate)
        with self.lock:
            available = min(self.total_samples_written, self.capacity_samples)
            count = min(samples_needed, available)
            if count <= 0:
                return np.zeros(0, dtype=np.float32)

            start_idx = (self.write_pos - count) % self.capacity_samples
            if start_idx + count <= self.capacity_samples:
                return self.buffer[start_idx:start_idx + count].copy()
            else:
                first_part = self.capacity_samples - start_idx
                second_part = count - first_part
                return np.concatenate([
                    self.buffer[start_idx:],
                    self.buffer[:second_part]
                ])

    def get_padded_audio(
        self,
        base_audio: Optional[np.ndarray] = None,
        pre_pad_seconds: Optional[float] = None,
        post_pad_seconds: Optional[float] = None,
        current_speech: Optional[np.ndarray] = None,
        pre_ms: Optional[int] = None,
        post_ms: Optional[int] = None,
    ) -> np.ndarray:
        """
        Devuelve el audio de voz con pre-roll extraído exactamente del búfer antes del inicio
        del habla y post-roll (hangover) para evitar truncar fonemas.
        """
        audio = current_speech if current_speech is not None else base_audio
        if audio is None or len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        pre_s = (pre_ms / 1000.0) if pre_ms is not None else (pre_pad_seconds if pre_pad_seconds is not None else 0.30)
        post_s = (post_ms / 1000.0) if post_ms is not None else (post_pad_seconds if post_pad_seconds is not None else 0.30)

        pre_samples = int(pre_s * self.sample_rate)
        post_samples = int(post_s * self.sample_rate)

        with self.lock:
            available_history = min(self.total_samples_written - len(audio), self.capacity_samples)
            actual_pre = min(pre_samples, max(0, available_history))

            if actual_pre > 0:
                speech_start_pos = (self.write_pos - len(audio)) % self.capacity_samples
                pre_start_pos = (speech_start_pos - actual_pre) % self.capacity_samples
                if pre_start_pos + actual_pre <= self.capacity_samples:
                    pre_buf = self.buffer[pre_start_pos : pre_start_pos + actual_pre].copy()
                else:
                    first = self.capacity_samples - pre_start_pos
                    second = actual_pre - first
                    pre_buf = np.concatenate([self.buffer[pre_start_pos:], self.buffer[:second]])
                if len(pre_buf) < pre_samples:
                    pad_zeros = np.zeros(pre_samples - len(pre_buf), dtype=np.float32)
                    pre_buf = np.concatenate([pad_zeros, pre_buf])
            else:
                pre_buf = np.zeros(pre_samples, dtype=np.float32)

        post_buf = np.zeros(post_samples, dtype=np.float32)
        return np.concatenate([pre_buf, audio, post_buf])

    def clear(self) -> None:
        """Reinicia el búfer."""
        with self.lock:
            self.buffer.fill(0.0)
            self.write_pos = 0
            self.total_samples_written = 0

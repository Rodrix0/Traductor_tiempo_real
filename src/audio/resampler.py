"""
resampler.py
Resampler de streaming con memoria de estado entre bloques (StatefulResampler).
Evita discontinuidades, chasquidos y aliasing en capturas continuas de 20-32ms.
Soporta tasas nativas arbitrarias (44.1kHz, 48kHz, 96kHz, 192kHz) hacia 16kHz mono float32.
"""

import math
from typing import Union, Optional
import numpy as np
from scipy.signal import resample_poly


class StatefulResampler:
    """Resampler continuo con persistencia de remanentes entre bloques de audio."""

    def __init__(
        self,
        in_rate: Optional[int] = None,
        out_rate: Optional[int] = None,
        channels: int = 2,
        input_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
    ):
        effective_in = input_rate if input_rate is not None else in_rate
        effective_out = target_rate if target_rate is not None else out_rate
        self.in_rate = int(effective_in if effective_in is not None else 48000)
        self.out_rate = int(effective_out if effective_out is not None else 16000)
        self.channels = int(channels)
        gcd = math.gcd(self.in_rate, self.out_rate)
        self.up = self.out_rate // gcd
        self.down = self.in_rate // gcd
        self.remainder = np.empty(0, dtype=np.float32)

    def resample(self, raw_bytes_or_array: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Convierte un bloque de audio de entrada a mono float32 a la tasa objetivo (16kHz).
        Mantiene el estado de muestras residuales entre llamadas consecutivas.
        """
        if isinstance(raw_bytes_or_array, bytes):
            samples = np.frombuffer(raw_bytes_or_array, dtype=np.float32)
            if self.channels > 1:
                samples = samples.reshape(-1, self.channels).mean(axis=1)
        else:
            samples = np.asarray(raw_bytes_or_array, dtype=np.float32)
            if samples.ndim > 1 and samples.shape[1] > 1:
                samples = samples.mean(axis=1)

        if len(samples) == 0:
            return np.empty(0, dtype=np.float32)

        # Si ya está en la tasa requerida
        if self.in_rate == self.out_rate:
            return samples.astype(np.float32)

        # Concatenar remanente del bloque anterior si existe
        if len(self.remainder) > 0:
            samples = np.concatenate([self.remainder, samples])
            self.remainder = np.empty(0, dtype=np.float32)

        # Optimización para múltiplos enteros exactos (ej. 48k -> 16k, factor 3)
        if self.in_rate % self.out_rate == 0:
            factor = self.in_rate // self.out_rate
            rem_len = len(samples) % factor
            if rem_len > 0:
                self.remainder = samples[-rem_len:]
                samples = samples[:-rem_len]
            if len(samples) == 0:
                return np.empty(0, dtype=np.float32)
            return samples.reshape(-1, factor).mean(axis=1).astype(np.float32)

        # Resampling polifase para tasas no enteras (ej. 44.1k -> 16k)
        rem_len = len(samples) % self.down
        if rem_len > 0:
            self.remainder = samples[-rem_len:]
            samples = samples[:-rem_len]
        if len(samples) == 0:
            return np.empty(0, dtype=np.float32)

        return resample_poly(samples, self.up, self.down).astype(np.float32)

    def reset(self) -> None:
        """Limpia el buffer de remanente."""
        self.remainder = np.empty(0, dtype=np.float32)

    process = resample


def to_mono_16k(
    raw: Union[bytes, np.ndarray],
    channels: Optional[int] = None,
    rate: Optional[int] = None,
) -> np.ndarray:
    """Función de compatibilidad global para remuestreo mono float32 a 16kHz."""
    # Si se llamó como to_mono_16k(raw, rate) con solo 2 argumentos
    if rate is None and isinstance(channels, int) and channels > 32:
        effective_rate = channels
        effective_channels = 2 if getattr(raw, "ndim", 1) > 1 and raw.shape[1] > 1 else 1
    else:
        effective_rate = rate if rate is not None else 48000
        effective_channels = channels if channels is not None else (2 if getattr(raw, "ndim", 1) > 1 and raw.shape[1] > 1 else 1)

    resampler = StatefulResampler(in_rate=effective_rate, out_rate=16000, channels=effective_channels)
    return resampler.resample(raw)


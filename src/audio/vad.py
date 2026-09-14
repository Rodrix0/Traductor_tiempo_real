"""
Detector de Actividad Vocal (VAD) y segmentador de frases.
Agrupa frames de audio en frases completas detectando el inicio de voz y los silencios naturales.
"""

from collections import deque
from typing import Optional, List, Union
import numpy as np

from config.settings import (
    SAMPLE_RATE,
    ENERGY_THRESHOLD,
    SILENCE_DURATION_MS,
    MIN_SPEECH_DURATION_MS,
    MAX_SPEECH_DURATION_MS,
    PRE_SPEECH_PADDING_MS,
)
from src.pipeline.models import VADProfile


class VoiceActivityDetector:
    """
    Segmentador de voz basado en energía y silencios con buffer circular de pre-voz.
    Permite detectar frases naturales y emitir el audio completo listo para transcribir.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        energy_threshold: float = ENERGY_THRESHOLD,
        silence_duration_ms: Optional[int] = None,
        min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
        max_speech_duration_ms: int = MAX_SPEECH_DURATION_MS,
        pre_speech_padding_ms: int = PRE_SPEECH_PADDING_MS,
        profile: Optional[Union[VADProfile, str]] = None,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold

        if profile is not None:
            self.profile = VADProfile(profile)
            self.silence_duration_ms = self.profile.silence_duration_ms
        else:
            self.profile = VADProfile.NATURAL
            self.silence_duration_ms = silence_duration_ms if silence_duration_ms is not None else SILENCE_DURATION_MS

        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms

        # Buffer circular para conservar audio inmediatamente anterior al inicio de voz (pre-roll 200-300ms)
        # evitando cortar la primera sílaba
        pre_speech_samples = int(sample_rate * (pre_speech_padding_ms / 1000.0))
        self.pre_speech_buffer: deque = deque(maxlen=max(1, pre_speech_samples // 480))

        self.current_speech_frames: List[np.ndarray] = []
        self.is_speaking = False
        self.silence_samples_count = 0
        self.speech_samples_count = 0
        self.last_finalize_reason: Optional[str] = None
        self.last_trailing_silence_ms: int = 0
        # Atomic snapshot in audio time, independent of ASR/queue latency.
        self.acoustic_progress = (0.0, 0.0)
        self._processed_samples = 0
        self._last_voice_sample = 0

    @staticmethod
    def calculate_energy(frame: np.ndarray) -> float:
        """Calcula el valor RMS (Root Mean Square) de energía de un frame de audio."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame**2)))

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un frame continuo de audio.
        Retorna un array numpy con el audio completo de la frase cuando detecta fin de habla,
        o None si aún se está escuchando/acumulando.
        """
        frame_len = len(frame)
        if frame_len == 0:
            return None

        energy = self.calculate_energy(frame)
        frame_is_speech = energy >= self.energy_threshold
        self._processed_samples += frame_len
        if frame_is_speech:
            self._last_voice_sample = self._processed_samples
        self.acoustic_progress = (
            self._processed_samples / self.sample_rate,
            self._last_voice_sample / self.sample_rate,
        )

        # Estado 1: Actualmente en silencio (esperando que alguien comience a hablar)
        if not self.is_speaking:
            if frame_is_speech:
                # Comienza la voz: incluir buffer pre-voz para no perder consonantes/sílabas iniciales
                self.is_speaking = True
                self.current_speech_frames = list(self.pre_speech_buffer)
                self.current_speech_frames.append(frame)
                self.speech_samples_count = sum(len(f) for f in self.current_speech_frames)
                self.silence_samples_count = 0
                self.pre_speech_buffer.clear()
            else:
                self.pre_speech_buffer.append(frame)
            return None

        # Estado 2: Actualmente hablando (acumulando frames de la frase)
        self.current_speech_frames.append(frame)
        self.speech_samples_count += frame_len

        if frame_is_speech:
            # Resetea el contador de silencio continuo
            self.silence_samples_count = 0
        else:
            self.silence_samples_count += frame_len

        silence_ms = (self.silence_samples_count / self.sample_rate) * 1000.0
        speech_ms = (self.speech_samples_count / self.sample_rate) * 1000.0

        # Caso A: El silencio posterior superó el umbral configurado (ej. 700ms)
        if silence_ms >= self.silence_duration_ms:
            return self._finalize_speech_segment(speech_ms, reason="silence", trailing_silence_ms=int(silence_ms))

        # Caso B: El usuario habló continuamente durante mucho tiempo
        # Cortar de forma inteligente en el valle de menor energía para no cortar palabras
        if speech_ms >= self.max_speech_duration_ms:
            return self._split_at_energy_valley()

        return None

    def _split_at_energy_valley(self) -> Optional[np.ndarray]:
        """Corta de forma inteligente en el punto de menor energía (valle) para no cortar palabras."""
        if not self.current_speech_frames:
            return None

        total_frames = len(self.current_speech_frames)
        frame_len = len(self.current_speech_frames[0]) if total_frames > 0 else 480
        search_window = min(int((1.5 * self.sample_rate) / frame_len), max(2, total_frames // 2))

        if total_frames < 4 or search_window < 2:
            return self._finalize_speech_segment((self.speech_samples_count / self.sample_rate) * 1000.0, reason="max_duration")

        start_idx = total_frames - search_window
        energies = [self.calculate_energy(f) for f in self.current_speech_frames[start_idx:]]
        min_idx = int(np.argmin(energies))
        split_point = start_idx + min_idx + 1

        emitted_frames = self.current_speech_frames[:split_point]
        remaining_frames = self.current_speech_frames[split_point:]

        audio_segment = np.concatenate(emitted_frames) if emitted_frames else None

        # Mantener los frames restantes como inicio del siguiente fragmento para continuidad
        self.current_speech_frames = remaining_frames
        self.speech_samples_count = sum(len(f) for f in remaining_frames)
        self.silence_samples_count = 0
        self.last_finalize_reason = "max_duration"
        self.last_trailing_silence_ms = 0

        return audio_segment

    def _finalize_speech_segment(self, speech_ms: float, reason: str = "silence", trailing_silence_ms: int = 0) -> Optional[np.ndarray]:
        """Finaliza y resetea el segmento actual de voz."""
        audio_segment = None
        if speech_ms >= self.min_speech_duration_ms and self.current_speech_frames:
            audio_segment = np.concatenate(self.current_speech_frames)

        # Resetear estado
        self.is_speaking = False
        self.current_speech_frames = []
        self.speech_samples_count = 0
        self.silence_samples_count = 0
        self.pre_speech_buffer.clear()
        self.last_finalize_reason = reason
        self.last_trailing_silence_ms = max(0, int(trailing_silence_ms))

        return audio_segment

    def reset(self) -> None:
        """Reinicia el estado del detector."""
        self.is_speaking = False
        self.current_speech_frames.clear()
        self.pre_speech_buffer.clear()
        self.speech_samples_count = 0
        self.silence_samples_count = 0
        self.last_finalize_reason = None
        self.last_trailing_silence_ms = 0
        self._processed_samples = 0
        self._last_voice_sample = 0
        self.acoustic_progress = (0.0, 0.0)

    def set_profile(self, profile: Union[VADProfile, str]) -> None:
        """Cambia dinámicamente el perfil VAD y actualiza la ventana de silencio."""
        self.profile = VADProfile(profile) if isinstance(profile, str) else profile
        self.silence_duration_ms = self.profile.silence_duration_ms

    def flush(self) -> Optional[np.ndarray]:
        """Finaliza y emite cualquier segmento de voz que haya quedado abierto en el buffer."""
        if self.is_speaking and self.current_speech_frames:
            speech_ms = (self.speech_samples_count / self.sample_rate) * 1000.0
            return self._finalize_speech_segment(speech_ms, reason="shutdown")
        return None

    process_chunk = process_frame
